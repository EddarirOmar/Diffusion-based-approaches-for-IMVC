import itertools

import numpy as np
import torch
import torch.nn.functional as F

from baseModels import Autoencoder, AttentionLayer, ClusterProject, NoiseScheduler, Unet
from evaluation import evaluation
from loss import ClusterLoss, EPS, InstanceLoss, MMI
from util import target_l2


class icdm:

    def __init__(self, config):
        self._config = config

        ae_cfg = config['Autoencoder']
        diff_cfg = config['diffusion']

        archs = ae_cfg.get('archs')
        activations = ae_cfg.get('activations')
        out_dims = diff_cfg.get('out_dims')

        # Backward compatibility with the original 2-view config keys.
        if archs is None:
            archs = [ae_cfg['arch1'], ae_cfg['arch2']]
        if activations is None:
            activations = [ae_cfg['activations1'], ae_cfg['activations2']]
        if out_dims is None:
            out_dims = [diff_cfg['out_dim1'], diff_cfg['out_dim2']]

        if len(archs) < 2:
            raise ValueError('Need at least two views for clustering.')
        if len(activations) != len(archs):
            raise ValueError('Autoencoder activations must match number of views.')
        if len(out_dims) != len(archs):
            raise ValueError('Diffusion out_dims must match number of views.')

        latent_dims = [arch[-1] for arch in archs]
        if len(set(latent_dims)) != 1:
            raise ValueError('Inconsistent latent dim across views.')

        self.n_views = len(archs)
        self._latent_dim = latent_dims[0]

        self.autoencoders = torch.nn.ModuleList(
            [Autoencoder(archs[i], activations[i], ae_cfg['batchnorm']) for i in range(self.n_views)]
        )
        self.dfs = torch.nn.ModuleList(
            [
                Unet(diff_cfg['emb_size'], diff_cfg['time_type'], out_dims[i])
                for i in range(self.n_views)
            ]
        )

        # Legacy aliases to minimize breakage for existing scripts.
        self.autoencoder1 = self.autoencoders[0]
        self.autoencoder2 = self.autoencoders[1]
        self.df1 = self.dfs[0]
        self.df2 = self.dfs[1]

        self.noise_scheduler = NoiseScheduler(
            num_timesteps=config['noise_scheduler']['num_timesteps'],
            beta_schedule=config['noise_scheduler'].get('beta_schedule', 'linear')
        )
        self.clusterLayer = ClusterProject(self._latent_dim, config['training']['n_clusters'])
        self.AttentionLayer = AttentionLayer(self._latent_dim, n_views=self.n_views)

    def to_device(self, device):
        self.autoencoders.to(device)
        self.dfs.to(device)
        self.clusterLayer.to(device)
        self.AttentionLayer.to(device)

    def checkpoint_state(self):
        """Return a serializable model state for checkpointing."""
        return {
            'autoencoders': self.autoencoders.state_dict(),
            'dfs': self.dfs.state_dict(),
            'clusterLayer': self.clusterLayer.state_dict(),
            'AttentionLayer': self.AttentionLayer.state_dict(),
            'n_views': self.n_views,
            'latent_dim': self._latent_dim,
        }

    def load_checkpoint_state(self, state):
        """Load model state from a checkpoint payload."""
        self.autoencoders.load_state_dict(state['autoencoders'])
        self.dfs.load_state_dict(state['dfs'])
        self.clusterLayer.load_state_dict(state['clusterLayer'])
        self.AttentionLayer.load_state_dict(state['AttentionLayer'])

    def _reverse_diffuse(self, latent, denoiser, device):
        out = latent
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        for t in timesteps:
            tvec = torch.full((out.shape[0],), t, dtype=torch.long, device=device)
            with torch.no_grad():
                pred = denoiser(out, tvec)
            out = self.noise_scheduler.step(pred, t, out)
        return out

    def _pairwise_mean(self, values):
        if not values:
            return torch.tensor(0.0)
        return sum(values) / len(values)

    def train(self, config, x_train_list, Y_list, mask, optimizer, device):
        criterion_cluster = ClusterLoss(config['training']['n_clusters'], 0.5, device).to(device)
        train_cfg = config['training']
        lambda_rec = float(train_cfg.get('lambda_rec', 1.0))
        lambda_df = float(train_cfg.get('lambda_df', 1.0))
        lambda_ce = float(train_cfg.get('lambda_ce', 1.0))
        lambda_mmi = float(train_cfg.get('lambda_mmi', train_cfg.get('lambda1', 1.0)))
        lambda_cluster = float(train_cfg.get('lambda_cluster', train_cfg.get('lambda2', 1.0)))
        lambda_hc = float(train_cfg.get('lambda_hc', lambda_cluster))
        mmi_internal_lambda = float(train_cfg.get('mmi_internal_lambda', 1.0))
        mmi_temperature = float(train_cfg.get('mmi_temperature', 1.0))

        best_acc, best_nmi, best_ari = 0, 0, 0
        n_samples = x_train_list[0].shape[0]

        for epoch in range(config['training']['epoch'] + 1):
            perm = torch.randperm(n_samples, device=device)
            x_shuffled = [x[perm] for x in x_train_list]
            mask_shuffled = mask[perm]

            loss_all, loss_rec, loss_mmi, loss_df, loss_cluster, loss_hc, loss_ce = 0, 0, 0, 0, 0, 0, 0
            valid_batches = 0

            for start in range(0, n_samples, config['training']['batch_size']):
                end = min(start + config['training']['batch_size'], n_samples)
                batch_views = [x[start:end] for x in x_shuffled]
                batch_mask = mask_shuffled[start:end]

                if end - start <= 1:
                    continue

                complete_idx = (batch_mask.sum(dim=1) == self.n_views)
                complete_count = int(complete_idx.sum().item())
                if complete_count <= 1:
                    continue

                z_complete = [
                    self.autoencoders[v].encoder(batch_views[v][complete_idx])
                    for v in range(self.n_views)
                ]

                rec_terms = [
                    F.mse_loss(self.autoencoders[v].decoder(z_complete[v]), batch_views[v][complete_idx])
                    for v in range(self.n_views)
                ]
                rec_loss = sum(rec_terms)

                h_complete = self.AttentionLayer(*z_complete)
                mmi_terms = [
                    MMI(h_complete, z_complete[v], lamb=mmi_internal_lambda, temperature=mmi_temperature)
                    for v in range(self.n_views)
                ]
                mmi_loss = sum(mmi_terms)

                y_views = [self.clusterLayer(z_complete[v])[0] for v in range(self.n_views)]
                cluster_terms = []
                for i, j in itertools.combinations(range(self.n_views), 2):
                    cluster_terms.append(criterion_cluster(y_views[i], y_views[j]))
                cluster_loss = sum(cluster_terms) / max(1, len(cluster_terms))

                y_fused, _ = self.clusterLayer(h_complete)
                y_stack = torch.stack(y_views + [y_fused], dim=0)
                y_max = target_l2(torch.max(y_stack, dim=0).values)
                y_fused = torch.where(y_fused < EPS, torch.tensor([EPS], device=y_fused.device), y_fused)
                hc_loss = F.kl_div(y_fused.log(), y_max.detach(), reduction='batchmean')

                df_terms = []
                for v in range(self.n_views):
                    obs_idx = (batch_mask[:, v] == 1)
                    obs_count = int(obs_idx.sum().item())
                    if obs_count == 0:
                        continue
                    z_obs = self.autoencoders[v].encoder(batch_views[v][obs_idx])
                    noise = torch.randn_like(z_obs)
                    timesteps = torch.randint(
                        0, config['noise_scheduler']['num_timesteps'], (z_obs.shape[0],), device=device
                    ).long()
                    noisy = self.noise_scheduler.add_noise(z_obs, noise, timesteps, device)
                    noise_pred = self.dfs[v](noisy, timesteps)
                    df_terms.append(F.mse_loss(noise_pred, noise))
                dfloss = sum(df_terms) if df_terms else torch.tensor(0.0, device=device)

                ce_loss = torch.tensor(0.0, device=device)
                if complete_count > 1:
                    criterion_instance = InstanceLoss(complete_count, 1.0, device).to(device)
                    recovered = [self._reverse_diffuse(z_complete[v], self.dfs[v], device) for v in range(self.n_views)]
                    ce_terms = []
                    for i, j in itertools.combinations(range(self.n_views), 2):
                        ce_terms.append(criterion_instance(recovered[i], z_complete[j]))
                        ce_terms.append(criterion_instance(recovered[j], z_complete[i]))
                    if ce_terms:
                        ce_loss = sum(ce_terms) / len(ce_terms)

                loss = (
                    lambda_rec * rec_loss
                    + lambda_df * dfloss
                    + lambda_ce * ce_loss
                    + lambda_mmi * mmi_loss
                    + lambda_cluster * cluster_loss
                    + lambda_hc * hc_loss
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                valid_batches += 1
                loss_all += loss.item()
                loss_rec += rec_loss.item()
                loss_df += dfloss.item()
                loss_mmi += mmi_loss.item()
                loss_cluster += cluster_loss.item()
                loss_hc += hc_loss.item()
                loss_ce += ce_loss.item()

            if (epoch) % config['print_num'] == 0:
                denom = max(1, valid_batches)
                output = (
                    'Epoch: {:.0f}/{:.0f} ==> loss = {:.4f} '
                    '| rec={:.4f} df={:.4f} ce={:.4f} mmi={:.4f} clu={:.4f} hc={:.4f}'
                ).format(
                    epoch,
                    config['training']['epoch'],
                    loss_all,
                    loss_rec / denom,
                    loss_df / denom,
                    loss_ce / denom,
                    loss_mmi / denom,
                    loss_cluster / denom,
                    loss_hc / denom,
                )
                print(output)

                scores = self.evaluation(config, mask, x_train_list, Y_list, device)
                if scores['accuracy'] >= best_acc:
                    best_acc = scores['accuracy']
                    best_nmi = scores['NMI']
                    best_ari = scores['ARI']

        return best_acc, best_nmi, best_ari

    def evaluation(self, config, mask, x_train_list, Y_list, device):
        with torch.no_grad():
            self.autoencoders.eval()
            self.dfs.eval()

            n_samples = x_train_list[0].shape[0]
            latent_eval = [
                torch.zeros(n_samples, self._latent_dim, device=device)
                for _ in range(self.n_views)
            ]

            for v in range(self.n_views):
                observed = (mask[:, v] == 1)
                if int(observed.sum().item()) > 0:
                    latent_eval[v][observed] = self.autoencoders[v].encoder(x_train_list[v][observed])

            # Recover each missing view latent using mean latent from other observed views.
            for v in range(self.n_views):
                missing = (mask[:, v] == 0)
                miss_count = int(missing.sum().item())
                if miss_count == 0:
                    continue

                seed_latent = torch.zeros(miss_count, self._latent_dim, device=device)
                contrib_count = torch.zeros(miss_count, 1, device=device)

                for u in range(self.n_views):
                    if u == v:
                        continue
                    source_obs = (mask[missing, u] == 1)
                    if int(source_obs.sum().item()) == 0:
                        continue
                    src_encoded = self.autoencoders[u].encoder(x_train_list[u][missing][source_obs])
                    seed_latent[source_obs] += src_encoded
                    contrib_count[source_obs] += 1.0

                has_source = contrib_count.squeeze(1) > 0
                if int(has_source.sum().item()) > 0:
                    seed_latent[has_source] = seed_latent[has_source] / contrib_count[has_source]

                no_source = ~has_source
                if int(no_source.sum().item()) > 0:
                    seed_latent[no_source] = torch.randn(int(no_source.sum().item()), self._latent_dim, device=device)

                recovered = self._reverse_diffuse(seed_latent, self.dfs[v], device)
                latent_eval[v][missing] = recovered

            latent_fusion = self.AttentionLayer(*latent_eval)
            y, _ = self.clusterLayer(latent_fusion)
            y = y.data.cpu().numpy().argmax(1)
            scores = evaluation(y_pred=y, y_true=Y_list[0])

            self.autoencoders.train()
            self.dfs.train()

        return scores
