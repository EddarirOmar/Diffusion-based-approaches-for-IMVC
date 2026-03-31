"""
Multi-view dataset loaders for IMVCDC.

Supports CARL, CUB, and Synthetic3D datasets with view masking
for IMVC (Incomplete Multi-View Clustering) simulation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from pathlib import Path
from typing import Tuple, Optional, List


class MultiViewDataset(Dataset):
    """
    Base class for multi-view datasets with view masking capability.
    
    Args:
        data_path: Path to dataset file (.mat format)
        dataset_name: Name of dataset (e.g., 'CARL', 'CUB', 'Synthetic3D')
        missing_rate: Fraction of views to mask (0.0 to 1.0)
        normalize: If True, normalize data to [0, 1]
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        data_path: str,
        dataset_name: str = "CARL",
        missing_rate: float = 0.0,
        normalize: bool = True,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.missing_rate = missing_rate
        self.normalize = normalize
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load data
        self._load_data()
        
        # Generate view masks for missing views
        self._generate_masks()
    
    def _load_data(self):
        """Load dataset from .mat file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        print(f"Loading {self.dataset_name} dataset from {self.data_path}...")
        mat_data = sio.loadmat(str(self.data_path))
        
        # Extract views from common MATLAB storage layouts.
        self.views = []
        if "X" in mat_data:
            x = mat_data["X"]
            if isinstance(x, np.ndarray) and x.dtype == object:
                for idx in np.ndindex(x.shape):
                    arr = x[idx]
                    if isinstance(arr, np.ndarray) and arr.ndim == 2:
                        self.views.append(arr)
            elif isinstance(x, np.ndarray) and x.ndim == 2:
                self.views.append(x)

        # Some datasets expose each view as fea1, fea2, ...
        if not self.views and "fea1" in mat_data and "fea2" in mat_data:
            fea_keys = sorted([k for k in mat_data.keys() if k.startswith("fea")])
            self.views = [mat_data[k] for k in fea_keys if isinstance(mat_data[k], np.ndarray)]

        # Fallback: use all 2D numeric arrays with matching sample count.
        if not self.views:
            candidates = []
            for k, v in mat_data.items():
                if k.startswith("_") or not isinstance(v, np.ndarray) or v.ndim != 2:
                    continue
                if v.dtype == object:
                    continue
                candidates.append((k, v))
            if candidates:
                sample_count = max(v.shape[0] for _, v in candidates)
                self.views = [v for _, v in candidates if v.shape[0] == sample_count and v.shape[1] > 1]
        
        # Extract labels if available
        if 'y' in mat_data:
            self.labels = mat_data['y'].flatten()
        elif 'labels' in mat_data:
            self.labels = mat_data['labels'].flatten()
        elif 'Y' in mat_data and isinstance(mat_data['Y'], np.ndarray) and mat_data['Y'].shape[1] == 1:
            self.labels = mat_data['Y'].flatten()
        elif 'gt' in mat_data:
            self.labels = mat_data['gt'].flatten()
        else:
            self.labels = None

        if not self.views:
            raise ValueError("No valid views found in dataset file.")
        
        # Convert to float32 and normalize
        self.views = [v.astype(np.float32) for v in self.views]
        self.n_views = len(self.views)
        self.n_samples = self.views[0].shape[0]
        self.view_dims = [v.shape[1] for v in self.views]
        self.max_view_dim = max(self.view_dims)
        
        if self.normalize:
            self._normalize_views()

        # Pad features so all views can be stacked in a single tensor.
        self._pad_views_to_same_dim()
        
        print(f"  - Loaded {self.n_samples} samples, {self.n_views} views")
        print(f"  - View dimensions: {self.view_dims}")
        print(f"  - Padded view dimension: {self.max_view_dim}")

    def _pad_views_to_same_dim(self):
        """Pad each view to the maximum feature dimension across views."""
        padded_views = []
        for view in self.views:
            if view.shape[1] == self.max_view_dim:
                padded_views.append(view)
                continue
            pad_width = self.max_view_dim - view.shape[1]
            padded = np.pad(view, ((0, 0), (0, pad_width)), mode="constant")
            padded_views.append(padded)
        self.views = padded_views
    
    def _normalize_views(self):
        """Normalize each view to [0, 1] range."""
        for i, view in enumerate(self.views):
            v_min = view.min()
            v_max = view.max()
            if v_max > v_min:
                self.views[i] = (view - v_min) / (v_max - v_min)
            else:
                self.views[i] = np.zeros_like(view)
    
    def _generate_masks(self):
        """Generate binary masks for missing views per sample."""
        self.view_masks = np.ones((self.n_samples, self.n_views), dtype=np.float32)
        
        if self.missing_rate > 0:
            n_missing = max(1, int(self.n_views * self.missing_rate))
            for i in range(self.n_samples):
                missing_views = np.random.choice(
                    self.n_views, size=n_missing, replace=False
                )
                self.view_masks[i, missing_views] = 0.0
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample with all views and masks.
        
        Returns:
            dict with keys:
                - 'views': tensor of shape (n_views, dim)
                - 'view_mask': tensor of shape (n_views,) indicating available views
                - 'index': sample index
                - 'label': class label (if available)
        """
        sample = {
            'views': torch.stack([
                torch.from_numpy(self.views[i][idx])
                for i in range(self.n_views)
            ]),  # shape: (n_views, view_dim)
            'view_mask': torch.from_numpy(self.view_masks[idx]),  # shape: (n_views,)
            'index': idx,
        }
        
        if self.labels is not None:
            sample['label'] = torch.from_numpy(np.array(self.labels[idx], dtype=np.int64))
        
        return sample


class CARLDataset(MultiViewDataset):
    """CARL multi-view dataset (Object views + Textual features)."""
    
    def __init__(self, data_path: str = None, **kwargs):
        if data_path is None:
            data_path = "data/raw/DCG/cub_googlenet_doc2vec_c10.mat"
        super().__init__(data_path=data_path, dataset_name="CARL", **kwargs)


class Synthetic3DDataset(MultiViewDataset):
    """Synthetic 3D multi-view dataset."""
    
    def __init__(self, data_path: str = None, **kwargs):
        if data_path is None:
            data_path = "data/raw/DCG/synthetic3d.mat"
        super().__init__(data_path=data_path, dataset_name="Synthetic3D", **kwargs)


def get_dataloader(
    dataset_name: str,
    data_path: str,
    batch_size: int = 64,
    missing_rate: float = 0.3,
    test_split: float = 0.2,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        dataset_name: One of 'CARL', 'CUB', 'Synthetic3D'
        data_path: Path to dataset file
        batch_size: Batch size
        missing_rate: Fraction of views to mask
        test_split: Fraction of data for testing
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle training data
        seed: Random seed
    
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Load full dataset
    dataset = MultiViewDataset(
        data_path=data_path,
        dataset_name=dataset_name,
        missing_rate=missing_rate,
        normalize=True,
        seed=seed,
    )
    
    # Split into train/test
    n_test = int(len(dataset) * test_split)
    n_train = len(dataset) - n_test
    
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    
    try:
        dataset = MultiViewDataset(
            data_path="../DCG/data/synthetic3d.mat",
            dataset_name="Synthetic3D",
            missing_rate=0.3,
            normalize=True,
        )
        print(f"[OK] Loaded dataset with {len(dataset)} samples")
        
        sample = dataset[0]
        print(f"[OK] Sample keys: {sample.keys()}")
        print(f"[OK] Views shape: {sample['views'].shape}")
        print(f"[OK] View mask: {sample['view_mask']}")
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
