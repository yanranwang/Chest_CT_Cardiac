#!/usr/bin/env python3
"""
Fast data loader - Efficiently read from HDF5 preprocessed data

This module provides efficient data loaders that directly read data from preprocessed HDF5 files,
significantly reducing training data loading time and improving GPU utilization.

Main features:
1. Direct reading from HDF5 preprocessed data
2. Support memory caching and preloading
3. Efficient multi-process data reading
4. Support data augmentation (optional)
5. Intelligent data splitting
"""

import os
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split
import threading
import queue
import warnings
warnings.filterwarnings('ignore')


class FastCardiacDataset(Dataset):
    """Fast cardiac function dataset - Read preprocessed data from HDF5 files"""
    
    def __init__(self, 
                 hdf5_path: str, 
                 metadata_path: str,
                 item_ids: List[str],
                 enable_cache: bool = True,
                 cache_size: int = 1000,
                 preload_data: bool = False):
        """
        Initialize fast dataset
        
        Args:
            hdf5_path: HDF5 file path
            metadata_path: Metadata file path
            item_ids: List of data item IDs
            enable_cache: Whether to enable memory cache
            cache_size: Cache size
            preload_data: Whether to preload all data to memory
        """
        self.hdf5_path = Path(hdf5_path)
        self.metadata_path = Path(metadata_path)
        self.item_ids = item_ids
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.preload_data = preload_data
        
        # Validate file existence
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load metadata index
        self.metadata_index = self._load_metadata_index()
        
        # Validate item IDs
        self._validate_item_ids()
        
        # Initialize cache
        self._cache = {} if enable_cache else None
        self._cache_lock = threading.Lock() if enable_cache else None
        
        # Preload data
        if preload_data:
            self._preload_all_data()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _load_metadata_index(self) -> Dict[str, Any]:
        """Load metadata index"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _validate_item_ids(self):
        """Validate item IDs"""
        available_ids = set(self.metadata_index['items'].keys())
        missing_ids = set(self.item_ids) - available_ids
        
        if missing_ids:
            raise ValueError(f"Following item IDs not found: {missing_ids}")
    
    def _preload_all_data(self):
        """Preload all data to memory"""
        self.logger.info(f"Preloading {len(self.item_ids)} data items...")
        
        self._preloaded_data = {}
        
        with h5py.File(self.hdf5_path, 'r') as f:
            images_group = f['images']
            metadata_group = f['metadata']
            
            for item_id in self.item_ids:
                # Load image data
                image_data = images_group[item_id][:]
                
                # Load metadata
                metadata_str = metadata_group[item_id][()]
                if isinstance(metadata_str, bytes):
                    metadata_str = metadata_str.decode('utf-8')
                metadata = json.loads(metadata_str)
                
                self._preloaded_data[item_id] = {
                    'image': image_data,
                    'metadata': metadata
                }
        
        self.logger.info("Data preloading completed")
    
    def _get_from_cache(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get data from cache"""
        if not self.enable_cache:
            return None
        
        with self._cache_lock:
            return self._cache.get(item_id)
    
    def _add_to_cache(self, item_id: str, data: Dict[str, Any]):
        """Add data to cache"""
        if not self.enable_cache:
            return
        
        with self._cache_lock:
            if len(self._cache) >= self.cache_size:
                # Remove oldest cache item
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[item_id] = data
    
    def _load_item_from_hdf5(self, item_id: str) -> Dict[str, Any]:
        """Load data item from HDF5 file"""
        if self.preload_data:
            return self._preloaded_data[item_id]
        
        # Check cache
        cached_data = self._get_from_cache(item_id)
        if cached_data is not None:
            return cached_data
        
        # Load from HDF5 file
        with h5py.File(self.hdf5_path, 'r') as f:
            images_group = f['images']
            metadata_group = f['metadata']
            
            # Load image data
            image_data = images_group[item_id][:]
            
            # Load metadata
            metadata_str = metadata_group[item_id][()]
            if isinstance(metadata_str, bytes):
                metadata_str = metadata_str.decode('utf-8')
            metadata = json.loads(metadata_str)
            
            data = {
                'image': image_data,
                'metadata': metadata
            }
            
            # Add to cache
            self._add_to_cache(item_id, data)
            
            return data
    
    def __len__(self) -> int:
        return len(self.item_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_id = self.item_ids[idx]
        data = self._load_item_from_hdf5(item_id)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(data['image']).float()
        
        # Get cardiac function metrics
        cardiac_metrics = data['metadata']['cardiac_metrics']
        if cardiac_metrics is not None:
            cardiac_metrics = torch.tensor(cardiac_metrics, dtype=torch.float32)
        else:
            # Generate dummy labels
            cardiac_metrics = torch.tensor(self._generate_dummy_labels(), dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'cardiac_metrics': cardiac_metrics,
            'patient_id': data['metadata']['patient_id'],
            'basename': data['metadata']['basename'],
            'folder': data['metadata']['folder']
        }
    
    def _generate_dummy_labels(self) -> np.ndarray:
        """Generate dummy labels for demo"""
        # Generate LVEF (30-70) and AS (0 or 1)
        lvef = np.random.uniform(30, 70)
        as_label = np.float32(np.random.randint(0, 2))
        
        return np.array([lvef, as_label], dtype=np.float32)


class FastDataLoaderManager:
    """Fast data loader manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # File paths
        self.hdf5_path = Path(config['preprocessed_data_dir']) / 'preprocessed_data.h5'
        self.metadata_path = Path(config['preprocessed_data_dir']) / 'data_metadata.json'
        
        # Validate file existence
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"Preprocessed data file not found: {self.hdf5_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        # Load metadata
        self.metadata_index = self._load_metadata_index()
        
        self.logger.info(f"Loaded metadata index: {len(self.metadata_index['items'])} data items")
    
    def _load_metadata_index(self) -> Dict[str, Any]:
        """Load metadata index"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_all_item_ids(self) -> List[str]:
        """Get all data item IDs"""
        return list(self.metadata_index['items'].keys())
    
    def get_items_by_patient(self, patient_id: str) -> List[str]:
        """Get data items by patient ID"""
        return self.metadata_index['patient_mapping'].get(patient_id, [])
    
    def get_items_by_folder(self, folder: str) -> List[str]:
        """Get data items by folder"""
        return self.metadata_index['folder_mapping'].get(folder, [])
    
    def split_data(self, 
                   split_method: str = 'random',
                   train_ratio: float = 0.8,
                   random_state: int = 42) -> Tuple[List[str], List[str]]:
        """Split data into training and validation sets"""
        all_item_ids = self.get_all_item_ids()
        
        if split_method == 'random':
            # Random split
            train_ids, val_ids = train_test_split(
                all_item_ids,
                train_size=train_ratio,
                random_state=random_state,
                shuffle=True
            )
        
        elif split_method == 'patient_based':
            # Patient-based split
            all_patients = list(self.metadata_index['patient_mapping'].keys())
            train_patients, val_patients = train_test_split(
                all_patients,
                train_size=train_ratio,
                random_state=random_state,
                shuffle=True
            )
            
            train_ids = []
            val_ids = []
            
            for patient in train_patients:
                train_ids.extend(self.get_items_by_patient(patient))
            
            for patient in val_patients:
                val_ids.extend(self.get_items_by_patient(patient))
        
        elif split_method == 'sequential':
            # Sequential split
            split_idx = int(len(all_item_ids) * train_ratio)
            train_ids = all_item_ids[:split_idx]
            val_ids = all_item_ids[split_idx:]
        
        else:
            raise ValueError(f"Unsupported split method: {split_method}")
        
        self.logger.info(f"Data split completed: Training set {len(train_ids)}, Validation set {len(val_ids)}")
        return train_ids, val_ids
    
    def create_dataset(self, 
                      item_ids: List[str],
                      enable_cache: bool = True,
                      cache_size: int = 1000,
                      preload_data: bool = False) -> FastCardiacDataset:
        """Create dataset"""
        return FastCardiacDataset(
            hdf5_path=self.hdf5_path,
            metadata_path=self.metadata_path,
            item_ids=item_ids,
            enable_cache=enable_cache,
            cache_size=cache_size,
            preload_data=preload_data
        )
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""
        # Split data
        train_ids, val_ids = self.split_data(
            split_method=self.config.get('split_method', 'random'),
            train_ratio=self.config.get('train_val_split', 0.8),
            random_state=self.config.get('seed', 42)
        )
        
        # Create dataset
        cache_config = self.config.get('cache_config', {})
        
        train_dataset = self.create_dataset(
            item_ids=train_ids,
            enable_cache=cache_config.get('enable_cache', True),
            cache_size=cache_config.get('cache_size', 1000),
            preload_data=cache_config.get('preload_train_data', False)
        )
        
        val_dataset = self.create_dataset(
            item_ids=val_ids,
            enable_cache=cache_config.get('enable_cache', True),
            cache_size=cache_config.get('cache_size', 500),
            preload_data=cache_config.get('preload_val_data', False)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 4),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.config.get('num_workers', 4) > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 4),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.config.get('num_workers', 4) > 0 else False
        )
        
        return train_loader, val_loader
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get data statistics"""
        stats = self.metadata_index['statistics'].copy()
        
        # Add file size information
        hdf5_size = self.hdf5_path.stat().st_size / (1024 * 1024)  # MB
        metadata_size = self.metadata_path.stat().st_size / 1024  # KB
        
        stats.update({
            'hdf5_file_size_mb': round(hdf5_size, 2),
            'metadata_file_size_kb': round(metadata_size, 2),
            'data_loading_method': 'HDF5_Fast_Loading'
        })
        
        return stats


def create_fast_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create convenient function for fast data loaders"""
    manager = FastDataLoaderManager(config)
    return manager.create_data_loaders()


def benchmark_data_loading(config: Dict[str, Any], num_batches: int = 10) -> Dict[str, float]:
    """Benchmark data loading performance"""
    import time
    
    manager = FastDataLoaderManager(config)
    train_loader, val_loader = manager.create_data_loaders()
    
    # Test training data loading
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        # Simulated data usage
        _ = batch['image'].shape
    
    train_time = time.time() - start_time
    
    # Test validation data loading
    start_time = time.time()
    for i, batch in enumerate(val_loader):
        if i >= num_batches:
            break
        # Simulated data usage
        _ = batch['image'].shape
    
    val_time = time.time() - start_time
    
    return {
        'train_loading_time': train_time,
        'val_loading_time': val_time,
        'train_batches_per_second': num_batches / train_time,
        'val_batches_per_second': num_batches / val_time
    }


if __name__ == '__main__':
    # Test fast data loader
    config = {
        'preprocessed_data_dir': 'outputs/preprocessed_data',
        'batch_size': 4,
        'num_workers': 4,
        'split_method': 'random',
        'train_val_split': 0.8,
        'seed': 42,
        'cache_config': {
            'enable_cache': True,
            'cache_size': 1000,
            'preload_train_data': False,
            'preload_val_data': False
        }
    }
    
    try:
        # Create data loaders
        train_loader, val_loader = create_fast_data_loaders(config)
        
        print(f"Training set size: {len(train_loader.dataset)}")
        print(f"Validation set size: {len(val_loader.dataset)}")
        
        # Test data loading
        batch = next(iter(train_loader))
        print(f"Batch shape: {batch['image'].shape}")
        print(f"Cardiac function metrics shape: {batch['cardiac_metrics'].shape}")
        
        # Performance benchmark test
        print("\nStarting performance benchmark test...")
        benchmark_results = benchmark_data_loading(config)
        
        print("Benchmark test results:")
        for key, value in benchmark_results.items():
            print(f"  {key}: {value:.4f}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Please run data preprocessing script to generate HDF5 file first") 