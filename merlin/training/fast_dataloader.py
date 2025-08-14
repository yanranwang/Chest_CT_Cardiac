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
6. Hybrid data loading (CSV labels + HDF5 images)
"""

import os
import h5py
import json
import numpy as np
import pandas as pd
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


class HybridCardiacDataset(Dataset):
    """混合心脏功能数据集 - 从CSV读取标签，从HDF5读取图像数据"""
    
    def __init__(self, 
                 csv_path: str,
                 hdf5_path: str, 
                 enable_cache: bool = True,
                 cache_size: int = 100,
                 label_columns: List[str] = None):
        """
        初始化混合数据集
        
        Args:
            csv_path: CSV文件路径，包含标签数据
            hdf5_path: HDF5文件路径，包含预处理的图像数据
            enable_cache: 是否启用缓存
            cache_size: 缓存大小
            label_columns: 标签列名列表，默认为['lvef', 'AS_maybe']
        """
        self.csv_path = csv_path
        self.hdf5_path = hdf5_path
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.label_columns = label_columns or ['lvef', 'AS_maybe']
        
        # 初始化缓存
        if self.enable_cache:
            self._cache = {}
        
        # 读取CSV数据
        self.df = pd.read_csv(csv_path)
        print(f"从CSV文件读取了 {len(self.df)} 行数据")
        
        # 检查必需的列
        required_columns = ['basename', 'folder'] + self.label_columns
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件中缺少必需的列: {missing_columns}")
        
        # 清理数据：移除缺失标签的行
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=self.label_columns)
        print(f"移除缺失标签后剩余 {len(self.df)} 行数据")
        
        if len(self.df) == 0:
            raise ValueError("清理后没有有效的数据行")
        
        # 验证HDF5文件
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5文件不存在: {hdf5_path}")
        
        # 检查HDF5中的数据项
        with h5py.File(hdf5_path, 'r') as f:
            if 'images' not in f:
                raise ValueError(f"HDF5文件中没有'images'组: {hdf5_path}")
            self.hdf5_keys = set(f['images'].keys())
        
        print(f"HDF5文件中有 {len(self.hdf5_keys)} 个图像数据项")
        
        # 尝试加载元数据文件来获取哈希映射
        metadata_path = os.path.join(os.path.dirname(hdf5_path), 'data_metadata.json')
        hash_to_info = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                if 'items' in metadata and isinstance(metadata['items'], list):
                    for item in metadata['items']:
                        if 'item_id' in item and 'basename' in item and 'folder' in item:
                            hash_to_info[item['item_id']] = {
                                'basename': item['basename'],
                                'folder': item['folder']
                            }
                    print(f"从元数据文件加载了 {len(hash_to_info)} 个哈希映射")
                else:
                    print("元数据文件格式不支持，使用传统格式")
            except Exception as e:
                print(f"警告: 无法读取元数据文件 {metadata_path}: {e}")
        
        # 构建item_id到CSV行的映射
        self.valid_items = []
        missing_in_hdf5 = []
        
        for idx, row in self.df.iterrows():
            basename = row['basename']
            folder = row['folder']
            
            # 首先尝试传统格式 {folder}_{basename}
            traditional_item_id = f"{folder}_{basename}"
            
            # 然后尝试从哈希映射中查找
            hash_item_id = None
            for hash_id, info in hash_to_info.items():
                if info['basename'] == basename and info['folder'] == folder:
                    hash_item_id = hash_id
                    break
            
            # 确定实际的item_id
            actual_item_id = None
            if traditional_item_id in self.hdf5_keys:
                actual_item_id = traditional_item_id
            elif hash_item_id and hash_item_id in self.hdf5_keys:
                actual_item_id = hash_item_id
            
            if actual_item_id:
                # 提取标签数据
                labels = []
                for col in self.label_columns:
                    labels.append(float(row[col]))
                
                self.valid_items.append({
                    'item_id': actual_item_id,
                    'csv_idx': idx,
                    'basename': basename,
                    'folder': folder,
                    'labels': labels
                })
            else:
                missing_in_hdf5.append(f"{folder}_{basename}")
        
        print(f"匹配到 {len(self.valid_items)} 个有效数据项")
        
        if missing_in_hdf5:
            print(f"警告: {len(missing_in_hdf5)} 个CSV条目在HDF5中找不到对应的图像数据")
            if len(missing_in_hdf5) <= 5:
                print("缺失的item_id:", missing_in_hdf5)
        
        if len(self.valid_items) == 0:
            raise ValueError("没有找到CSV和HDF5中都存在的有效数据项")
        
        print(f"最终数据集大小: {len(self.valid_items)} 个样本")
        
        # 打印标签统计信息
        self._print_label_stats()
    
    def _print_label_stats(self):
        """打印标签统计信息"""
        print(f"\n📊 标签统计信息:")
        for i, col in enumerate(self.label_columns):
            values = [item['labels'][i] for item in self.valid_items]
            print(f"{col}:")
            print(f"  均值: {np.mean(values):.2f}")
            print(f"  标准差: {np.std(values):.2f}")
            print(f"  范围: [{np.min(values):.2f}, {np.max(values):.2f}]")
            
            # 如果是分类标签（AS_maybe），显示分布
            if 'AS' in col.upper():
                unique_values = np.unique(values)
                if len(unique_values) <= 5:  # 假设是分类标签
                    for val in unique_values:
                        count = np.sum(np.array(values) == val)
                        percentage = (count / len(values)) * 100
                        print(f"  类别 {int(val)}: {count} 样本 ({percentage:.1f}%)")
    
    def __len__(self):
        return len(self.valid_items)
    
    def _get_from_cache(self, item_id: str):
        """从缓存获取数据"""
        if not self.enable_cache:
            return None
        return self._cache.get(item_id)
    
    def _add_to_cache(self, item_id: str, data: np.ndarray):
        """添加数据到缓存"""
        if not self.enable_cache:
            return
            
        if len(self._cache) >= self.cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[item_id] = data
    
    def _load_image_from_hdf5(self, item_id: str) -> np.ndarray:
        """从HDF5文件加载图像数据"""
        # 检查缓存
        cached_data = self._get_from_cache(item_id)
        if cached_data is not None:
            return cached_data
        
        # 从HDF5文件加载
        with h5py.File(self.hdf5_path, 'r') as f:
            image_data = f['images'][item_id][:]
        
        # 添加到缓存
        self._add_to_cache(item_id, image_data)
        
        return image_data
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_info = self.valid_items[idx]
        item_id = item_info['item_id']
        
        # 从HDF5加载图像数据
        image_data = self._load_image_from_hdf5(item_id)
        
        # 转换为tensor
        image_tensor = torch.from_numpy(image_data).float()
        
        # 构建标签tensor
        labels_tensor = torch.tensor(item_info['labels'], dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'cardiac_metrics': labels_tensor,  # 保持与原有接口一致
            'labels': labels_tensor,  # 额外提供labels字段
            'patient_id': item_info['basename'],
            'basename': item_info['basename'],
            'folder': item_info['folder'],
            'item_id': item_id
        }


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
        
        # Setup logging first (before preloading data)
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache
        self._cache = {} if enable_cache else None
        self._cache_lock = threading.Lock() if enable_cache else None
        
        # Preload data
        if preload_data:
            self._preload_all_data()
    
    def _load_metadata_index(self) -> Dict[str, Any]:
        """Load metadata index"""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate metadata structure
            if not isinstance(metadata, dict):
                raise ValueError(f"Metadata file should contain a dict, got {type(metadata)}")
            
            required_keys = ['items']
            missing_keys = [key for key in required_keys if key not in metadata]
            if missing_keys:
                raise ValueError(f"Metadata file missing required keys: {missing_keys}")
            
            return metadata
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse metadata file {self.metadata_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load metadata file {self.metadata_path}: {e}")
    
    def _validate_item_ids(self):
        """Validate item IDs"""
        items = self.metadata_index.get('items')
        if items is None:
            raise ValueError("Metadata index does not contain 'items' key")
        
        if isinstance(items, list):
            # If items is a list, this indicates an incorrect metadata format
            raise ValueError(
                "元数据文件格式不正确。'items' 应该是字典而不是列表。\n"
                "这通常表示数据预处理没有正确完成。请重新运行数据预处理：\n"
                "python -m merlin.training.data_preprocessor --config config.json --force"
            )
        elif isinstance(items, dict):
            # If items is a dict, use the keys
            available_ids = set(items.keys())
        else:
            raise ValueError(f"Unexpected type for metadata 'items': {type(items)}. Expected dict or list.")
        
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
            # 如果没有真实标签，抛出错误而不是使用模拟数据
            raise ValueError(f"样本 {item_id} 缺少心脏功能标签数据。请确保预处理数据中包含有效的cardiac_metrics。")
        
        return {
            'image': image_tensor,
            'cardiac_metrics': cardiac_metrics,
            'patient_id': data['metadata']['patient_id'],
            'basename': data['metadata']['basename'],
            'folder': data['metadata']['folder']
        }
    



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
        
        # Debug metadata structure
        self.logger.info(f"Metadata index keys: {list(self.metadata_index.keys())}")
        items = self.metadata_index.get('items')
        if items is not None:
            self.logger.info(f"Items type: {type(items)}, length: {len(items)}")
            if isinstance(items, dict):
                self.logger.info(f"Loaded metadata index: {len(items)} data items")
            elif isinstance(items, list):
                self.logger.info(f"Loaded metadata index: {len(items)} data items (list format)")
            else:
                self.logger.error(f"Unexpected items type: {type(items)}")
        else:
            self.logger.error("No 'items' key found in metadata index")
    
    def _load_metadata_index(self) -> Dict[str, Any]:
        """Load metadata index"""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate metadata structure
            if not isinstance(metadata, dict):
                raise ValueError(f"Metadata file should contain a dict, got {type(metadata)}")
            
            required_keys = ['items']
            missing_keys = [key for key in required_keys if key not in metadata]
            if missing_keys:
                raise ValueError(f"Metadata file missing required keys: {missing_keys}")
            
            return metadata
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse metadata file {self.metadata_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load metadata file {self.metadata_path}: {e}")
    
    def get_all_item_ids(self) -> List[str]:
        """Get all data item IDs"""
        items = self.metadata_index.get('items')
        if items is None:
            raise ValueError("Metadata index does not contain 'items' key")
        
        if isinstance(items, list):
            # If items is a list, this indicates an incorrect metadata format
            self.logger.error("Metadata 'items' is a list, but should be a dict. This indicates corrupted or incomplete preprocessing.")
            raise ValueError(
                "元数据文件格式不正确。'items' 应该是字典而不是列表。\n"
                "这通常表示数据预处理没有正确完成。请重新运行数据预处理：\n"
                "python -m merlin.training.data_preprocessor --config config.json --force"
            )
        elif isinstance(items, dict):
            # If items is a dict, return the keys
            return list(items.keys())
        else:
            raise ValueError(f"Unexpected type for metadata 'items': {type(items)}. Expected dict or list.")
    
    def get_items_by_patient(self, patient_id: str) -> List[str]:
        """Get data items by patient ID"""
        patient_mapping = self.metadata_index.get('patient_mapping', {})
        if not isinstance(patient_mapping, dict):
            self.logger.warning("Patient mapping is not available or not a dict")
            return []
        return patient_mapping.get(patient_id, [])
    
    def get_items_by_folder(self, folder: str) -> List[str]:
        """Get data items by folder"""
        folder_mapping = self.metadata_index.get('folder_mapping', {})
        if not isinstance(folder_mapping, dict):
            self.logger.warning("Folder mapping is not available or not a dict")
            return []
        return folder_mapping.get(folder, [])
    
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
            hdf5_path=str(self.hdf5_path),
            metadata_path=str(self.metadata_path),
            item_ids=item_ids,
            enable_cache=enable_cache,
            cache_size=cache_size,
            preload_data=preload_data
        )
    
    def create_dataloaders(self,
                          split_method: str = 'random',
                          train_ratio: float = 0.8,
                          batch_size: int = 4,
                          num_workers: int = 4,
                          enable_cache: bool = True,
                          cache_size: int = 1000,
                          preload_data: bool = False,
                          random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""
        
        # Split data
        train_ids, val_ids = self.split_data(
            split_method=split_method,
            train_ratio=train_ratio,
            random_state=random_state
        )
        
        self.logger.info(f"Data split completed: {len(train_ids)} training, {len(val_ids)} validation")
        
        # Create datasets
        train_dataset = FastCardiacDataset(
            hdf5_path=str(self.hdf5_path),
            metadata_path=str(self.metadata_path),
            item_ids=train_ids,
            enable_cache=enable_cache,
            cache_size=cache_size,
            preload_data=preload_data
        )
        
        val_dataset = FastCardiacDataset(
            hdf5_path=str(self.hdf5_path),
            metadata_path=str(self.metadata_path),
            item_ids=val_ids,
            enable_cache=enable_cache,
            cache_size=cache_size,
            preload_data=preload_data
        ) if val_ids else None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        ) if val_dataset else None
        
        return train_loader, val_loader
    
    def create_hybrid_dataloaders(self,
                                 csv_path: str,
                                 label_columns: List[str] = None,
                                 split_method: str = 'random',
                                 train_ratio: float = 0.8,
                                 batch_size: int = 4,
                                 num_workers: int = 4,
                                 enable_cache: bool = True,
                                 cache_size: int = 100,
                                 random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
        """
        创建混合数据加载器 - 从CSV读取标签，从HDF5读取图像
        
        Args:
            csv_path: CSV文件路径，包含标签数据
            label_columns: 标签列名列表，默认为['lvef', 'AS_maybe']
            split_method: 数据分割方法 ('random' 或 'patient_based')
            train_ratio: 训练集比例
            batch_size: 批量大小
            num_workers: 数据加载进程数
            enable_cache: 是否启用缓存
            cache_size: 缓存大小
            random_state: 随机种子
            
        Returns:
            train_loader, val_loader: 训练和验证数据加载器
        """
        
        print("=" * 80)
        print("🔄 创建混合数据加载器 (CSV标签 + HDF5图像)")
        print("=" * 80)
        print(f"📊 CSV标签文件: {csv_path}")
        print(f"🖼️ HDF5图像文件: {self.hdf5_path}")
        
        # 验证CSV文件存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        
        # 创建完整数据集
        full_dataset = HybridCardiacDataset(
            csv_path=csv_path,
            hdf5_path=str(self.hdf5_path),
            enable_cache=enable_cache,
            cache_size=cache_size,
            label_columns=label_columns
        )
        
        print(f"✅ 成功创建数据集，共 {len(full_dataset)} 个样本")
        
        # 数据分割
        all_items = full_dataset.valid_items
        
        if split_method == 'random':
            # 随机分割
            train_items, val_items = train_test_split(
                all_items,
                train_size=train_ratio,
                random_state=random_state,
                shuffle=True
            )
        elif split_method == 'patient_based':
            # 基于患者的分割
            patient_items = {}
            for item in all_items:
                patient_id = item['basename']  # 使用basename作为patient_id
                if patient_id not in patient_items:
                    patient_items[patient_id] = []
                patient_items[patient_id].append(item)
            
            patients = list(patient_items.keys())
            train_patients, val_patients = train_test_split(
                patients,
                train_size=train_ratio,
                random_state=random_state,
                shuffle=True
            )
            
            train_items = []
            val_items = []
            for patient in train_patients:
                train_items.extend(patient_items[patient])
            for patient in val_patients:
                val_items.extend(patient_items[patient])
        else:
            raise ValueError(f"不支持的数据分割方法: {split_method}")
        
        print(f"📊 数据分割完成:")
        print(f"   训练集: {len(train_items)} 个样本")
        print(f"   验证集: {len(val_items)} 个样本")
        print(f"   分割方法: {split_method}")
        
        # 创建训练和验证数据集
        train_dataset = HybridCardiacDataset.__new__(HybridCardiacDataset)
        train_dataset.__dict__.update(full_dataset.__dict__)
        train_dataset.valid_items = train_items
        
        val_dataset = None
        if val_items:
            val_dataset = HybridCardiacDataset.__new__(HybridCardiacDataset)
            val_dataset.__dict__.update(full_dataset.__dict__)
            val_dataset.valid_items = val_items
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False
            )
        
        # 统计标签分布
        self._print_hybrid_label_stats(train_items, val_items, full_dataset.label_columns)
        
        print("=" * 80)
        
        return train_loader, val_loader
    
    def _print_hybrid_label_stats(self, train_items: List[Dict], val_items: List[Dict], label_columns: List[str]):
        """打印混合数据集的标签分布统计"""
        print(f"\n📈 标签分布统计:")
        
        # 训练集统计
        if train_items:
            print("训练集:")
            for i, col in enumerate(label_columns):
                values = [item['labels'][i] for item in train_items]
                print(f"  {col}:")
                print(f"    均值: {np.mean(values):.2f}")
                print(f"    标准差: {np.std(values):.2f}")
                print(f"    范围: [{np.min(values):.2f}, {np.max(values):.2f}]")
                
                # 如果是分类标签，显示分布
                if 'AS' in col.upper():
                    unique_values = np.unique(values)
                    if len(unique_values) <= 5:
                        for val in unique_values:
                            count = np.sum(np.array(values) == val)
                            percentage = (count / len(values)) * 100
                            print(f"    类别 {int(val)}: {count} 样本 ({percentage:.1f}%)")
        
        # 验证集统计
        if val_items:
            print("验证集:")
            for i, col in enumerate(label_columns):
                values = [item['labels'][i] for item in val_items]
                print(f"  {col}:")
                print(f"    均值: {np.mean(values):.2f}")
                print(f"    标准差: {np.std(values):.2f}")
                print(f"    范围: [{np.min(values):.2f}, {np.max(values):.2f}]")
                
                # 如果是分类标签，显示分布
                if 'AS' in col.upper():
                    unique_values = np.unique(values)
                    if len(unique_values) <= 5:
                        for val in unique_values:
                            count = np.sum(np.array(values) == val)
                            percentage = (count / len(values)) * 100
                            print(f"    类别 {int(val)}: {count} 样本 ({percentage:.1f}%)")
    
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


def check_preprocessing_status(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check the status of preprocessing data"""
    preprocessed_dir = config.get('preprocessed_data_dir', '')
    if not preprocessed_dir:
        return {
            'status': 'missing_config',
            'message': 'preprocessed_data_dir not specified in config'
        }
    
    hdf5_path = Path(preprocessed_dir) / 'preprocessed_data.h5'
    metadata_path = Path(preprocessed_dir) / 'data_metadata.json'
    
    status = {
        'preprocessed_dir': preprocessed_dir,
        'hdf5_exists': hdf5_path.exists(),
        'metadata_exists': metadata_path.exists(),
        'hdf5_path': str(hdf5_path),
        'metadata_path': str(metadata_path)
    }
    
    if not status['hdf5_exists'] or not status['metadata_exists']:
        status['status'] = 'files_missing'
        return status
    
    # Check metadata format
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if not isinstance(metadata, dict):
            status['status'] = 'invalid_metadata_format'
            status['metadata_type'] = str(type(metadata))
            return status
        
        if 'items' not in metadata:
            status['status'] = 'missing_items_key'
            status['metadata_keys'] = list(metadata.keys())
            return status
        
        items = metadata['items']
        if not isinstance(items, dict):
            status['status'] = 'invalid_items_format'
            status['items_type'] = str(type(items))
            return status
        
        status['status'] = 'ok'
        status['num_items'] = len(items)
        return status
        
    except Exception as e:
        status['status'] = 'metadata_error'
        status['error'] = str(e)
        return status


def create_fast_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create convenient function for fast data loaders"""
    try:
        manager = FastDataLoaderManager(config)
        return manager.create_dataloaders(
            split_method=config.get('split_method', 'random'),
            train_ratio=config.get('train_val_split', 0.8),
            batch_size=config.get('batch_size', 4),
            num_workers=config.get('num_workers', 4),
            enable_cache=config.get('cache_config', {}).get('enable_cache', True),
            cache_size=config.get('cache_config', {}).get('cache_size', 1000),
            preload_data=config.get('cache_config', {}).get('preload_train_data', False),
            random_state=config.get('seed', 42)
        )
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ 快速数据加载器初始化失败")
        print("=" * 80)
        print(f"错误: {e}")
        
        # Check preprocessing status
        status = check_preprocessing_status(config)
        print(f"\n🔍 预处理数据状态检查:")
        print(f"   目录: {status.get('preprocessed_dir', 'N/A')}")
        print(f"   HDF5文件存在: {status.get('hdf5_exists', False)}")
        print(f"   元数据文件存在: {status.get('metadata_exists', False)}")
        
        if status['status'] == 'files_missing':
            print("\n❌ 预处理数据文件缺失")
        elif status['status'] == 'invalid_metadata_format':
            print(f"\n❌ 元数据文件格式错误 (类型: {status.get('metadata_type', 'unknown')})")
        elif status['status'] == 'missing_items_key':
            print(f"\n❌ 元数据文件缺少'items'键 (现有键: {status.get('metadata_keys', [])})")
        elif status['status'] == 'invalid_items_format':
            print(f"\n❌ 元数据中'items'格式错误 (类型: {status.get('items_type', 'unknown')}，应为dict)")
        elif status['status'] == 'ok':
            print(f"\n✅ 预处理数据格式正确 ({status.get('num_items', 0)} 个数据项)")
        
        print("\n🔍 解决方案:")
        print("1. 运行或重新运行数据预处理:")
        print("   python -m merlin.training.data_preprocessor --config config.json --force")
        print("2. 确保CSV文件路径正确")
        print("3. 确保图像文件路径正确")
        print("4. 检查磁盘空间是否足够")
        print("=" * 80)
        raise


def benchmark_data_loading(config: Dict[str, Any], num_batches: int = 10) -> Dict[str, float]:
    """Benchmark data loading performance"""
    import time
    
    manager = FastDataLoaderManager(config)
    train_loader, val_loader = manager.create_dataloaders(
        split_method=config.get('split_method', 'random'),
        train_ratio=config.get('train_val_split', 0.8),
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 4),
        enable_cache=config.get('cache_config', {}).get('enable_cache', True),
        cache_size=config.get('cache_config', {}).get('cache_size', 1000),
        preload_data=config.get('cache_config', {}).get('preload_train_data', False),
        random_state=config.get('seed', 42)
    )
    
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
        'preprocessed_data_dir': '/data/joycewyr/cardiac_training_fast',
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


def create_hybrid_dataloaders_from_config(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    从配置创建混合数据加载器的便捷函数
    
    Args:
        config: 配置字典，必须包含以下键：
            - csv_path: CSV标签文件路径
            - hdf5_path: HDF5图像文件路径 (或 preprocessed_data_dir)
            - label_columns: 标签列名列表 (可选，默认['lvef', 'AS_maybe'])
            - split_method: 数据分割方法 (可选，默认'random')
            - train_val_split: 训练集比例 (可选，默认0.8)
            - batch_size: 批量大小 (可选，默认4)
            - num_workers: 数据加载进程数 (可选，默认4)
            - cache_size: 缓存大小 (可选，默认100)
            - seed: 随机种子 (可选，默认42)
    
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    
    # 检查必需的配置
    if 'csv_path' not in config:
        raise ValueError("配置中缺少csv_path")
    
    # 确定HDF5文件路径
    if 'hdf5_path' in config:
        hdf5_path = config['hdf5_path']
    elif 'preprocessed_data_dir' in config:
        hdf5_path = str(Path(config['preprocessed_data_dir']) / 'preprocessed_data.h5')
    else:
        raise ValueError("配置中必须包含hdf5_path或preprocessed_data_dir")
    
    # 验证文件存在
    if not os.path.exists(config['csv_path']):
        raise FileNotFoundError(f"CSV文件不存在: {config['csv_path']}")
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5文件不存在: {hdf5_path}")
    
    print("=" * 80)
    print("🚀 创建混合数据加载器 (从配置)")
    print("=" * 80)
    print(f"📊 CSV标签文件: {config['csv_path']}")
    print(f"🖼️ HDF5图像文件: {hdf5_path}")
    
    # 直接创建数据集和加载器，不需要FastDataLoaderManager
    # 创建完整数据集
    full_dataset = HybridCardiacDataset(
        csv_path=config['csv_path'],
        hdf5_path=hdf5_path,
        enable_cache=config.get('enable_cache', True),
        cache_size=config.get('cache_size', 100),
        label_columns=config.get('label_columns', ['lvef', 'AS_maybe'])
    )
    
    print(f"✅ 成功创建数据集，共 {len(full_dataset)} 个样本")
    
    # 数据分割
    split_method = config.get('split_method', 'random')
    train_ratio = config.get('train_val_split', 0.8)
    random_state = config.get('seed', 42)
    
    all_items = full_dataset.valid_items
    
    if split_method == 'random':
        train_items, val_items = train_test_split(
            all_items,
            train_size=train_ratio,
            random_state=random_state,
            shuffle=True
        )
    elif split_method == 'patient_based':
        # 基于患者的分割
        patient_items = {}
        for item in all_items:
            patient_id = item['basename']
            if patient_id not in patient_items:
                patient_items[patient_id] = []
            patient_items[patient_id].append(item)
        
        patients = list(patient_items.keys())
        train_patients, val_patients = train_test_split(
            patients,
            train_size=train_ratio,
            random_state=random_state,
            shuffle=True
        )
        
        train_items = []
        val_items = []
        for patient in train_patients:
            train_items.extend(patient_items[patient])
        for patient in val_patients:
            val_items.extend(patient_items[patient])
    else:
        raise ValueError(f"不支持的数据分割方法: {split_method}")
    
    print(f"📊 数据分割完成:")
    print(f"   训练集: {len(train_items)} 个样本")
    print(f"   验证集: {len(val_items)} 个样本")
    print(f"   分割方法: {split_method}")
    
    # 创建训练和验证数据集
    train_dataset = HybridCardiacDataset.__new__(HybridCardiacDataset)
    train_dataset.__dict__.update(full_dataset.__dict__)
    train_dataset.valid_items = train_items
    
    val_dataset = None
    if val_items:
        val_dataset = HybridCardiacDataset.__new__(HybridCardiacDataset)
        val_dataset.__dict__.update(full_dataset.__dict__)
        val_dataset.valid_items = val_items
    
    # 创建数据加载器
    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    # 打印标签分布统计
    _print_label_distribution_stats(train_items, val_items, full_dataset.label_columns)
    
    print("=" * 80)
    
    return train_loader, val_loader


def _print_label_distribution_stats(train_items: List[Dict], val_items: List[Dict], label_columns: List[str]):
    """打印标签分布统计的辅助函数"""
    print(f"\n📈 标签分布统计:")
    
    # 训练集统计
    if train_items:
        print("训练集:")
        for i, col in enumerate(label_columns):
            values = [item['labels'][i] for item in train_items]
            print(f"  {col}:")
            print(f"    均值: {np.mean(values):.2f}")
            print(f"    标准差: {np.std(values):.2f}")
            print(f"    范围: [{np.min(values):.2f}, {np.max(values):.2f}]")
            
            # 如果是分类标签，显示分布
            if 'AS' in col.upper():
                unique_values = np.unique(values)
                if len(unique_values) <= 5:
                    for val in unique_values:
                        count = np.sum(np.array(values) == val)
                        percentage = (count / len(values)) * 100
                        print(f"    类别 {int(val)}: {count} 样本 ({percentage:.1f}%)")
    
    # 验证集统计
    if val_items:
        print("验证集:")
        for i, col in enumerate(label_columns):
            values = [item['labels'][i] for item in val_items]
            print(f"  {col}:")
            print(f"    均值: {np.mean(values):.2f}")
            print(f"    标准差: {np.std(values):.2f}")
            print(f"    范围: [{np.min(values):.2f}, {np.max(values):.2f}]")
            
            # 如果是分类标签，显示分布
            if 'AS' in col.upper():
                unique_values = np.unique(values)
                if len(unique_values) <= 5:
                    for val in unique_values:
                        count = np.sum(np.array(values) == val)
                        percentage = (count / len(values)) * 100
                        print(f"    类别 {int(val)}: {count} 样本 ({percentage:.1f}%)")


# 为了向后兼容，提供别名
create_hybrid_data_loaders = create_hybrid_dataloaders_from_config 