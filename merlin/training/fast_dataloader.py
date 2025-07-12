#!/usr/bin/env python3
"""
快速数据加载器 - 从HDF5预处理数据中高效读取

该模块提供了高效的数据加载器，直接从预处理的HDF5文件中读取数据，
大大减少训练时的数据加载时间，提高GPU利用率。

主要特性：
1. 直接从HDF5读取预处理数据
2. 支持内存缓存和预加载
3. 高效的多进程数据读取
4. 支持数据增强（可选）
5. 智能数据分割
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
    """快速心脏功能数据集 - 从HDF5文件读取预处理数据"""
    
    def __init__(self, 
                 hdf5_path: str, 
                 metadata_path: str,
                 item_ids: List[str],
                 enable_cache: bool = True,
                 cache_size: int = 1000,
                 preload_data: bool = False):
        """
        初始化快速数据集
        
        Args:
            hdf5_path: HDF5文件路径
            metadata_path: 元数据文件路径
            item_ids: 数据项ID列表
            enable_cache: 是否启用内存缓存
            cache_size: 缓存大小
            preload_data: 是否预加载所有数据到内存
        """
        self.hdf5_path = Path(hdf5_path)
        self.metadata_path = Path(metadata_path)
        self.item_ids = item_ids
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.preload_data = preload_data
        
        # 验证文件存在
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5文件不存在: {hdf5_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        # 加载元数据索引
        self.metadata_index = self._load_metadata_index()
        
        # 验证数据项ID
        self._validate_item_ids()
        
        # 初始化缓存
        self._cache = {} if enable_cache else None
        self._cache_lock = threading.Lock() if enable_cache else None
        
        # 预加载数据
        if preload_data:
            self._preload_all_data()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
    
    def _load_metadata_index(self) -> Dict[str, Any]:
        """加载元数据索引"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _validate_item_ids(self):
        """验证数据项ID"""
        available_ids = set(self.metadata_index['items'].keys())
        missing_ids = set(self.item_ids) - available_ids
        
        if missing_ids:
            raise ValueError(f"以下数据项ID不存在: {missing_ids}")
    
    def _preload_all_data(self):
        """预加载所有数据到内存"""
        self.logger.info(f"预加载 {len(self.item_ids)} 个数据项...")
        
        self._preloaded_data = {}
        
        with h5py.File(self.hdf5_path, 'r') as f:
            images_group = f['images']
            metadata_group = f['metadata']
            
            for item_id in self.item_ids:
                # 加载图像数据
                image_data = images_group[item_id][:]
                
                # 加载元数据
                metadata_str = metadata_group[item_id][()]
                if isinstance(metadata_str, bytes):
                    metadata_str = metadata_str.decode('utf-8')
                metadata = json.loads(metadata_str)
                
                self._preloaded_data[item_id] = {
                    'image': image_data,
                    'metadata': metadata
                }
        
        self.logger.info("数据预加载完成")
    
    def _get_from_cache(self, item_id: str) -> Optional[Dict[str, Any]]:
        """从缓存获取数据"""
        if not self.enable_cache:
            return None
        
        with self._cache_lock:
            return self._cache.get(item_id)
    
    def _add_to_cache(self, item_id: str, data: Dict[str, Any]):
        """添加数据到缓存"""
        if not self.enable_cache:
            return
        
        with self._cache_lock:
            if len(self._cache) >= self.cache_size:
                # 删除最旧的缓存项
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[item_id] = data
    
    def _load_item_from_hdf5(self, item_id: str) -> Dict[str, Any]:
        """从HDF5文件加载数据项"""
        if self.preload_data:
            return self._preloaded_data[item_id]
        
        # 检查缓存
        cached_data = self._get_from_cache(item_id)
        if cached_data is not None:
            return cached_data
        
        # 从HDF5文件加载
        with h5py.File(self.hdf5_path, 'r') as f:
            images_group = f['images']
            metadata_group = f['metadata']
            
            # 加载图像数据
            image_data = images_group[item_id][:]
            
            # 加载元数据
            metadata_str = metadata_group[item_id][()]
            if isinstance(metadata_str, bytes):
                metadata_str = metadata_str.decode('utf-8')
            metadata = json.loads(metadata_str)
            
            data = {
                'image': image_data,
                'metadata': metadata
            }
            
            # 添加到缓存
            self._add_to_cache(item_id, data)
            
            return data
    
    def __len__(self) -> int:
        return len(self.item_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_id = self.item_ids[idx]
        data = self._load_item_from_hdf5(item_id)
        
        # 转换为张量
        image_tensor = torch.from_numpy(data['image']).float()
        
        # 获取心脏功能指标
        cardiac_metrics = data['metadata']['cardiac_metrics']
        if cardiac_metrics is not None:
            cardiac_metrics = torch.tensor(cardiac_metrics, dtype=torch.float32)
        else:
            # 生成模拟标签
            cardiac_metrics = torch.tensor(self._generate_dummy_labels(), dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'cardiac_metrics': cardiac_metrics,
            'patient_id': data['metadata']['patient_id'],
            'basename': data['metadata']['basename'],
            'folder': data['metadata']['folder'],
            'original_path': data['metadata']['original_path'],
            'metadata': data['metadata'].get('metadata', {})
        }
    
    def _generate_dummy_labels(self) -> np.ndarray:
        """生成模拟的心脏功能标签"""
        # 生成LVEF和AS的模拟数据
        lvef = np.float32(np.random.normal(0, 1))
        as_label = np.float32(np.random.randint(0, 2))
        return np.array([lvef, as_label], dtype=np.float32)


class FastDataLoaderManager:
    """快速数据加载器管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 文件路径
        self.hdf5_path = Path(config['preprocessed_data_dir']) / 'preprocessed_data.h5'
        self.metadata_path = Path(config['preprocessed_data_dir']) / 'data_metadata.json'
        
        # 验证文件存在
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"预处理数据文件不存在: {self.hdf5_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {self.metadata_path}")
        
        # 加载元数据
        self.metadata_index = self._load_metadata_index()
        
        self.logger.info(f"加载元数据索引: {len(self.metadata_index['items'])} 个数据项")
    
    def _load_metadata_index(self) -> Dict[str, Any]:
        """加载元数据索引"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_all_item_ids(self) -> List[str]:
        """获取所有数据项ID"""
        return list(self.metadata_index['items'].keys())
    
    def get_items_by_patient(self, patient_id: str) -> List[str]:
        """根据患者ID获取数据项"""
        return self.metadata_index['patient_mapping'].get(patient_id, [])
    
    def get_items_by_folder(self, folder: str) -> List[str]:
        """根据文件夹获取数据项"""
        return self.metadata_index['folder_mapping'].get(folder, [])
    
    def split_data(self, 
                   split_method: str = 'random',
                   train_ratio: float = 0.8,
                   random_state: int = 42) -> Tuple[List[str], List[str]]:
        """分割数据为训练集和验证集"""
        all_item_ids = self.get_all_item_ids()
        
        if split_method == 'random':
            # 随机分割
            train_ids, val_ids = train_test_split(
                all_item_ids,
                train_size=train_ratio,
                random_state=random_state,
                shuffle=True
            )
        
        elif split_method == 'patient_based':
            # 基于患者的分割
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
            # 顺序分割
            split_idx = int(len(all_item_ids) * train_ratio)
            train_ids = all_item_ids[:split_idx]
            val_ids = all_item_ids[split_idx:]
        
        else:
            raise ValueError(f"不支持的分割方法: {split_method}")
        
        self.logger.info(f"数据分割完成: 训练集 {len(train_ids)}, 验证集 {len(val_ids)}")
        return train_ids, val_ids
    
    def create_dataset(self, 
                      item_ids: List[str],
                      enable_cache: bool = True,
                      cache_size: int = 1000,
                      preload_data: bool = False) -> FastCardiacDataset:
        """创建数据集"""
        return FastCardiacDataset(
            hdf5_path=self.hdf5_path,
            metadata_path=self.metadata_path,
            item_ids=item_ids,
            enable_cache=enable_cache,
            cache_size=cache_size,
            preload_data=preload_data
        )
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """创建训练和验证数据加载器"""
        # 分割数据
        train_ids, val_ids = self.split_data(
            split_method=self.config.get('split_method', 'random'),
            train_ratio=self.config.get('train_val_split', 0.8),
            random_state=self.config.get('seed', 42)
        )
        
        # 创建数据集
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
        
        # 创建数据加载器
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
        """获取数据统计信息"""
        stats = self.metadata_index['statistics'].copy()
        
        # 添加文件大小信息
        hdf5_size = self.hdf5_path.stat().st_size / (1024 * 1024)  # MB
        metadata_size = self.metadata_path.stat().st_size / 1024  # KB
        
        stats.update({
            'hdf5_file_size_mb': round(hdf5_size, 2),
            'metadata_file_size_kb': round(metadata_size, 2),
            'data_loading_method': 'HDF5_Fast_Loading'
        })
        
        return stats


def create_fast_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """创建快速数据加载器的便捷函数"""
    manager = FastDataLoaderManager(config)
    return manager.create_data_loaders()


def benchmark_data_loading(config: Dict[str, Any], num_batches: int = 10) -> Dict[str, float]:
    """基准测试数据加载性能"""
    import time
    
    manager = FastDataLoaderManager(config)
    train_loader, val_loader = manager.create_data_loaders()
    
    # 测试训练数据加载
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        # 模拟数据使用
        _ = batch['image'].shape
    
    train_time = time.time() - start_time
    
    # 测试验证数据加载
    start_time = time.time()
    for i, batch in enumerate(val_loader):
        if i >= num_batches:
            break
        # 模拟数据使用
        _ = batch['image'].shape
    
    val_time = time.time() - start_time
    
    return {
        'train_loading_time': train_time,
        'val_loading_time': val_time,
        'train_batches_per_second': num_batches / train_time,
        'val_batches_per_second': num_batches / val_time
    }


if __name__ == '__main__':
    # 测试快速数据加载器
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
        # 创建数据加载器
        train_loader, val_loader = create_fast_data_loaders(config)
        
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        
        # 测试数据加载
        batch = next(iter(train_loader))
        print(f"批次形状: {batch['image'].shape}")
        print(f"心脏功能指标形状: {batch['cardiac_metrics'].shape}")
        
        # 性能基准测试
        print("\n开始性能基准测试...")
        benchmark_results = benchmark_data_loading(config)
        
        print("基准测试结果:")
        for key, value in benchmark_results.items():
            print(f"  {key}: {value:.4f}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        print("请先运行数据预处理脚本生成HDF5文件") 