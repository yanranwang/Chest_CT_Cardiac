#!/usr/bin/env python3
"""
数据预处理器 - 批量预处理心脏功能训练数据

该脚本用于批量预处理所有训练数据，将处理后的数据保存到HDF5文件中，
以加速训练过程并提高GPU利用率。

主要功能：
1. 批量处理所有图像数据
2. 保存到HDF5数据库
3. 生成数据索引
4. 验证数据完整性
5. 支持增量更新

使用方法:
    python data_preprocessor.py --config config.json
"""

import os
import h5py
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import hashlib
import multiprocessing as mp
from functools import partial
import logging
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from merlin.data.monai_transforms import ImageTransforms
from merlin.training.cardiac_trainer import load_and_validate_csv_data, build_data_list


class DataPreprocessor:
    """数据预处理器 - 批量预处理并保存到HDF5"""
    
    def __init__(self, config: dict):
        self.config = config
        self.setup_logging()
        
        # 输出路径
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # HDF5文件路径
        self.hdf5_path = self.output_dir / 'preprocessed_data.h5'
        self.metadata_path = self.output_dir / 'data_metadata.json'
        
        # 预处理转换
        self.transform = ImageTransforms
        
        # 数据统计
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'processing_time': 0
        }
    
    def setup_logging(self):
        """设置日志"""
        log_file = Path(self.config['output_dir']) / 'preprocessing.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_data_hash(self, data_item: dict) -> str:
        """生成数据项的唯一哈希值"""
        hash_content = f"{data_item['image']}_{data_item['basename']}_{data_item['folder']}"
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def preprocess_single_item(self, data_item: dict) -> Optional[Tuple[str, dict]]:
        """预处理单个数据项"""
        try:
            # 检查文件是否存在
            image_path = data_item.get('image', '')
            if not os.path.exists(image_path):
                self.logger.warning(f"图像文件不存在: {image_path}")
                return None
            
            # 生成唯一ID
            item_id = self.get_data_hash(data_item)
            
            # 应用变换
            processed_item = self.transform(data_item)
            
            # 提取图像数据
            image_tensor = processed_item['image']
            
            # 准备存储的数据
            storage_data = {
                'image': image_tensor.numpy(),
                'cardiac_metrics': data_item.get('cardiac_metrics'),
                'patient_id': data_item.get('patient_id'),
                'basename': data_item.get('basename'),
                'folder': data_item.get('folder'),
                'metadata': data_item.get('metadata', {}),
                'original_path': data_item.get('image'),
                'processed_shape': image_tensor.shape,
                'data_type': str(image_tensor.dtype)
            }
            
            return item_id, storage_data
            
        except Exception as e:
            self.logger.error(f"预处理失败: {data_item.get('image', 'unknown')}, 错误: {e}")
            return None
    
    def process_batch(self, data_batch: List[dict]) -> List[Tuple[str, dict]]:
        """批量处理数据"""
        results = []
        
        for data_item in data_batch:
            result = self.preprocess_single_item(data_item)
            if result:
                results.append(result)
                self.stats['processed_files'] += 1
            else:
                self.stats['failed_files'] += 1
        
        return results
    
    def save_to_hdf5(self, processed_data: List[Tuple[str, dict]]):
        """保存处理后的数据到HDF5文件"""
        self.logger.info(f"保存 {len(processed_data)} 个处理后的数据项到 {self.hdf5_path}")
        
        with h5py.File(self.hdf5_path, 'a') as f:
            # 创建组
            if 'images' not in f:
                f.create_group('images')
            if 'metadata' not in f:
                f.create_group('metadata')
            
            for item_id, data in tqdm(processed_data, desc="保存到HDF5"):
                # 保存图像数据
                if item_id not in f['images']:
                    f['images'].create_dataset(
                        item_id, 
                        data=data['image'],
                        compression='gzip',
                        compression_opts=9
                    )
                
                # 保存元数据
                if item_id not in f['metadata']:
                    # 将元数据转换为JSON字符串保存
                    metadata_json = json.dumps({
                        'cardiac_metrics': data['cardiac_metrics'].tolist() if data['cardiac_metrics'] is not None else None,
                        'patient_id': data['patient_id'],
                        'basename': data['basename'],
                        'folder': data['folder'],
                        'metadata': data['metadata'],
                        'original_path': data['original_path'],
                        'processed_shape': data['processed_shape'],
                        'data_type': data['data_type']
                    }, ensure_ascii=False)
                    
                    f['metadata'].create_dataset(
                        item_id,
                        data=metadata_json,
                        dtype=h5py.string_dtype(encoding='utf-8')
                    )
    
    def create_index(self, data_list: List[dict]) -> Dict[str, Any]:
        """创建数据索引"""
        index = {
            'items': {},
            'patient_mapping': {},
            'folder_mapping': {},
            'statistics': {
                'total_items': len(data_list),
                'unique_patients': len(set(item['patient_id'] for item in data_list)),
                'unique_folders': len(set(item['folder'] for item in data_list))
            }
        }
        
        for data_item in data_list:
            item_id = self.get_data_hash(data_item)
            
            # 基本信息
            index['items'][item_id] = {
                'patient_id': data_item['patient_id'],
                'basename': data_item['basename'],
                'folder': data_item['folder'],
                'original_path': data_item['image'],
                'has_cardiac_metrics': data_item.get('cardiac_metrics') is not None
            }
            
            # 患者映射
            patient_id = data_item['patient_id']
            if patient_id not in index['patient_mapping']:
                index['patient_mapping'][patient_id] = []
            index['patient_mapping'][patient_id].append(item_id)
            
            # 文件夹映射
            folder = data_item['folder']
            if folder not in index['folder_mapping']:
                index['folder_mapping'][folder] = []
            index['folder_mapping'][folder].append(item_id)
        
        return index
    
    def verify_data_integrity(self) -> bool:
        """验证数据完整性"""
        self.logger.info("验证数据完整性...")
        
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                images_group = f['images']
                metadata_group = f['metadata']
                
                # 检查数据一致性
                image_keys = set(images_group.keys())
                metadata_keys = set(metadata_group.keys())
                
                if image_keys != metadata_keys:
                    self.logger.error("图像数据和元数据键不匹配")
                    return False
                
                # 随机验证几个样本
                sample_keys = list(image_keys)[:min(10, len(image_keys))]
                
                for key in sample_keys:
                    # 检查图像数据
                    image_data = images_group[key][:]
                    if image_data.size == 0:
                        self.logger.error(f"图像数据为空: {key}")
                        return False
                    
                    # 检查元数据
                    metadata_str = metadata_group[key][()]
                    if isinstance(metadata_str, bytes):
                        metadata_str = metadata_str.decode('utf-8')
                    
                    try:
                        metadata = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        self.logger.error(f"元数据格式错误: {key}")
                        return False
                
                self.logger.info(f"数据完整性验证通过，共 {len(image_keys)} 个样本")
                return True
                
        except Exception as e:
            self.logger.error(f"数据完整性验证失败: {e}")
            return False
    
    def get_existing_items(self) -> set:
        """获取已存在的数据项"""
        if not self.hdf5_path.exists():
            return set()
        
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                if 'images' in f:
                    return set(f['images'].keys())
                return set()
        except Exception as e:
            self.logger.warning(f"读取现有数据失败: {e}")
            return set()
    
    def preprocess_data(self, data_list: List[dict], force_reprocess: bool = False):
        """预处理数据的主函数"""
        import time
        start_time = time.time()
        
        self.logger.info(f"开始预处理 {len(data_list)} 个数据项")
        self.stats['total_files'] = len(data_list)
        
        # 检查已存在的数据
        existing_items = set() if force_reprocess else self.get_existing_items()
        
        # 过滤需要处理的数据
        items_to_process = []
        for data_item in data_list:
            item_id = self.get_data_hash(data_item)
            if item_id not in existing_items:
                items_to_process.append(data_item)
            else:
                self.stats['skipped_files'] += 1
        
        if not items_to_process:
            self.logger.info("所有数据已经预处理完成")
            return
        
        self.logger.info(f"需要处理 {len(items_to_process)} 个新数据项")
        
        # 批量处理配置
        batch_size = self.config.get('preprocess_batch_size', 16)
        num_workers = self.config.get('num_workers', 1)  # 默认使用单进程
        enable_multiprocessing = self.config.get('enable_multiprocessing', False)
        
        # 检查是否启用多进程
        if enable_multiprocessing and num_workers > 1:
            self.logger.info(f"使用 {num_workers} 个进程进行并行处理")
            try:
                self._multiprocess_preprocess(items_to_process, batch_size, num_workers)
            except Exception as e:
                self.logger.error(f"多进程处理失败: {e}")
                self.logger.info("回退到单进程处理...")
                self._single_process_preprocess(items_to_process, batch_size)
        else:
            self.logger.info("使用单进程处理")
            self._single_process_preprocess(items_to_process, batch_size)
        
        # 创建索引
        self.logger.info("创建数据索引...")
        index = self.create_index(data_list)
        
        # 保存索引
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        # 验证数据完整性
        if not self.verify_data_integrity():
            raise RuntimeError("数据完整性验证失败")
        
        # 统计信息
        self.stats['processing_time'] = time.time() - start_time
        self._print_statistics()
        
        # 保存统计信息
        with open(self.output_dir / 'preprocessing_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _single_process_preprocess(self, items_to_process: List[dict], batch_size: int):
        """单进程预处理"""
        total_batches = (len(items_to_process) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(items_to_process), batch_size), 
                      desc="预处理批次", 
                      total=total_batches,
                      ncols=100):
            batch = items_to_process[i:i + batch_size]
            processed_batch = self.process_batch(batch)
            
            if processed_batch:
                self.save_to_hdf5(processed_batch)
                self.logger.info(f"已处理批次 {i//batch_size + 1}/{total_batches}, 成功处理 {len(processed_batch)} 个数据项")
            else:
                self.logger.warning(f"批次 {i//batch_size + 1}/{total_batches} 处理失败或无有效数据")
    
    def _multiprocess_preprocess(self, items_to_process: List[dict], batch_size: int, num_workers: int):
        """多进程预处理"""
        # 创建数据批次
        batches = [items_to_process[i:i + batch_size] for i in range(0, len(items_to_process), batch_size)]
        
        # 使用进程池处理
        with mp.Pool(processes=num_workers) as pool:
            # 使用独立的处理函数避免序列化问题
            results = pool.map(process_batch_standalone, batches)
        
        # 保存结果
        for processed_batch in results:
            if processed_batch:
                self.save_to_hdf5(processed_batch)
    
    def _print_statistics(self):
        """打印统计信息"""
        self.logger.info("=" * 80)
        self.logger.info("📊 预处理统计信息")
        self.logger.info("=" * 80)
        self.logger.info(f"📁 总文件数: {self.stats['total_files']}")
        self.logger.info(f"✅ 成功处理: {self.stats['processed_files']}")
        self.logger.info(f"⏭️  跳过文件: {self.stats['skipped_files']}")
        self.logger.info(f"❌ 失败文件: {self.stats['failed_files']}")
        self.logger.info(f"⏱️  处理时间: {self.stats['processing_time']:.2f} 秒")
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['processed_files'] / self.stats['total_files']) * 100
            self.logger.info(f"📈 成功率: {success_rate:.2f}%")
        self.logger.info(f"💾 输出文件: {self.hdf5_path}")
        self.logger.info(f"📋 索引文件: {self.metadata_path}")
        self.logger.info("=" * 80)


def process_batch_standalone(data_batch: List[dict]) -> List[Tuple[str, dict]]:
    """独立的批处理函数，用于多进程处理"""
    results = []
    
    # 导入必要的模块
    from merlin.data.monai_transforms import ImageTransforms
    import hashlib
    
    transform = ImageTransforms
    
    for data_item in data_batch:
        try:
            # 检查文件是否存在
            image_path = data_item.get('image', '')
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在: {image_path}")
                continue
            
            # 生成唯一ID
            hash_content = f"{data_item['image']}_{data_item['basename']}_{data_item['folder']}"
            item_id = hashlib.md5(hash_content.encode()).hexdigest()
            
            # 应用变换
            processed_item = transform(data_item)
            
            # 提取图像数据
            image_tensor = processed_item['image']
            
            # 准备存储的数据
            storage_data = {
                'image': image_tensor.numpy(),
                'cardiac_metrics': data_item.get('cardiac_metrics'),
                'patient_id': data_item.get('patient_id'),
                'basename': data_item.get('basename'),
                'folder': data_item.get('folder'),
                'metadata': data_item.get('metadata', {}),
                'original_path': data_item.get('image'),
                'processed_shape': image_tensor.shape,
                'data_type': str(image_tensor.dtype)
            }
            
            results.append((item_id, storage_data))
            
        except Exception as e:
            print(f"预处理失败: {data_item.get('image', 'unknown')}, 错误: {e}")
            continue
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据预处理器')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--force', action='store_true', help='强制重新处理所有数据')
    parser.add_argument('--output_dir', help='输出目录')
    parser.add_argument('--batch_size', type=int, help='批处理大小')
    parser.add_argument('--num_workers', type=int, help='并行处理进程数')
    parser.add_argument('--enable_multiprocessing', action='store_true', help='启用多进程处理')
    parser.add_argument('--test_mode', type=int, help='测试模式：仅处理指定数量的数据项')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 命令行参数覆盖
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.batch_size:
        config['preprocess_batch_size'] = args.batch_size
    if args.num_workers:
        config['num_workers'] = args.num_workers
    if args.enable_multiprocessing:
        config['enable_multiprocessing'] = True
    
    # 默认预处理配置
    config.setdefault('preprocess_batch_size', 16)
    config.setdefault('num_workers', 1)  # 默认单进程
    config.setdefault('enable_multiprocessing', False)  # 默认关闭多进程
    
    try:
        # 加载数据列表
        print("🔍 加载数据...")
        df, cardiac_metric_columns = load_and_validate_csv_data(config)
        data_list = build_data_list(df, config, cardiac_metric_columns)
        
        if not data_list:
            print("❌ 没有找到有效的数据")
            return
        
        # 测试模式：仅处理指定数量的数据
        if args.test_mode:
            original_size = len(data_list)
            data_list = data_list[:args.test_mode]
            print(f"🧪 测试模式：从 {original_size} 个数据项中选择前 {len(data_list)} 个进行处理")
        
        # 创建预处理器
        preprocessor = DataPreprocessor(config)
        
        # 开始预处理
        print("🚀 开始预处理数据...")
        preprocessor.preprocess_data(data_list, force_reprocess=args.force)
        
        print("🎉 数据预处理完成！")
        print(f"📁 预处理数据保存在: {preprocessor.hdf5_path}")
        print(f"📋 数据索引保存在: {preprocessor.metadata_path}")
        
    except Exception as e:
        print(f"❌ 预处理失败: {e}")
        raise


if __name__ == '__main__':
    main() 