#!/usr/bin/env python3
"""
数据加载器debug脚本
用于逐步检查数据加载器创建过程中的问题
"""

import os
import sys
import json
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

def check_config_file():
    """检查配置文件"""
    print("=" * 60)
    print("🔍 步骤1: 检查配置文件")
    print("=" * 60)
    
    config_path = "configs/fast_training_config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"✅ 配置文件读取成功: {config_path}")
        print(f"   快速数据加载器: {config.get('use_fast_loader', False)}")
        print(f"   预处理数据目录: {config.get('preprocessed_data_dir')}")
        print(f"   批次大小: {config.get('batch_size')}")
        print(f"   工作进程数: {config.get('num_workers')}")
        print(f"   预加载训练数据: {config.get('cache_config', {}).get('preload_train_data', False)}")
        print(f"   预加载验证数据: {config.get('cache_config', {}).get('preload_val_data', False)}")
        print(f"   缓存大小: {config.get('cache_config', {}).get('cache_size', 1000)}")
        
        return config
        
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return None

def check_preprocessed_data(config):
    """检查预处理数据"""
    print("\n" + "=" * 60)
    print("🔍 步骤2: 检查预处理数据")
    print("=" * 60)
    
    preprocessed_dir = config.get('preprocessed_data_dir')
    if not preprocessed_dir:
        print("❌ 未指定预处理数据目录")
        return False
    
    hdf5_path = Path(preprocessed_dir) / 'preprocessed_data.h5'
    metadata_path = Path(preprocessed_dir) / 'data_metadata.json'
    
    print(f"预处理数据目录: {preprocessed_dir}")
    print(f"HDF5文件: {hdf5_path}")
    print(f"元数据文件: {metadata_path}")
    
    if not hdf5_path.exists():
        print(f"❌ HDF5文件不存在: {hdf5_path}")
        return False
    
    if not metadata_path.exists():
        print(f"❌ 元数据文件不存在: {metadata_path}")
        return False
    
    # 检查文件大小
    hdf5_size = hdf5_path.stat().st_size / (1024*1024)  # MB
    metadata_size = metadata_path.stat().st_size / 1024  # KB
    
    print(f"✅ HDF5文件大小: {hdf5_size:.2f} MB")
    print(f"✅ 元数据文件大小: {metadata_size:.2f} KB")
    
    return True

def check_metadata_loading(config):
    """检查元数据加载"""
    print("\n" + "=" * 60)
    print("🔍 步骤3: 检查元数据加载")
    print("=" * 60)
    
    try:
        metadata_path = Path(config['preprocessed_data_dir']) / 'data_metadata.json'
        
        print("正在加载元数据...")
        start_time = time.time()
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        load_time = time.time() - start_time
        
        print(f"✅ 元数据加载成功，耗时: {load_time:.2f} 秒")
        print(f"   数据项数量: {len(metadata.get('items', {}))}")
        print(f"   患者数量: {len(metadata.get('patient_mapping', {}))}")
        print(f"   文件夹数量: {len(metadata.get('folder_mapping', {}))}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ 元数据加载失败: {e}")
        return None

def check_data_splitting(config, metadata):
    """检查数据分割"""
    print("\n" + "=" * 60)
    print("🔍 步骤4: 检查数据分割")
    print("=" * 60)
    
    try:
        from sklearn.model_selection import train_test_split
        
        all_item_ids = list(metadata['items'].keys())
        print(f"总数据项: {len(all_item_ids)}")
        
        start_time = time.time()
        
        # 执行数据分割
        train_ids, val_ids = train_test_split(
            all_item_ids,
            train_size=config.get('train_val_split', 0.8),
            random_state=config.get('seed', 42),
            shuffle=True
        )
        
        split_time = time.time() - start_time
        
        print(f"✅ 数据分割完成，耗时: {split_time:.2f} 秒")
        print(f"   训练集: {len(train_ids)} 项")
        print(f"   验证集: {len(val_ids)} 项")
        
        return train_ids, val_ids
        
    except Exception as e:
        print(f"❌ 数据分割失败: {e}")
        return None, None

def check_hdf5_access(config):
    """检查HDF5文件访问"""
    print("\n" + "=" * 60)
    print("🔍 步骤5: 检查HDF5文件访问")
    print("=" * 60)
    
    try:
        import h5py
        
        hdf5_path = Path(config['preprocessed_data_dir']) / 'preprocessed_data.h5'
        
        print("正在打开HDF5文件...")
        start_time = time.time()
        
        with h5py.File(hdf5_path, 'r') as f:
            open_time = time.time() - start_time
            
            print(f"✅ HDF5文件打开成功，耗时: {open_time:.2f} 秒")
            print(f"   根组: {list(f.keys())}")
            
            if 'images' in f:
                images_group = f['images']
                print(f"   图像组大小: {len(images_group)} 项")
                
                # 测试读取第一个数据项
                if len(images_group) > 0:
                    first_key = list(images_group.keys())[0]
                    print(f"   测试读取第一个数据项: {first_key}")
                    
                    start_time = time.time()
                    test_data = images_group[first_key][:]
                    read_time = time.time() - start_time
                    
                    print(f"   ✅ 数据读取成功，耗时: {read_time:.2f} 秒")
                    print(f"   数据形状: {test_data.shape}")
                    print(f"   数据类型: {test_data.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ HDF5文件访问失败: {e}")
        return False

def check_dataset_creation(config, train_ids, val_ids):
    """检查数据集创建"""
    print("\n" + "=" * 60)
    print("🔍 步骤6: 检查数据集创建")
    print("=" * 60)
    
    try:
        from merlin.training.fast_dataloader import FastCardiacDataset
        
        hdf5_path = Path(config['preprocessed_data_dir']) / 'preprocessed_data.h5'
        metadata_path = Path(config['preprocessed_data_dir']) / 'data_metadata.json'
        
        # 使用较小的样本进行测试
        test_train_ids = train_ids[:10] if len(train_ids) > 10 else train_ids
        test_val_ids = val_ids[:5] if len(val_ids) > 5 else val_ids
        
        print(f"测试训练集创建（{len(test_train_ids)} 项）...")
        start_time = time.time()
        
        # 创建训练数据集，不预加载数据
        train_dataset = FastCardiacDataset(
            hdf5_path=str(hdf5_path),
            metadata_path=str(metadata_path),
            item_ids=test_train_ids,
            enable_cache=False,  # 关闭缓存
            preload_data=False   # 不预加载数据
        )
        
        creation_time = time.time() - start_time
        
        print(f"✅ 训练数据集创建成功，耗时: {creation_time:.2f} 秒")
        print(f"   数据集大小: {len(train_dataset)}")
        
        # 测试数据访问
        print("测试数据访问...")
        start_time = time.time()
        
        sample = train_dataset[0]
        
        access_time = time.time() - start_time
        
        print(f"✅ 数据访问成功，耗时: {access_time:.2f} 秒")
        print(f"   图像形状: {sample['image'].shape}")
        print(f"   心脏功能指标: {sample['cardiac_metrics']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return False

def check_dataloader_creation(config, train_ids, val_ids):
    """检查数据加载器创建"""
    print("\n" + "=" * 60)
    print("🔍 步骤7: 检查数据加载器创建")
    print("=" * 60)
    
    try:
        from merlin.training.fast_dataloader import FastCardiacDataset
        from torch.utils.data import DataLoader
        
        hdf5_path = Path(config['preprocessed_data_dir']) / 'preprocessed_data.h5'
        metadata_path = Path(config['preprocessed_data_dir']) / 'data_metadata.json'
        
        # 使用较小的样本进行测试
        test_train_ids = train_ids[:10] if len(train_ids) > 10 else train_ids
        
        print(f"创建数据集（{len(test_train_ids)} 项）...")
        train_dataset = FastCardiacDataset(
            hdf5_path=str(hdf5_path),
            metadata_path=str(metadata_path),
            item_ids=test_train_ids,
            enable_cache=False,
            preload_data=False
        )
        
        # 创建数据加载器，使用较少的工作进程
        print("创建数据加载器...")
        start_time = time.time()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,  # 小批次
            shuffle=True,
            num_workers=0,  # 不使用多进程
            pin_memory=False,
            drop_last=True
        )
        
        creation_time = time.time() - start_time
        
        print(f"✅ 数据加载器创建成功，耗时: {creation_time:.2f} 秒")
        
        # 测试数据迭代
        print("测试数据迭代...")
        start_time = time.time()
        
        batch = next(iter(train_loader))
        
        iter_time = time.time() - start_time
        
        print(f"✅ 数据迭代成功，耗时: {iter_time:.2f} 秒")
        print(f"   批次大小: {batch['image'].shape[0]}")
        print(f"   图像形状: {batch['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始数据加载器debug")
    print("=" * 80)
    
    # 步骤1: 检查配置文件
    config = check_config_file()
    if not config:
        return
    
    # 步骤2: 检查预处理数据
    if not check_preprocessed_data(config):
        return
    
    # 步骤3: 检查元数据加载
    metadata = check_metadata_loading(config)
    if not metadata:
        return
    
    # 步骤4: 检查数据分割
    train_ids, val_ids = check_data_splitting(config, metadata)
    if not train_ids or not val_ids:
        return
    
    # 步骤5: 检查HDF5文件访问
    if not check_hdf5_access(config):
        return
    
    # 步骤6: 检查数据集创建
    if not check_dataset_creation(config, train_ids, val_ids):
        return
    
    # 步骤7: 检查数据加载器创建
    if not check_dataloader_creation(config, train_ids, val_ids):
        return
    
    print("\n" + "=" * 80)
    print("🎉 所有检查通过！")
    print("=" * 80)
    
    print("\n💡 建议的解决方案:")
    print("1. 如果在步骤6或7失败，可能是配置问题")
    print("2. 尝试减少num_workers的值")
    print("3. 设置preload_train_data和preload_val_data为false")
    print("4. 减少cache_size的值")
    print("5. 减少batch_size的值")

if __name__ == '__main__':
    main() 