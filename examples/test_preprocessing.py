#!/usr/bin/env python3
"""
数据预处理测试脚本

该脚本帮助用户逐步测试数据预处理系统，确保系统在处理大量数据之前能够正常工作。

使用方法:
    python test_preprocessing.py
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from merlin.training.data_preprocessor import DataPreprocessor
from merlin.training.cardiac_trainer import load_and_validate_csv_data, build_data_list


def test_single_item():
    """测试单个数据项的处理"""
    print("🧪 测试1：单个数据项处理")
    print("=" * 50)
    
    # 加载配置
    config_path = Path(__file__).parent / 'config_fast_training.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置测试输出目录
    config['output_dir'] = 'outputs/test_preprocessing'
    
    try:
        # 加载数据
        print("📂 加载数据...")
        df, cardiac_metric_columns = load_and_validate_csv_data(config)
        data_list = build_data_list(df, config, cardiac_metric_columns)
        
        if not data_list:
            print("❌ 没有找到有效的数据")
            return False
        
        # 选择第一个数据项
        test_item = data_list[0]
        print(f"📄 测试数据项: {test_item['basename']}")
        print(f"📁 图像路径: {test_item['image']}")
        
        # 检查文件是否存在
        if not os.path.exists(test_item['image']):
            print(f"❌ 图像文件不存在: {test_item['image']}")
            return False
        
        # 创建预处理器
        preprocessor = DataPreprocessor(config)
        
        # 处理单个数据项
        print("🔄 开始处理...")
        result = preprocessor.preprocess_single_item(test_item)
        
        if result:
            item_id, storage_data = result
            print(f"✅ 处理成功!")
            print(f"   数据ID: {item_id}")
            print(f"   图像形状: {storage_data['processed_shape']}")
            print(f"   数据类型: {storage_data['data_type']}")
            return True
        else:
            print("❌ 处理失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_small_batch():
    """测试小批量数据处理"""
    print("\n🧪 测试2：小批量数据处理")
    print("=" * 50)
    
    # 加载配置
    config_path = Path(__file__).parent / 'config_fast_training.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设置测试配置
    config['output_dir'] = 'outputs/test_preprocessing'
    config['preprocess_batch_size'] = 4
    config['enable_multiprocessing'] = False
    
    try:
        # 加载数据
        print("📂 加载数据...")
        df, cardiac_metric_columns = load_and_validate_csv_data(config)
        data_list = build_data_list(df, config, cardiac_metric_columns)
        
        if not data_list:
            print("❌ 没有找到有效的数据")
            return False
        
        # 选择前10个数据项
        test_data = data_list[:10]
        print(f"📄 测试数据项数量: {len(test_data)}")
        
        # 创建预处理器
        preprocessor = DataPreprocessor(config)
        
        # 处理小批量数据
        print("🔄 开始处理...")
        preprocessor.preprocess_data(test_data, force_reprocess=True)
        
        # 验证结果
        hdf5_path = Path(config['output_dir']) / 'preprocessed_data.h5'
        if hdf5_path.exists():
            print(f"✅ 预处理完成!")
            print(f"   HDF5文件大小: {hdf5_path.stat().st_size / (1024*1024):.2f} MB")
            return True
        else:
            print("❌ 预处理失败：未生成HDF5文件")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n🧪 测试3：数据加载测试")
    print("=" * 50)
    
    # 加载配置
    config_path = Path(__file__).parent / 'config_fast_training.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config['preprocessed_data_dir'] = 'outputs/test_preprocessing'
    
    try:
        from merlin.training.fast_dataloader import FastDataLoaderManager
        
        # 创建数据加载器管理器
        print("📂 创建数据加载器...")
        manager = FastDataLoaderManager(config)
        
        # 获取数据统计
        stats = manager.get_data_statistics()
        print(f"📊 数据统计:")
        print(f"   总数据项: {stats['total_items']}")
        print(f"   HDF5文件大小: {stats['hdf5_file_size_mb']} MB")
        
        # 创建数据加载器
        print("🔄 创建数据加载器...")
        train_loader, val_loader = manager.create_data_loaders()
        
        print(f"✅ 数据加载器创建成功!")
        print(f"   训练集大小: {len(train_loader.dataset)}")
        print(f"   验证集大小: {len(val_loader.dataset)}")
        
        # 测试数据读取
        print("🔄 测试数据读取...")
        batch = next(iter(train_loader))
        print(f"✅ 数据读取成功!")
        print(f"   批次大小: {batch['image'].shape[0]}")
        print(f"   图像形状: {batch['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始数据预处理系统测试")
    print("=" * 80)
    
    # 创建测试输出目录
    test_output = Path('outputs/test_preprocessing')
    test_output.mkdir(parents=True, exist_ok=True)
    
    # 运行测试
    tests = [
        ("单个数据项处理", test_single_item),
        ("小批量数据处理", test_small_batch),
        ("数据加载测试", test_data_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 开始测试: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} 通过")
        else:
            print(f"❌ {test_name} 失败")
    
    print("\n" + "=" * 80)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！可以开始处理完整数据集。")
        print("\n💡 下一步:")
        print("   1. 处理完整数据集:")
        print("      python -m merlin.training.data_preprocessor --config examples/config_fast_training.json")
        print("   2. 如果需要多进程处理:")
        print("      python -m merlin.training.data_preprocessor --config examples/config_fast_training.json --enable_multiprocessing --num_workers 4")
    else:
        print("⚠️  部分测试失败，请检查配置和数据路径。")
        print("\n🔍 故障排除:")
        print("   1. 检查CSV文件路径是否正确")
        print("   2. 检查图像文件路径是否正确")
        print("   3. 确认有足够的磁盘空间")
        print("   4. 检查Python环境是否包含所有必需的依赖")


if __name__ == '__main__':
    main() 