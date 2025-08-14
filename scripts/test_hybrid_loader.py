#!/usr/bin/env python3
"""
测试混合数据加载器
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_hybrid_loader():
    """测试混合数据加载器"""
    
    try:
        from merlin.training.fast_dataloader import HybridCardiacDataset
        print("✅ 成功导入 HybridCardiacDataset")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # 配置参数
    csv_path = "/pasteur2/u/xhanwang/Chest_CT_Cardiac/merged_ct_echo_data.csv"
    hdf5_path = "/pasteur2/u/xhanwang/Chest_CT_Cardiac/outputs/cardiac_training_as_maybe/preprocessed_data.h5"
    label_columns = ['lvef', 'AS_maybe']
    
    print(f"📊 CSV文件: {csv_path}")
    print(f"🖼️ HDF5文件: {hdf5_path}")
    print(f"🏷️ 标签列: {label_columns}")
    
    # 检查文件存在
    if not os.path.exists(csv_path):
        print(f"❌ CSV文件不存在: {csv_path}")
        return False
    
    if not os.path.exists(hdf5_path):
        print(f"❌ HDF5文件不存在: {hdf5_path}")
        return False
    
    try:
        # 创建数据集
        print("\n🔄 创建混合数据集...")
        dataset = HybridCardiacDataset(
            csv_path=csv_path,
            hdf5_path=hdf5_path,
            enable_cache=True,
            cache_size=10,
            label_columns=label_columns
        )
        
        print(f"✅ 数据集创建成功，包含 {len(dataset)} 个样本")
        
        # 测试获取一个样本
        if len(dataset) > 0:
            print("\n🔍 测试获取样本...")
            sample = dataset[0]
            
            print(f"样本键: {list(sample.keys())}")
            print(f"图像形状: {sample['image'].shape}")
            print(f"标签: {sample['labels']}")
            print(f"心脏指标: {sample['cardiac_metrics']}")
            print(f"患者ID: {sample['patient_id']}")
            
            print("✅ 样本获取成功！")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 测试混合数据加载器")
    print("=" * 50)
    
    success = test_hybrid_loader()
    
    if success:
        print("\n🎉 测试通过！混合数据加载器工作正常。")
        print("现在可以使用以下命令开始训练:")
        print("python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json")
    else:
        print("\n❌ 测试失败，请检查错误信息。") 