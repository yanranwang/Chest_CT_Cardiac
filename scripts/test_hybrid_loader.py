#!/usr/bin/env python3
"""
Test hybrid data loader
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_hybrid_loader():
    """Test hybrid data loader"""
    
    try:
        from merlin.training.fast_dataloader import HybridCardiacDataset
        print("✅ Successfully imported HybridCardiacDataset")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # configparameters
    csv_path = "/pasteur2/u/xhanwang/Chest_CT_Cardiac/merged_ct_echo_data.csv"
    hdf5_path = "/pasteur2/u/xhanwang/Chest_CT_Cardiac/outputs/cardiac_training_as_maybe/preprocessed_data.h5"
    label_columns = ['lvef', 'AS_maybe']
    
    print(f"📊 CSV文件: {csv_path}")
    print(f"🖼️ HDF5文件: {hdf5_path}")
    print(f"🏷️ 标签列: {label_columns}")
    
    # Checkfile存in
    if not os.path.exists(csv_path):
        print(f"❌ CSV file does not exist: {csv_path}")
        return False
    
    if not os.path.exists(hdf5_path):
        print(f"❌ HDF5 file does not exist: {hdf5_path}")
        return False
    
    try:
        # Createdata集
        print("\n🔄 Create混合数据集...")
        dataset = HybridCardiacDataset(
            csv_path=csv_path,
            hdf5_path=hdf5_path,
            enable_cache=True,
            cache_size=10,
            label_columns=label_columns
        )
        
        print(f"✅ 数据集Create成功，包含 {len(dataset)} samples")
        
        # TestGet一个sample
        if len(dataset) > 0:
            print("\n🔍 测试GetSample...")
            sample = dataset[0]
            
            print(f"Sample键: {list(sample.keys())}")
            print(f"图像形状: {sample['image'].shape}")
            print(f"标签: {sample['labels']}")
            print(f"心脏指标: {sample['cardiac_metrics']}")
            print(f"患者ID: {sample['patient_id']}")
            
            print("✅ SampleGet成功！")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Test hybrid data loader")
    print("=" * 50)
    
    success = test_hybrid_loader()
    
    if success:
        print("\n🎉 测试通过！混合数据Load器工作正常。")
        print("现在可以使用以下命令开始训练:")
        print("python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json")
    else:
        print("\n❌ 测试失败，请Check错误info。") 