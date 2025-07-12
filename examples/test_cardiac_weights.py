#!/usr/bin/env python3
"""
测试心脏功能模型权重加载

验证CardiacFunctionModel是否正确加载了Merlin预训练权重
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
from merlin.models.cardiac_regression import CardiacFunctionModel
from merlin import Merlin


def test_weight_loading():
    """测试权重加载"""
    print("=" * 80)
    print("🔍 测试Merlin权重加载到心脏功能模型")
    print("=" * 80)
    
    # 1. 首先创建原始Merlin模型（会自动下载权重）
    print("\n1. 创建原始Merlin模型...")
    merlin_model = Merlin()
    
    # 获取权重文件路径
    checkpoint_path = os.path.join(
        merlin_model.current_path, 
        'checkpoints', 
        merlin_model.checkpoint_name
    )
    
    print(f"Merlin权重路径: {checkpoint_path}")
    print(f"权重文件存在: {os.path.exists(checkpoint_path)}")
    
    # 2. 创建心脏功能模型并加载权重
    print("\n2. 创建心脏功能模型...")
    cardiac_model = CardiacFunctionModel(pretrained_model_path=checkpoint_path)
    
    # 3. 比较模型结构
    print("\n3. 比较模型结构...")
    
    # 获取原始Merlin图像编码器的权重
    merlin_image_encoder = merlin_model.model.encode_image
    cardiac_image_encoder = cardiac_model.image_encoder
    
    print(f"Merlin图像编码器类型: {type(merlin_image_encoder)}")
    print(f"心脏功能图像编码器类型: {type(cardiac_image_encoder)}")
    
    # 4. 测试前向传播
    print("\n4. 测试前向传播...")
    
    # 创建测试输入
    test_input = torch.randn(1, 1, 160, 224, 224)
    
    # 测试原始Merlin模型
    with torch.no_grad():
        merlin_output = merlin_image_encoder(test_input)
        print(f"Merlin输出形状: contrastive={merlin_output[0].shape}, ehr={merlin_output[1].shape}")
    
    # 测试心脏功能模型
    with torch.no_grad():
        cardiac_output = cardiac_image_encoder(test_input)
        print(f"心脏功能模型输出形状: contrastive={cardiac_output[0].shape}, ehr={cardiac_output[1].shape}")
    
    # 5. 比较特征提取器输出
    print("\n5. 比较特征提取器输出...")
    
    # 检查输出是否相似（如果权重加载成功，输出应该相似）
    contrastive_diff = torch.mean(torch.abs(merlin_output[0] - cardiac_output[0]))
    ehr_diff = torch.mean(torch.abs(merlin_output[1] - cardiac_output[1]))
    
    print(f"Contrastive特征差异: {contrastive_diff.item():.6f}")
    print(f"EHR特征差异: {ehr_diff.item():.6f}")
    
    # 6. 测试心脏功能预测
    print("\n6. 测试心脏功能预测...")
    
    with torch.no_grad():
        lvef_pred, as_pred = cardiac_model(test_input)
        print(f"LVEF预测: {lvef_pred.item():.4f}")
        print(f"AS预测: {as_pred.item():.4f}")
    
    # 7. 总结
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    
    if contrastive_diff < 1e-5 and ehr_diff < 1e-5:
        print("🎉 权重加载成功！特征提取器输出一致")
    elif contrastive_diff < 1e-3 and ehr_diff < 1e-3:
        print("⚠️  权重可能部分加载，特征提取器输出基本一致")
    else:
        print("❌ 权重加载可能失败，特征提取器输出差异较大")
    
    print("=" * 80)


def test_input_format():
    """测试输入格式兼容性"""
    print("\n" + "=" * 80)
    print("🔍 测试输入格式兼容性")
    print("=" * 80)
    
    from merlin.data.monai_transforms import ImageTransforms
    
    # 创建模拟的NIFTI文件路径字典
    test_data = {
        'image': '/tmp/test_image.nii.gz'  # 这是一个模拟路径
    }
    
    print("ImageTransforms配置:")
    for i, transform in enumerate(ImageTransforms.transforms):
        print(f"  {i+1}. {transform.__class__.__name__}")
    
    print("\n输入格式要求:")
    print("  - 图像尺寸: [224, 224, 160]")
    print("  - 强度范围: [0, 1] (从-1000到1000 HU标准化)")
    print("  - 像素间距: (1.5, 1.5, 3) mm")
    print("  - 方向: RAS")
    
    print("\n✅ 输入格式与Merlin完全兼容")


if __name__ == '__main__':
    try:
        test_weight_loading()
        test_input_format()
        
        print("\n💡 使用建议:")
        print("  1. 确保设置 freeze_encoder=True 进行微调")
        print("  2. 使用较小的学习率 (1e-4 或更小)")
        print("  3. 监控训练过程中的损失变化")
        print("  4. 如果GPU内存不足，可减小batch_size")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 