#!/usr/bin/env python3
"""
简单的心脏功能预测示例

这个示例展示了如何使用Merlin进行心脏功能预测的基本流程。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from merlin.models.cardiac_regression import CardiacFunctionModel, CardiacMetricsCalculator


def simple_cardiac_prediction_example():
    """简单的心脏功能预测示例"""
    
    print("=== Merlin 心脏功能预测示例 ===\n")
    
    # 1. 创建心脏功能预测模型
    print("1. 创建心脏功能预测模型...")
    model = CardiacFunctionModel(
        pretrained_model_path=None  # 在实际使用中，这里应该是Merlin预训练权重路径
    )
    model.eval()
    
    print(f"   模型设备: {next(model.parameters()).device}")
    print(f"   心脏功能指标数量: 2 (LVEF + AS)")
    
    # 2. 模拟CT图像输入（实际使用中应该是真实的CT数据）
    print("\n2. 准备输入数据...")
    batch_size = 1
    # CT图像格式: [batch_size, channels, depth, height, width]
    # Merlin期望的输入格式: [1, 1, 160, 224, 224]
    fake_ct_image = torch.randn(batch_size, 1, 160, 224, 224)
    
    print(f"   输入CT图像形状: {fake_ct_image.shape}")
    print(f"   图像值范围: [{fake_ct_image.min():.2f}, {fake_ct_image.max():.2f}]")
    
    # 3. 执行心脏功能预测
    print("\n3. 执行心脏功能预测...")
    with torch.no_grad():
        lvef_predictions, as_predictions = model(fake_ct_image)
    
    print(f"   LVEF预测形状: {lvef_predictions.shape}")
    print(f"   AS预测形状: {as_predictions.shape}")
    
    # 4. 处理预测结果
    print("\n4. 处理预测结果...")
    lvef_value = lvef_predictions.squeeze().item()
    as_probability = as_predictions.squeeze().item()
    as_prediction = "阳性" if as_probability > 0.5 else "阴性"
    
    # 5. 显示预测结果
    print("\n5. 心脏功能预测结果:")
    print("-" * 50)
    
    print(f"   左室射血分数 (LVEF): {lvef_value:8.2f}")
    print(f"   主动脉瓣狭窄概率:   {as_probability:8.3f}")
    print(f"   主动脉瓣狭窄预测:   {as_prediction}")
    
    # 6. 生成简单的评估报告
    print("\n6. 生成评估报告:")
    print("-" * 50)
    
    # LVEF评估
    if lvef_value >= 0.5:  # 假设模型输出已标准化，正常LVEF约55-70%
        lvef_status = "正常"
    elif lvef_value >= 0:
        lvef_status = "轻度降低"
    elif lvef_value >= -0.5:
        lvef_status = "中度降低"
    else:
        lvef_status = "重度降低"
    
    # AS评估
    if as_probability > 0.7:
        as_status = "高风险"
    elif as_probability > 0.3:
        as_status = "中等风险"
    else:
        as_status = "低风险"
    
    print(f"   LVEF评估: {lvef_status}")
    print(f"   AS风险评估: {as_status}")
    
    # 7. 保存结果（可选）
    print("\n7. 保存预测结果...")
    results = {
        'lvef': float(lvef_value),
        'as_probability': float(as_probability),
        'as_prediction': as_prediction,
        'lvef_status': lvef_status,
        'as_status': as_status
    }
    
    print(f"   预测结果已准备好保存: {len(results)} 个指标")
    
    print("\n=== 示例完成 ===")
    print("\n注意事项:")
    print("- 本示例使用模拟数据，实际使用需要真实的CT图像")
    print("- 预测结果仅供演示，不应用于实际诊断")
    print("- 在实际应用中需要使用训练好的模型权重")
    
    return results


def cardiac_model_info():
    """显示心脏功能模型的详细信息"""
    
    print("\n=== 心脏功能模型信息 ===")
    
    # 模型架构信息
    model = CardiacFunctionModel()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 心脏功能任务信息
    print(f"\n心脏功能预测任务:")
    print(f"  1. LVEF回归 - 左室射血分数预测")
    print(f"  2. AS分类 - 主动脉瓣狭窄检测")
    
    # 输入输出格式
    print(f"\n输入输出格式:")
    print(f"  输入CT图像: [batch_size, 1, 160, 224, 224]")
    print(f"  LVEF输出: [batch_size, 1] (回归值)")
    print(f"  AS输出: [batch_size, 1] (概率值)")


if __name__ == "__main__":
    # 运行简单示例
    simple_cardiac_prediction_example()
    
    # 显示模型信息
    cardiac_model_info() 