#!/usr/bin/env python3
"""
调试心脏功能训练的脚本
检查损失函数的数值稳定性和输入输出范围
"""

import torch
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent))

from merlin.models.cardiac_regression import CardiacFunctionModel, CardiacLoss


def test_model_output_ranges():
    """测试模型输出范围"""
    print("🔍 测试模型输出范围...")
    
    # 创建模型
    model = CardiacFunctionModel()
    model.eval()
    
    # 创建随机输入
    batch_size = 4
    dummy_image = torch.randn(batch_size, 1, 16, 224, 224)
    
    with torch.no_grad():
        lvef_pred, as_pred = model(dummy_image)
        
        print(f"✅ 模型输出:")
        print(f"   LVEF预测范围: [{lvef_pred.min():.6f}, {lvef_pred.max():.6f}]")
        print(f"   AS预测范围: [{as_pred.min():.6f}, {as_pred.max():.6f}]")
        print(f"   AS预测值是否在[0,1]范围内: {torch.all(as_pred >= 0) and torch.all(as_pred <= 1)}")
    
    return lvef_pred, as_pred


def test_loss_function_stability():
    """测试损失函数稳定性"""
    print("\n🔍 测试损失函数稳定性...")
    
    # 创建损失函数
    criterion = CardiacLoss()
    
    # 测试正常情况
    print("\n📊 测试正常输入:")
    lvef_pred = torch.randn(4, 1) * 20 + 60  # 模拟LVEF预测值
    as_pred = torch.sigmoid(torch.randn(4, 1))  # 模拟AS预测概率
    lvef_true = torch.randn(4) * 10 + 55  # 模拟LVEF真实值
    as_true = torch.randint(0, 2, (4,)).float()  # 模拟AS真实标签
    
    print(f"   LVEF预测: {lvef_pred.squeeze().tolist()}")
    print(f"   AS预测: {as_pred.squeeze().tolist()}")
    print(f"   LVEF真实: {lvef_true.tolist()}")
    print(f"   AS真实: {as_true.tolist()}")
    
    try:
        loss_dict = criterion(lvef_pred, as_pred, lvef_true, as_true)
        print(f"✅ 正常情况损失计算成功:")
        print(f"   总损失: {loss_dict['total_loss']:.6f}")
        print(f"   回归损失: {loss_dict['regression_loss']:.6f}")
        print(f"   分类损失: {loss_dict['classification_loss']:.6f}")
    except Exception as e:
        print(f"❌ 正常情况损失计算失败: {e}")
    
    # 测试边界情况
    print("\n📊 测试边界情况:")
    
    # 测试AS预测值接近0和1的情况
    as_pred_boundary = torch.tensor([[0.0001], [0.9999], [0.5], [0.001]])
    as_true_boundary = torch.tensor([0.0, 1.0, 0.0, 1.0])
    
    print(f"   AS预测边界值: {as_pred_boundary.squeeze().tolist()}")
    print(f"   AS真实边界值: {as_true_boundary.tolist()}")
    
    try:
        loss_dict = criterion(lvef_pred, as_pred_boundary, lvef_true, as_true_boundary)
        print(f"✅ 边界情况损失计算成功:")
        print(f"   总损失: {loss_dict['total_loss']:.6f}")
        print(f"   分类损失: {loss_dict['classification_loss']:.6f}")
    except Exception as e:
        print(f"❌ 边界情况损失计算失败: {e}")
    
    # 测试极端情况
    print("\n📊 测试极端情况:")
    
    # 测试AS预测值为0或1的情况
    as_pred_extreme = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    as_true_extreme = torch.tensor([0.0, 1.0, 1.0, 0.0])
    
    print(f"   AS预测极端值: {as_pred_extreme.squeeze().tolist()}")
    print(f"   AS真实极端值: {as_true_extreme.tolist()}")
    
    try:
        loss_dict = criterion(lvef_pred, as_pred_extreme, lvef_true, as_true_extreme)
        print(f"✅ 极端情况损失计算成功:")
        print(f"   总损失: {loss_dict['total_loss']:.6f}")
        print(f"   分类损失: {loss_dict['classification_loss']:.6f}")
    except Exception as e:
        print(f"❌ 极端情况损失计算失败: {e}")


def test_gradient_flow():
    """测试梯度流动"""
    print("\n🔍 测试梯度流动...")
    
    # 创建模型和损失函数
    model = CardiacFunctionModel()
    criterion = CardiacLoss()
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 模拟训练步骤
    dummy_image = torch.randn(2, 1, 16, 224, 224)
    lvef_true = torch.tensor([60.0, 50.0])
    as_true = torch.tensor([0.0, 1.0])
    
    # 前向传播
    lvef_pred, as_pred = model(dummy_image)
    
    # 计算损失
    loss_dict = criterion(lvef_pred, as_pred, lvef_true, as_true)
    loss = loss_dict['total_loss']
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    total_grad_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2)
            total_grad_norm += param_grad_norm.item() ** 2
            param_count += 1
    
    total_grad_norm = total_grad_norm ** (1. / 2)
    
    print(f"✅ 梯度流动测试:")
    print(f"   总梯度范数: {total_grad_norm:.6f}")
    print(f"   参数数量: {param_count}")
    print(f"   梯度是否正常: {0 < total_grad_norm < 1000}")
    
    # 执行优化步骤
    optimizer.step()
    print(f"✅ 优化步骤执行成功")


def test_cuda_compatibility():
    """测试CUDA兼容性"""
    print("\n🔍 测试CUDA兼容性...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过CUDA测试")
        return
    
    device = torch.device('cuda')
    
    # 创建模型和损失函数
    model = CardiacFunctionModel().to(device)
    criterion = CardiacLoss()
    
    # 创建输入数据
    dummy_image = torch.randn(2, 1, 16, 224, 224).to(device)
    lvef_true = torch.tensor([60.0, 50.0]).to(device)
    as_true = torch.tensor([0.0, 1.0]).to(device)
    
    try:
        # 前向传播
        lvef_pred, as_pred = model(dummy_image)
        
        # 计算损失
        loss_dict = criterion(lvef_pred, as_pred, lvef_true, as_true)
        loss = loss_dict['total_loss']
        
        # 反向传播
        loss.backward()
        
        print(f"✅ CUDA测试成功:")
        print(f"   设备: {device}")
        print(f"   损失值: {loss.item():.6f}")
        print(f"   LVEF预测范围: [{lvef_pred.min():.6f}, {lvef_pred.max():.6f}]")
        print(f"   AS预测范围: [{as_pred.min():.6f}, {as_pred.max():.6f}]")
        
    except Exception as e:
        print(f"❌ CUDA测试失败: {e}")


def main():
    """主函数"""
    print("=" * 80)
    print("🩺 心脏功能训练调试脚本")
    print("=" * 80)
    print("这个脚本将测试修复后的损失函数是否能正确处理数值稳定性问题")
    print()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    test_model_output_ranges()
    test_loss_function_stability()
    test_gradient_flow()
    test_cuda_compatibility()
    
    print("\n" + "=" * 80)
    print("🎉 调试测试完成！")
    print("=" * 80)
    print("💡 如果所有测试都通过，说明修复成功。")
    print("💡 如果仍有问题，请检查:")
    print("   1. PyTorch版本是否兼容")
    print("   2. CUDA版本是否正确")
    print("   3. 数据预处理是否正确")
    print("   4. 模型输入数据格式是否正确")
    print("=" * 80)


if __name__ == '__main__':
    main() 