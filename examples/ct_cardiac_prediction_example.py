#!/usr/bin/env python3
"""
CT图像心脏功能预测示例

本示例展示如何：
1. 使用nibabel读取nii.gz格式的CT图像
2. 预处理512x512x121等大尺寸图像
3. 使用心脏功能预测模型进行分析
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from merlin.utils.image_utils import CTImageProcessor, load_and_preprocess_ct
from merlin.models.cardiac_regression import CardiacFunctionModel, CardiacMetricsCalculator


def predict_cardiac_function_from_ct(ct_file_path: str, 
                                   model_path: str = None,
                                   window_center: float = None,
                                   window_width: float = None,
                                   device: str = 'auto') -> dict:
    """
    从CT图像预测心脏功能
    
    Args:
        ct_file_path: CT图像文件路径 (.nii或.nii.gz)
        model_path: 预训练模型路径（可选）
        window_center: 窗位设置
        window_width: 窗宽设置
        device: 计算设备
        
    Returns:
        prediction_results: 预测结果字典
    """
    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    print(f"使用设备: {device}")
    print(f"CT文件路径: {ct_file_path}")
    
    # 1. 加载和预处理CT图像
    print("\n=== 第1步: 加载和预处理CT图像 ===")
    processed_ct = load_and_preprocess_ct(
        ct_file_path,
        target_size=(16, 224, 224),  # 模型期望的输入尺寸
        window_center=window_center,
        window_width=window_width
    )
    
    # 移动到指定设备
    processed_ct = processed_ct.to(device)
    print(f"预处理完成，张量形状: {processed_ct.shape}")
    
    # 2. 加载心脏功能预测模型
    print("\n=== 第2步: 加载心脏功能预测模型 ===")
    model = CardiacFunctionModel(pretrained_model_path=model_path)
    model = model.to(device)
    model.eval()
    print("模型加载完成")
    
    # 3. 执行预测
    print("\n=== 第3步: 执行心脏功能预测 ===")
    with torch.no_grad():
        lvef_pred, as_pred = model(processed_ct)
    
    # 转换为numpy数组
    lvef_value = lvef_pred.squeeze().cpu().numpy()
    as_probability = as_pred.squeeze().cpu().numpy()
    as_prediction = (as_probability > 0.5)
    
    # 4. 整理预测结果
    results = {
        'lvef': float(lvef_value),
        'as_probability': float(as_probability),
        'as_prediction': bool(as_prediction),
        'device_used': str(device),
        'input_shape': list(processed_ct.shape)
    }
    
    return results


def predict_full_cardiac_metrics_from_ct(ct_file_path: str,
                                        model_path: str = None,
                                        window_settings: dict = None) -> dict:
    """
    使用完整的心脏功能指标预测（演示版本）
    
    Args:
        ct_file_path: CT图像文件路径
        model_path: 模型路径
        window_settings: 窗宽窗位设置
        
    Returns:
        comprehensive_results: 完整的心脏功能分析结果
    """
    print("=== 完整心脏功能分析 ===")
    
    # 使用默认窗设置（胸部CT）
    if window_settings is None:
        window_settings = {"center": 50, "width": 400}  # 纵隔窗
    
    # 加载图像处理器
    processor = CTImageProcessor(target_size=(16, 224, 224))
    
    # 加载CT图像
    image_data, header = processor.load_nifti(ct_file_path)
    
    # 生成模拟的完整心脏功能预测（在真实应用中，这里应该是训练好的模型）
    print("\n生成模拟的心脏功能预测...")
    
    # 模拟预测值
    np.random.seed(42)  # 为了可重现的结果
    raw_predictions = np.random.randn(1, 10) * 0.5  # 较小的随机值
    
    # 使用CardiacMetricsCalculator标准化预测值
    normalized_predictions = CardiacMetricsCalculator.normalize_predictions(
        torch.from_numpy(raw_predictions)
    ).numpy()
    
    # 评估预测结果
    evaluation = CardiacMetricsCalculator.evaluate_predictions(
        normalized_predictions[0], return_status=True
    )
    
    # 整理结果
    comprehensive_results = {
        'image_info': {
            'original_shape': image_data.shape,
            'voxel_spacing': list(header.get_zooms()),
            'value_range': [float(image_data.min()), float(image_data.max())]
        },
        'window_settings': window_settings,
        'cardiac_metrics': evaluation
    }
    
    return comprehensive_results


def demonstrate_different_window_settings(ct_file_path: str):
    """
    演示不同窗宽窗位设置对图像处理的影响
    """
    print("=== 不同窗宽窗位设置演示 ===")
    
    # 定义不同的窗设置
    window_configs = {
        "肺窗": {"center": -600, "width": 1600},
        "纵隔窗": {"center": 50, "width": 400},
        "骨窗": {"center": 300, "width": 1500},
        "软组织窗": {"center": 40, "width": 80}
    }
    
    processor = CTImageProcessor()
    image_data, _ = processor.load_nifti(ct_file_path)
    
    print(f"\n原始图像统计:")
    print(f"  形状: {image_data.shape}")
    print(f"  数值范围: [{image_data.min():.1f}, {image_data.max():.1f}] HU")
    print(f"  均值: {image_data.mean():.1f} HU")
    print(f"  标准差: {image_data.std():.1f} HU")
    
    for window_name, settings in window_configs.items():
        normalized = processor.normalize_intensity(
            image_data,
            window_center=settings["center"],
            window_width=settings["width"]
        )
        
        print(f"\n{window_name} (窗位:{settings['center']}, 窗宽:{settings['width']}):")
        print(f"  标准化后范围: [{normalized.min():.4f}, {normalized.max():.4f}]")
        print(f"  均值: {normalized.mean():.4f}")
        print(f"  标准差: {normalized.std():.4f}")


def main():
    """主函数：演示完整的CT图像心脏功能预测流程"""
    
    print("CT图像心脏功能预测演示")
    print("=" * 60)
    
    # 模拟CT文件路径（在实际使用中，请提供真实的.nii.gz文件路径）
    # ct_file_path = "path/to/your/ct_scan.nii.gz"
    
    # 由于没有真实的CT文件，我们创建一个模拟的示例
    print("注意: 本演示使用模拟数据，在实际使用中请提供真实的CT文件路径")
    print("\n如果您有真实的CT文件，可以这样使用:")
    print("python ct_cardiac_prediction_example.py --ct_file your_scan.nii.gz")
    
    # 演示图像处理流程
    print("\n=== 模拟CT图像处理演示 ===")
    
    # 创建模拟的512x512x121 CT图像
    print("创建模拟的512x512x121 CT图像...")
    simulated_ct_data = np.random.randint(-1000, 3000, (512, 512, 121)).astype(np.float32)
    
    # 创建处理器并演示预处理
    processor = CTImageProcessor(target_size=(16, 224, 224))
    processed_tensor = processor.preprocess_for_model(simulated_ct_data)
    
    print(f"处理结果: {simulated_ct_data.shape} -> {processed_tensor.shape}")
    
    # 演示模型预测
    print("\n=== 模型预测演示 ===")
    model = CardiacFunctionModel()
    model.eval()
    
    with torch.no_grad():
        lvef_pred, as_pred = model(processed_tensor)
    
    print(f"LVEF预测: {lvef_pred.squeeze().item():.2f}")
    print(f"AS概率: {as_pred.squeeze().item():.3f}")
    
    # 演示完整的心脏功能指标
    print("\n=== 完整心脏功能指标演示 ===")
    
    # 模拟完整预测
    raw_predictions = torch.randn(1, 10) * 0.5
    normalized_predictions = CardiacMetricsCalculator.normalize_predictions(raw_predictions)
    evaluation = CardiacMetricsCalculator.evaluate_predictions(normalized_predictions[0])
    
    print("心脏功能评估结果:")
    for metric_name, metric_info in evaluation.items():
        print(f"  {metric_info['description']}: {metric_info['value']:.2f} ({metric_info['status']})")
    
    print("\n=== 使用说明 ===")
    print("在实际使用中，请按以下步骤操作:")
    print("1. 准备CT图像文件 (.nii 或 .nii.gz 格式)")
    print("2. 调用 load_and_preprocess_ct() 函数加载和预处理图像")
    print("3. 使用 CardiacFunctionModel 进行预测")
    print("4. 使用 CardiacMetricsCalculator 解释和评估结果")
    
    print("\n示例代码:")
    print("""
# 加载和预处理CT图像
from merlin.utils.image_utils import load_and_preprocess_ct
processed_ct = load_and_preprocess_ct('your_ct_scan.nii.gz')

# 心脏功能预测
from merlin.models.cardiac_regression import CardiacFunctionModel
model = CardiacFunctionModel()
lvef_pred, as_pred = model(processed_ct)

# 结果解释
print(f"LVEF预测: {lvef_pred.squeeze().item():.1f}%")
print(f"AS存在概率: {as_pred.squeeze().item():.2f}")
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CT图像心脏功能预测')
    parser.add_argument('--ct_file', type=str, help='CT图像文件路径 (.nii或.nii.gz)')
    parser.add_argument('--model_path', type=str, help='预训练模型路径')
    parser.add_argument('--window_center', type=float, help='窗位设置')
    parser.add_argument('--window_width', type=float, help='窗宽设置')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    
    args = parser.parse_args()
    
    if args.ct_file and os.path.exists(args.ct_file):
        # 如果提供了真实的CT文件
        print(f"处理CT文件: {args.ct_file}")
        
        # 执行预测
        results = predict_cardiac_function_from_ct(
            args.ct_file,
            model_path=args.model_path,
            window_center=args.window_center,
            window_width=args.window_width,
            device=args.device
        )
        
        print("\n=== 预测结果 ===")
        print(f"LVEF: {results['lvef']:.2f}%")
        print(f"AS概率: {results['as_probability']:.3f}")
        print(f"AS预测: {'存在' if results['as_prediction'] else '不存在'}")
        
        # 演示不同窗设置
        demonstrate_different_window_settings(args.ct_file)
        
        # 完整心脏功能分析
        comprehensive_results = predict_full_cardiac_metrics_from_ct(args.ct_file)
        print("\n=== 完整心脏功能分析 ===")
        for metric_name, metric_info in comprehensive_results['cardiac_metrics'].items():
            print(f"{metric_info['description']}: {metric_info['value']:.2f} ({metric_info['status']})")
    
    else:
        # 运行演示
        main() 