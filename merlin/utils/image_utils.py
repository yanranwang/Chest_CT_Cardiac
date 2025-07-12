import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union


class CTImageProcessor:
    """CT图像处理工具类"""
    
    def __init__(self, target_size: Tuple[int, int, int] = (16, 224, 224)):
        """
        初始化CT图像处理器
        
        Args:
            target_size: 目标尺寸 (depth, height, width)
        """
        self.target_size = target_size
    
    @staticmethod
    def load_nifti(file_path: str) -> Tuple[np.ndarray, nib.Nifti1Header]:
        """
        加载NIfTI文件 (.nii 或 .nii.gz)
        
        Args:
            file_path: NIfTI文件路径
            
        Returns:
            image_data: 图像数据数组
            header: NIfTI头信息
        """
        print(f"正在加载NIfTI文件: {file_path}")
        
        # 加载NIfTI文件
        nifti_img = nib.load(file_path)
        image_data = nifti_img.get_fdata()
        header = nifti_img.header
        
        print(f"原始图像形状: {image_data.shape}")
        print(f"数据类型: {image_data.dtype}")
        print(f"像素间距: {header.get_zooms()}")
        print(f"数值范围: [{image_data.min():.2f}, {image_data.max():.2f}]")
        
        return image_data, header
    
    @staticmethod
    def normalize_intensity(image: np.ndarray, 
                          window_center: Optional[float] = None,
                          window_width: Optional[float] = None,
                          clip_range: Tuple[float, float] = (-1000, 3000)) -> np.ndarray:
        """
        标准化CT图像强度值
        
        Args:
            image: 输入图像
            window_center: 窗位 (如果提供则使用窗宽窗位)
            window_width: 窗宽
            clip_range: 裁剪范围 (HU值)
            
        Returns:
            normalized_image: 标准化后的图像 [0, 1]
        """
        # 裁剪异常值
        image = np.clip(image, clip_range[0], clip_range[1])
        
        if window_center is not None and window_width is not None:
            # 使用窗宽窗位标准化
            min_val = window_center - window_width / 2
            max_val = window_center + window_width / 2
            image = np.clip(image, min_val, max_val)
            image = (image - min_val) / (max_val - min_val)
        else:
            # 使用最小最大值标准化
            image = (image - image.min()) / (image.max() - image.min())
        
        return image.astype(np.float32)
    
    def resize_image(self, image: np.ndarray, 
                    method: str = 'trilinear') -> np.ndarray:
        """
        调整图像尺寸到目标大小
        
        Args:
            image: 输入图像 [H, W, D]
            method: 插值方法 ('trilinear', 'nearest')
            
        Returns:
            resized_image: 调整后的图像 [target_depth, target_height, target_width]
        """
        # 转换为tensor进行插值
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, D]
        
        # 重新排列维度为 [1, 1, D, H, W]
        image_tensor = image_tensor.permute(0, 1, 4, 2, 3)
        
        # 使用F.interpolate调整尺寸
        if method == 'trilinear':
            resized_tensor = F.interpolate(
                image_tensor, 
                size=self.target_size, 
                mode='trilinear', 
                align_corners=False
            )
        elif method == 'nearest':
            resized_tensor = F.interpolate(
                image_tensor, 
                size=self.target_size, 
                mode='nearest'
            )
        else:
            raise ValueError(f"不支持的插值方法: {method}")
        
        # 转换回numpy
        resized_image = resized_tensor.squeeze().numpy()  # [D, H, W]
        
        print(f"图像尺寸从 {image.shape} 调整到 {resized_image.shape}")
        
        return resized_image
    
    def preprocess_for_model(self, image_data: np.ndarray,
                           normalize: bool = True,
                           add_batch_dim: bool = True) -> torch.Tensor:
        """
        为模型预处理图像数据
        
        Args:
            image_data: 原始图像数据 [H, W, D] 或 [D, H, W]
            normalize: 是否进行强度标准化
            add_batch_dim: 是否添加批次维度
            
        Returns:
            processed_tensor: 预处理后的张量 [1, 1, D, H, W] 或 [1, D, H, W]
        """
        # 确保输入是numpy数组
        if isinstance(image_data, torch.Tensor):
            image_data = image_data.numpy()
        
        # 标准化强度值
        if normalize:
            image_data = self.normalize_intensity(image_data)
        
        # 调整尺寸
        resized_image = self.resize_image(image_data)
        
        # 转换为tensor
        tensor = torch.from_numpy(resized_image).float()
        
        # 添加通道维度 [D, H, W] -> [1, D, H, W]
        tensor = tensor.unsqueeze(0)
        
        # 添加批次维度 [1, D, H, W] -> [1, 1, D, H, W]
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)
        
        print(f"预处理完成，输出张量形状: {tensor.shape}")
        
        return tensor


def load_and_preprocess_ct(file_path: str, 
                          target_size: Tuple[int, int, int] = (16, 224, 224),
                          window_center: Optional[float] = None,
                          window_width: Optional[float] = None) -> torch.Tensor:
    """
    便捷函数：加载并预处理CT图像
    
    Args:
        file_path: CT图像文件路径 (.nii或.nii.gz)
        target_size: 目标尺寸 (depth, height, width)
        window_center: 窗位
        window_width: 窗宽
        
    Returns:
        processed_tensor: 预处理后的张量 [1, 1, D, H, W]
    """
    # 创建处理器
    processor = CTImageProcessor(target_size=target_size)
    
    # 加载图像
    image_data, header = processor.load_nifti(file_path)
    
    # 使用窗宽窗位（如果提供）
    if window_center is not None and window_width is not None:
        image_data = processor.normalize_intensity(
            image_data, window_center=window_center, window_width=window_width
        )
    
    # 预处理
    processed_tensor = processor.preprocess_for_model(
        image_data, normalize=(window_center is None)
    )
    
    return processed_tensor


# 使用示例
if __name__ == "__main__":
    # 示例：处理512x512x121的CT图像
    print("=== CT图像处理示例 ===")
    
    # 如果您有实际的nii.gz文件，可以这样使用：
    # file_path = "path/to/your/ct_scan.nii.gz"
    # processed_ct = load_and_preprocess_ct(file_path)
    
    # 这里我们创建一个模拟的512x512x121图像进行演示
    print("创建模拟的512x512x121 CT图像...")
    simulated_ct = np.random.randint(-1000, 3000, (512, 512, 121)).astype(np.float32)
    print(f"模拟CT图像形状: {simulated_ct.shape}")
    print(f"数值范围: [{simulated_ct.min():.0f}, {simulated_ct.max():.0f}] HU")
    
    # 创建处理器
    processor = CTImageProcessor(target_size=(16, 224, 224))
    
    # 预处理图像
    print("\n开始预处理...")
    processed_tensor = processor.preprocess_for_model(simulated_ct)
    
    print(f"\n预处理结果:")
    print(f"输出张量形状: {processed_tensor.shape}")
    print(f"数据类型: {processed_tensor.dtype}")
    print(f"数值范围: [{processed_tensor.min():.4f}, {processed_tensor.max():.4f}]")
    
    # 演示如何使用不同的窗宽窗位设置
    print("\n=== 窗宽窗位示例 ===")
    
    # 胸部CT常用窗设置
    window_settings = {
        "肺窗": {"center": -600, "width": 1600},
        "纵隔窗": {"center": 50, "width": 400},
        "骨窗": {"center": 300, "width": 1500}
    }
    
    for window_name, settings in window_settings.items():
        normalized = processor.normalize_intensity(
            simulated_ct, 
            window_center=settings["center"], 
            window_width=settings["width"]
        )
        print(f"{window_name}: 窗位={settings['center']}, 窗宽={settings['width']}")
        print(f"  标准化后范围: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    print("\n=== 使用示例 ===")
    print("# 加载真实的CT文件:")
    print("file_path = 'your_ct_scan.nii.gz'")
    print("processed_ct = load_and_preprocess_ct(file_path)")
    print("print(f'处理后形状: {processed_ct.shape}')")
    print("")
    print("# 使用肺窗设置:")
    print("processed_ct = load_and_preprocess_ct(")
    print("    file_path, window_center=-600, window_width=1600")
    print(")")
    print("")
    print("# 然后可以直接输入到心脏功能预测模型:")
    print("from merlin.models.cardiac_regression import CardiacFunctionModel")
    print("model = CardiacFunctionModel()")
    print("lvef_pred, as_pred = model(processed_ct)") 