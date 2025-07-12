import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union


class CTImageProcessor:
    """CT image processing utility class"""
    
    def __init__(self, target_size: Tuple[int, int, int] = (16, 224, 224)):
        """
        Initialize CT image processor
        
        Args:
            target_size: Target size (depth, height, width)
        """
        self.target_size = target_size
    
    @staticmethod
    def load_nifti(file_path: str) -> Tuple[np.ndarray, nib.Nifti1Header]:
        """
        Load NIfTI file (.nii or .nii.gz)
        
        Args:
            file_path: NIfTI file path
            
        Returns:
            image_data: Image data array
            header: NIfTI header information
        """
        print(f"Loading NIfTI file: {file_path}")
        
        # Load NIfTI file
        nifti_img = nib.load(file_path)
        image_data = nifti_img.get_fdata()
        header = nifti_img.header
        
        print(f"Original shape: {image_data.shape}")
        print(f"Data type: {image_data.dtype}")
        print(f"Pixel spacing: {header.get_zooms()}")
        print(f"Value range: [{image_data.min():.2f}, {image_data.max():.2f}]")
        
        return image_data, header
    
    @staticmethod
    def normalize_intensity(image: np.ndarray, 
                          window_center: Optional[float] = None,
                          window_width: Optional[float] = None,
                          clip_range: Tuple[float, float] = (-1000, 3000)) -> np.ndarray:
        """
        Normalize CT image intensity values
        
        Args:
            image: Input image
            window_center: Window center (if provided, use windowing)
            window_width: Window width
            clip_range: Clipping range (HU values)
            
        Returns:
            normalized_image: Normalized image [0, 1]
        """
        # Clip outliers
        image = np.clip(image, clip_range[0], clip_range[1])
        
        if window_center is not None and window_width is not None:
            # Use windowing normalization
            min_val = window_center - window_width / 2
            max_val = window_center + window_width / 2
            image = np.clip(image, min_val, max_val)
            image = (image - min_val) / (max_val - min_val)
        else:
            # Use min-max normalization
            image = (image - image.min()) / (image.max() - image.min())
        
        return image.astype(np.float32)
    
    def resize_image(self, image: np.ndarray, 
                    method: str = 'trilinear') -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image [H, W, D]
            method: Interpolation method ('trilinear', 'nearest')
            
        Returns:
            resized_image: Resized image [target_depth, target_height, target_width]
        """
        # Convert to tensor for interpolation
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, D]
        
        # Reorder dimensions to [1, 1, D, H, W]
        image_tensor = image_tensor.permute(0, 1, 4, 2, 3)
        
        # Resize using F.interpolate
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
            raise ValueError(f"Unsupported interpolation method: {method}")
        
        # Convert back to numpy
        resized_image = resized_tensor.squeeze().numpy()  # [D, H, W]
        
        print(f"Resized from {image.shape} to {resized_image.shape}")
        
        return resized_image
    
    def preprocess_for_model(self, image_data: np.ndarray,
                           normalize: bool = True,
                           add_batch_dim: bool = True) -> torch.Tensor:
        """
        Preprocess image data for model
        
        Args:
            image_data: Raw image data [H, W, D] or [D, H, W]
            normalize: Whether to normalize intensity
            add_batch_dim: Whether to add batch dimension
            
        Returns:
            processed_tensor: Preprocessed tensor [1, 1, D, H, W] or [1, D, H, W]
        """
        # Ensure input is numpy array
        if isinstance(image_data, torch.Tensor):
            image_data = image_data.numpy()
        
        # Normalize intensity
        if normalize:
            image_data = self.normalize_intensity(image_data)
        
        # Resize
        resized_image = self.resize_image(image_data)
        
        # Convert to tensor
        tensor = torch.from_numpy(resized_image).float()
        
        # Add channel dimension [D, H, W] -> [1, D, H, W]
        tensor = tensor.unsqueeze(0)
        
        # Add batch dimension [1, D, H, W] -> [1, 1, D, H, W]
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)
        
        print(f"Preprocessing complete, output shape: {tensor.shape}")
        
        return tensor


def load_and_preprocess_ct(file_path: str, 
                          target_size: Tuple[int, int, int] = (16, 224, 224),
                          window_center: Optional[float] = None,
                          window_width: Optional[float] = None) -> torch.Tensor:
    """
    Convenience function: load and preprocess CT image
    
    Args:
        file_path: CT image file path (.nii or .nii.gz)
        target_size: Target size (depth, height, width)
        window_center: Window center
        window_width: Window width
        
    Returns:
        processed_tensor: Preprocessed tensor [1, 1, D, H, W]
    """
    # Create processor
    processor = CTImageProcessor(target_size=target_size)
    
    # Load image
    image_data, header = processor.load_nifti(file_path)
    
    # Use windowing if provided
    if window_center is not None and window_width is not None:
        image_data = processor.normalize_intensity(
            image_data, window_center=window_center, window_width=window_width
        )
    
    # Preprocess
    processed_tensor = processor.preprocess_for_model(
        image_data, normalize=(window_center is None)
    )
    
    return processed_tensor


# Usage example
if __name__ == "__main__":
    # Example: process 512x512x121 CT image
    print("=== CT Image Processing Example ===")
    
    # If you have an actual nii.gz file, you can use it like this:
    # file_path = "path/to/your/ct_scan.nii.gz"
    # processed_ct = load_and_preprocess_ct(file_path)
    
    # Here we create a simulated 512x512x121 image for demonstration
    print("Creating simulated 512x512x121 CT image...")
    simulated_ct = np.random.randint(-1000, 3000, (512, 512, 121)).astype(np.float32)
    print(f"Simulated CT image shape: {simulated_ct.shape}")
    print(f"Value range: [{simulated_ct.min():.0f}, {simulated_ct.max():.0f}] HU")
    
    # Create processor
    processor = CTImageProcessor(target_size=(16, 224, 224))
    
    # Preprocess image
    print("\nStarting preprocessing...")
    processed_tensor = processor.preprocess_for_model(simulated_ct)
    
    print(f"\nPreprocessing result:")
    print(f"Output tensor shape: {processed_tensor.shape}")
    print(f"Data type: {processed_tensor.dtype}")
    print(f"Value range: [{processed_tensor.min():.4f}, {processed_tensor.max():.4f}]")
    
    # Demonstrate how to use different window center and width settings
    print("\n=== Window Center and Width Example ===")
    
    # Chest CT common window settings
    window_settings = {
        "Lung Window": {"center": -600, "width": 1600},
        "Mediastinum Window": {"center": 50, "width": 400},
        "Bone Window": {"center": 300, "width": 1500}
    }
    
    for window_name, settings in window_settings.items():
        normalized = processor.normalize_intensity(
            simulated_ct, 
            window_center=settings["center"], 
            window_width=settings["width"]
        )
        print(f"{window_name}: Window center={settings['center']}, Window width={settings['width']}")
        print(f"   Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    print("\n=== Usage Example ===")
    print("# Load real CT file:")
    print("file_path = 'your_ct_scan.nii.gz'")
    print("processed_ct = load_and_preprocess_ct(file_path)")
    print("print(f'Processed shape: {processed_ct.shape}')")
    print("")
    print("# Use Lung Window setting:")
    print("processed_ct = load_and_preprocess_ct(")
    print("    file_path, window_center=-600, window_width=1600")
    print(")")
    print("")
    print("# Then you can directly input to cardiac function prediction model:")
    print("from merlin.models.cardiac_regression import CardiacFunctionModel")
    print("model = CardiacFunctionModel()")
    print("lvef_pred, as_pred = model(processed_ct)") 