#!/usr/bin/env python3
"""
CT image cardiac function prediction example

This example demonstrates how to:
1. Use nibabel to read nii.gz format CT images
2. Preprocess large images like 512x512x121
3. Use cardiac function prediction model for analysis
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
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
    Predict cardiac function from CT image
    
    Args:
        ct_file_path: CT image file path (.nii or .nii.gz)
        model_path: Pretrained model path (optional)
        window_center: Window center setting
        window_width: Window width setting
        device: Computing device
        
    Returns:
        prediction_results: Prediction results dictionary
    """
    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    print(f"Using device: {device}")
    print(f"CT file path: {ct_file_path}")
    
    # 1. Load and preprocess CT image
    print("\n=== Step 1: Load and preprocess CT image ===")
    processed_ct = load_and_preprocess_ct(
        ct_file_path,
        target_size=(16, 224, 224),  # Model expected input size
        window_center=window_center,
        window_width=window_width
    )
    
    # Move to specified device
    processed_ct = processed_ct.to(device)
    print(f"Preprocessing completed, tensor shape: {processed_ct.shape}")
    
    # 2. Load cardiac function prediction model
    print("\n=== Step 2: Load cardiac function prediction model ===")
    model = CardiacFunctionModel(pretrained_model_path=model_path)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # 3. Execute prediction
    print("\n=== Step 3: Execute cardiac function prediction ===")
    with torch.no_grad():
        lvef_pred, as_pred = model(processed_ct)
    
    # Convert to numpy array
    lvef_value = lvef_pred.squeeze().cpu().numpy()
    as_probability = as_pred.squeeze().cpu().numpy()
    as_prediction = (as_probability > 0.5)
    
    # 4. Organize prediction results
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
    Complete cardiac function metrics prediction (demo version)
    
    Args:
        ct_file_path: CT image file path
        model_path: Model path
        window_settings: Window center and width settings
        
    Returns:
        comprehensive_results: Complete cardiac function analysis results
    """
    print("=== Complete cardiac function analysis ===")
    
    # Use default window settings (chest CT)
    if window_settings is None:
        window_settings = {"center": 50, "width": 400}  # Mediastinal window
    
    # Load image processor
    processor = CTImageProcessor(target_size=(16, 224, 224))
    
    # Load CT image
    image_data, header = processor.load_nifti(ct_file_path)
    
    # Generate simulated complete cardiac function prediction
    print("\nGenerating simulated cardiac function prediction...")
    
    # Simulate prediction values
    np.random.seed(42)  # For reproducible results
    raw_predictions = np.random.randn(1, 10) * 0.5  # Small random values
    
    # Use CardiacMetricsCalculator to normalize predictions
    normalized_predictions = CardiacMetricsCalculator.normalize_predictions(
        torch.from_numpy(raw_predictions)
    ).numpy()
    
    # Evaluate prediction results
    evaluation = CardiacMetricsCalculator.evaluate_predictions(
        normalized_predictions[0], return_status=True
    )
    
    # Organize results
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
    Demonstrate the impact of different window settings on image processing
    """
    print("=== Different window settings demonstration ===")
    
    # Define different window settings
    window_configs = {
        "Lung Window": {"center": -600, "width": 1600},
        "Mediastinal Window": {"center": 50, "width": 400},
        "Bone Window": {"center": 300, "width": 1500},
        "Soft Tissue Window": {"center": 40, "width": 80}
    }
    
    processor = CTImageProcessor()
    image_data, _ = processor.load_nifti(ct_file_path)
    
    print(f"\nOriginal image statistics:")
    print(f"  Shape: {image_data.shape}")
    print(f"  Value range: [{image_data.min():.1f}, {image_data.max():.1f}] HU")
    print(f"  Mean: {image_data.mean():.1f} HU")
    print(f"  Std: {image_data.std():.1f} HU")
    
    for window_name, settings in window_configs.items():
        normalized = processor.normalize_intensity(
            image_data,
            window_center=settings["center"],
            window_width=settings["width"]
        )
        
        print(f"\n{window_name} (Center: {settings['center']}, Width: {settings['width']}):")
        print(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
        print(f"  Mean: {normalized.mean():.4f}")
        print(f"  Std: {normalized.std():.4f}")


def main():
    """Main function: Demonstrate complete CT image cardiac function prediction workflow"""
    
    print("CT Image Cardiac Function Prediction Demo")
    print("=" * 60)
    
    # Simulated CT file path (in actual use, please provide real .nii.gz file path)
    # ct_file_path = "path/to/your/ct_scan.nii.gz"
    
    # Since there's no real CT file, we create a simulated example
    print("Note: This demo uses simulated data. In actual use, please provide real CT file path")
    print("\nIf you have a real CT file, you can use it like this:")
    print("python ct_cardiac_prediction_example.py --ct_file your_scan.nii.gz")
    
    # Demonstrate image processing workflow
    print("\n=== Simulated CT Image Processing Demo ===")
    
    # Create simulated CT image
    print("Creating simulated CT image...")
    simulated_ct = np.random.randint(-1000, 3000, (512, 512, 121)).astype(np.float32)
    print(f"Simulated CT shape: {simulated_ct.shape}")
    print(f"Value range: [{simulated_ct.min():.0f}, {simulated_ct.max():.0f}] HU")
    
    # Create processor
    processor = CTImageProcessor(target_size=(16, 224, 224))
    
    # Preprocess image
    print("\nPreprocessing image...")
    processed_tensor = processor.preprocess_for_model(simulated_ct)
    print(f"Processed tensor shape: {processed_tensor.shape}")
    
    # Load model
    print("\nLoading cardiac function prediction model...")
    model = CardiacFunctionModel()
    model.eval()
    print("Model loaded successfully")
    
    # Execute prediction
    print("\nExecuting cardiac function prediction...")
    with torch.no_grad():
        lvef_pred, as_pred = model(processed_tensor)
    
    # Process results
    lvef_value = lvef_pred.squeeze().item()
    as_probability = as_pred.squeeze().item()
    as_prediction = (as_probability > 0.5)
    
    print(f"\nPrediction results:")
    print(f"  LVEF prediction: {lvef_value:.1f}%")
    print(f"  AS presence probability: {as_probability:.2f}")
    print(f"  AS prediction: {'Yes' if as_prediction else 'No'}")
    
    # Demonstrate different window settings
    print("\n=== Window Settings Demonstration ===")
    
    # Different window settings for chest CT
    window_settings = {
        "Lung Window": {"center": -600, "width": 1600},
        "Mediastinal Window": {"center": 50, "width": 400},
        "Bone Window": {"center": 300, "width": 1500}
    }
    
    for window_name, settings in window_settings.items():
        normalized = processor.normalize_intensity(
            simulated_ct,
            window_center=settings["center"],
            window_width=settings["width"]
        )
        print(f"{window_name}: Center={settings['center']}, Width={settings['width']}")
        print(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Complete cardiac function evaluation
    print("\n=== Complete Cardiac Function Evaluation ===")
    
    # Generate simulated complete cardiac function data
    np.random.seed(42)
    raw_predictions = np.random.randn(10) * 0.5
    
    # Use CardiacMetricsCalculator for evaluation
    calculator = CardiacMetricsCalculator()
    evaluation = calculator.evaluate_predictions(raw_predictions, return_status=True)
    
    print("Cardiac function evaluation results:")
    
    for metric_name, metric_info in evaluation.items():
        print(f"  {metric_info['description']}: {metric_info['value']:.2f} ({metric_info['status']})")
    
    print("\n=== Usage Instructions ===")
    print("In actual use, please follow these steps:")
    print("1. Prepare CT image file (.nii or .nii.gz format)")
    print("2. Call load_and_preprocess_ct() function to load and preprocess image")
    print("3. Use CardiacFunctionModel for prediction")
    print("4. Use CardiacMetricsCalculator to interpret and evaluate results")
    
    print("\nExample code:")
    print("""
# Load and preprocess CT image
from merlin.utils.image_utils import load_and_preprocess_ct
processed_ct = load_and_preprocess_ct('your_ct_scan.nii.gz')

# Cardiac function prediction
from merlin.models.cardiac_regression import CardiacFunctionModel
model = CardiacFunctionModel()
lvef_pred, as_pred = model(processed_ct)

# Result interpretation
print(f"LVEF prediction: {lvef_pred.squeeze().item():.1f}%")
print(f"AS presence probability: {as_pred.squeeze().item():.2f}")
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CT Image Cardiac Function Prediction')
    parser.add_argument('--ct_file', type=str, help='CT Image File Path (.nii or .nii.gz)')
    parser.add_argument('--model_path', type=str, help='Pretrained Model Path')
    parser.add_argument('--window_center', type=float, help='Window Center Setting')
    parser.add_argument('--window_width', type=float, help='Window Width Setting')
    parser.add_argument('--device', type=str, default='auto', help='Computing Device')
    
    args = parser.parse_args()
    
    if args.ct_file and os.path.exists(args.ct_file):
        # If a real CT file is provided
        print(f"Processing CT file: {args.ct_file}")
        
        # Execute prediction
        results = predict_cardiac_function_from_ct(
            args.ct_file,
            model_path=args.model_path,
            window_center=args.window_center,
            window_width=args.window_width,
            device=args.device
        )
        
        print("\n=== Prediction Results ===")
        print(f"LVEF: {results['lvef']:.2f}%")
        print(f"AS probability: {results['as_probability']:.3f}")
        print(f"AS prediction: {'Yes' if results['as_prediction'] else 'No'}")
        
        # Demonstrate different window settings
        demonstrate_different_window_settings(args.ct_file)
        
        # Complete cardiac function analysis
        comprehensive_results = predict_full_cardiac_metrics_from_ct(args.ct_file)
        print("\n=== Complete Cardiac Function Analysis ===")
        for metric_name, metric_info in comprehensive_results['cardiac_metrics'].items():
            print(f"{metric_info['description']}: {metric_info['value']:.2f} ({metric_info['status']})")
    
    else:
        # Run demo
        main() 