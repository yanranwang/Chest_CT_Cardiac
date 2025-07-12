#!/usr/bin/env python3
"""
Cardiac function prediction demo script

This script demonstrates how to use Merlin for cardiac function regression training and prediction:
1. Load Merlin pretrained model
2. Perform cardiac function regression training
3. Execute cardiac function prediction
4. Generate prediction reports

Usage:
    python cardiac_demo.py --mode train    # Training mode
    python cardiac_demo.py --mode inference --model_path outputs/cardiac_training/best_model.pth
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from merlin.models.cardiac_regression import CardiacFunctionModel, CardiacMetricsCalculator
from merlin.training.cardiac_trainer import CardiacTrainer, create_data_loaders
from merlin.inference.cardiac_inference import CardiacInference
from merlin.data import download_sample_data, DataLoader


def create_demo_config():
    """Create demo training configuration"""
    config = {
        'output_dir': 'outputs/cardiac_training',
        'pretrained_model_path': None,  # Will auto-download Merlin pretrained weights
        'num_cardiac_metrics': 10,
        'epochs': 20,  # Fewer epochs for demo
        'batch_size': 2,  # Smaller batch size for demo
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'loss_function': 'mse',
        'scheduler': {
            'type': 'cosine',
            'eta_min': 1e-6
        },
        'freeze_encoder': True,  # Freeze pretrained encoder
        'grad_clip': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'num_workers': 2,
        'log_interval': 5,
        'save_interval': 5,
        'use_tensorboard': True,
        'train_val_split': 0.8,
        'num_samples': 50  # Fewer samples for demo
    }
    return config


def generate_synthetic_cardiac_data(num_samples=50):
    """Generate synthetic cardiac function data for demo"""
    print("Generating synthetic cardiac function data...")
    
    data_list = []
    
    # Simulate different types of cardiac conditions
    conditions = ['normal', 'mild_dysfunction', 'moderate_dysfunction', 'severe_dysfunction']
    
    for i in range(num_samples):
        condition = conditions[i % len(conditions)]
        
        # Generate corresponding functional indicators based on cardiac condition
        if condition == 'normal':
            # Normal cardiac function
            ef = np.random.normal(60, 5)  # Ejection fraction
            sv = np.random.normal(70, 10)  # Stroke volume
            co = np.random.normal(5.0, 0.5)  # Cardiac output
            hrv = np.random.normal(30, 5)  # Heart rate variability
            lvm = np.random.normal(150, 20)  # Left ventricular mass
            wt = np.random.normal(10, 1)  # Wall thickness
            cv = np.random.normal(120, 15)  # Cardiac volume
            ci = np.random.normal(1.0, 0.1)  # Contractility index
            df = np.random.normal(1.0, 0.1)  # Diastolic function
            vf = np.random.normal(1.0, 0.1)  # Valve function
            
        elif condition == 'mild_dysfunction':
            # Mild dysfunction
            ef = np.random.normal(50, 5)
            sv = np.random.normal(60, 10)
            co = np.random.normal(4.0, 0.5)
            hrv = np.random.normal(25, 5)
            lvm = np.random.normal(170, 20)
            wt = np.random.normal(12, 1)
            cv = np.random.normal(140, 15)
            ci = np.random.normal(0.8, 0.1)
            df = np.random.normal(0.8, 0.1)
            vf = np.random.normal(0.9, 0.1)
            
        elif condition == 'moderate_dysfunction':
            # Moderate dysfunction
            ef = np.random.normal(40, 5)
            sv = np.random.normal(50, 10)
            co = np.random.normal(3.5, 0.5)
            hrv = np.random.normal(20, 5)
            lvm = np.random.normal(200, 20)
            wt = np.random.normal(15, 1)
            cv = np.random.normal(160, 15)
            ci = np.random.normal(0.6, 0.1)
            df = np.random.normal(0.6, 0.1)
            vf = np.random.normal(0.7, 0.1)
            
        else:  # severe_dysfunction
            # Severe dysfunction
            ef = np.random.normal(30, 5)
            sv = np.random.normal(40, 10)
            co = np.random.normal(2.5, 0.5)
            hrv = np.random.normal(15, 5)
            lvm = np.random.normal(250, 20)
            wt = np.random.normal(18, 1)
            cv = np.random.normal(180, 15)
            ci = np.random.normal(0.4, 0.1)
            df = np.random.normal(0.4, 0.1)
            vf = np.random.normal(0.5, 0.1)
        
        # Convert to normalized values (model input format)
        cardiac_metrics = np.array([ef, sv, co, hrv, lvm, wt, cv, ci, df, vf])
        
        # Normalize to [-1, 1] range
        cardiac_metrics_normalized = (cardiac_metrics - np.array([60, 70, 5, 30, 150, 10, 120, 1, 1, 1])) / \
                                    np.array([10, 15, 1, 10, 30, 2, 25, 0.2, 0.3, 0.2])
        
        data_list.append({
            'image': f'synthetic_ct_{i:03d}.nii.gz',  # Simulated CT image filename
            'cardiac_metrics': cardiac_metrics_normalized.astype(np.float32),
            'patient_id': f'DEMO_{i:03d}',
            'condition': condition,
            'raw_metrics': cardiac_metrics
        })
    
    print(f"Generated {len(data_list)} synthetic data samples")
    return data_list


def train_cardiac_model():
    """Train cardiac function prediction model"""
    print("Starting cardiac function prediction model training...")
    
    # Create configuration
    config = create_demo_config()
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_cardiac_data(config['num_samples'])
    
    # Create data loaders
    print("Creating data loaders...")
    
    # Split training and validation data
    split_idx = int(len(synthetic_data) * config['train_val_split'])
    train_data = synthetic_data[:split_idx]
    val_data = synthetic_data[split_idx:]
    
    # Use Merlin's data loader structure
    from merlin.training.cardiac_trainer import CardiacDataset
    from torch.utils.data import DataLoader
    
    train_dataset = CardiacDataset(train_data)
    val_dataset = CardiacDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Create trainer
    trainer = CardiacTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    print("Training completed!")
    print(f"Best model saved at: {config['output_dir']}/best_model.pth")
    
    return config['output_dir']


def test_cardiac_inference(model_path):
    """Test cardiac function inference"""
    print(f"Running cardiac function inference with model: {model_path}")
    
    # Create inference system
    inference = CardiacInference(model_path)
    
    # Generate test data
    test_data = generate_synthetic_cardiac_data(10)
    
    # Run inference
    results = []
    for i, data in enumerate(test_data):
        # Create simulated CT image tensor
        # In practice, this would be loaded from actual nii.gz files
        simulated_ct = torch.randn(1, 1, 16, 224, 224)
        
        # Run prediction
        prediction = inference.predict(simulated_ct)
        
        # Create result record
        result = {
            'patient_id': data['patient_id'],
            'condition': data['condition'],
            'predicted_metrics': prediction,
            'actual_metrics': data['raw_metrics']
        }
        results.append(result)
        
        print(f"Patient {data['patient_id']}: "
              f"Condition={data['condition']}, "
              f"Predicted EF={prediction[0]:.1f}%, "
              f"Actual EF={data['raw_metrics'][0]:.1f}%")
    
    print(f"\nInference completed for {len(results)} patients")
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Cardiac function prediction demo')
    parser.add_argument('--mode', choices=['train', 'inference'], default='train',
                        help='Demo mode: train or inference')
    parser.add_argument('--model_path', type=str,
                        help='Path to trained model (required for inference mode)')
    parser.add_argument('--output_dir', type=str, default='outputs/cardiac_training',
                        help='Training output directory')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs for demo')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for demo')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of synthetic samples for demo')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("=" * 60)
        print("🚀 Starting cardiac function model training demo")
        print("=" * 60)
        
        # Train model
        output_dir = train_cardiac_model()
        
        print("=" * 60)
        print("✅ Training demo completed!")
        print(f"📁 Results saved in: {output_dir}")
        print("📊 To view training progress:")
        print(f"   tensorboard --logdir {output_dir}/tensorboard")
        print("🔬 To run inference:")
        print(f"   python cardiac_demo.py --mode inference --model_path {output_dir}/best_model.pth")
        print("=" * 60)
        
    elif args.mode == 'inference':
        if not args.model_path:
            print("❌ Model path is required for inference mode")
            print("Please provide --model_path argument")
            return
        
        if not os.path.exists(args.model_path):
            print(f"❌ Model file not found: {args.model_path}")
            print("Please train a model first using --mode train")
            return
        
        print("=" * 60)
        print("🔬 Starting cardiac function inference demo")
        print("=" * 60)
        
        # Test inference
        results = test_cardiac_inference(args.model_path)
        
        print("=" * 60)
        print("✅ Inference demo completed!")
        print(f"📊 Processed {len(results)} test cases")
        print("=" * 60)
    
    else:
        print("❌ Invalid mode. Please choose 'train' or 'inference'")


if __name__ == '__main__':
    main() 