#!/usr/bin/env python3
"""
Cardiac function regression training example

This script demonstrates how to use Merlin for complete cardiac function regression training:
1. Load cardiac function data from CSV files
2. Configure training parameters  
3. Create data loaders
4. Train cardiac function prediction model
5. Save training results and model

Usage:
    python cardiac_training_example.py
    
With custom configuration:
    python cardiac_training_example.py --config my_config.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from merlin.training.cardiac_trainer import CardiacTrainer, create_data_loaders


def create_default_config():
    """Create default training configuration"""
    config = {
        'output_dir': 'outputs/cardiac_training',
        'pretrained_model_path': '/dataNAS/people/joycewyr/Merlin/merlin/models/checkpoints/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt',  # Use Merlin pretrained model weights
        'num_cardiac_metrics': 2,  # LVEF regression + AS classification
        'epochs': 100,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'loss_function': 'mse',
        'scheduler': {
            'type': 'cosine',
            'eta_min': 1e-6
        },
        'freeze_encoder': True,  # Important: freeze pretrained encoder for fine-tuning
        'grad_clip': 1.0,
        'device': 'cuda',
        'seed': 42,
        'num_workers': 0,
        'log_interval': 5,  # More frequent logging (every 5 batches)
        'save_interval': 5,  # More frequent model saving (every 5 epochs)
        'use_tensorboard': True,
        'drop_last': True,  # Must be True for training to avoid BatchNorm errors
        
        # Progress monitoring enhancement
        'progress_bar': True,  # Enable progress bar
        'show_gpu_memory': True,  # Show GPU memory usage
        'show_eta': True,  # Show estimated time remaining
        'detailed_metrics': True,  # Show detailed evaluation metrics
        
        # Data splitting configuration
        'train_val_split': 0.8,
        'split_method': 'random',  # 'random', 'sequential', 'patient_based'
        
        # CSV data configuration
        'csv_path': '/dataNAS/people/joycewyr/Merlin/filtered_echo_chestCT_data_filtered_chest_data.csv',
        'required_columns': ['basename', 'folder'],
        'cardiac_metric_columns': [],  # Set to list of cardiac function metric column names in CSV
        'metadata_columns': ['patient_id'],  # Additional metadata columns to save
        
        # File path configuration
        'base_path': '/dataNAS/data/ct_data/ct_scans',
        'image_path_template': '{base_path}/stanford_{folder}/{basename}.nii.gz',
        'check_file_exists': False,  # Set to True to check if files exist
        
        # Data cleaning configuration
        'remove_missing_files': True,
        'remove_duplicates': True,
        
        # Fast data loader configuration
        'use_fast_loader': False,  # Set to True to enable fast data loader
        'preprocessed_data_dir': 'outputs/preprocessed_data',  # Preprocessed data directory
        'preprocess_batch_size': 16,  # Preprocessing batch size
        'cache_config': {
            'enable_cache': True,      # Enable memory cache
            'cache_size': 1000,        # Cache size
            'preload_train_data': False,  # Whether to preload training data to memory
            'preload_val_data': False,    # Whether to preload validation data to memory
        }
    }
    return config


def load_config_from_file(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def save_config(config, output_path):
    """Save configuration to JSON file"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def print_training_info(config):
    """Print training configuration information"""
    print("=" * 80)
    print("🔧 Training Configuration")
    print("=" * 80)
    print(f"📁 Output directory: {config['output_dir']}")
    print(f"📊 CSV file: {config['csv_path']}")
    print(f"🏥 Data path: {config['base_path']}")
    print(f"🎯 Epochs: {config['epochs']}")
    print(f"📦 Batch size: {config['batch_size']}")
    print(f"🎓 Learning rate: {config['learning_rate']}")
    print(f"🔧 Optimizer: {config['optimizer']}")
    print(f"💾 Log interval: {config['log_interval']} batches")
    print(f"💾 Save interval: {config['save_interval']} epochs")
    print(f"🖥️  Device: {config['device']}")
    print(f"📈 TensorBoard: {'✅' if config['use_tensorboard'] else '❌'}")
    print(f"🎯 Data split: {config['train_val_split']:.1%} train / {1-config['train_val_split']:.1%} val")
    print("=" * 80)


def check_dependencies():
    """Check required dependencies for training"""
    print("🔍 Checking training dependencies...")
    
    missing_deps = []
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available, {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  CUDA not available, will use CPU training")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import tqdm
        print(f"✅ tqdm (progress bars)")
    except ImportError:
        missing_deps.append("tqdm")
        print("❌ tqdm not installed, progress bars will be unavailable")
    
    try:
        import tensorboard
        print(f"✅ TensorBoard")
    except ImportError:
        missing_deps.append("tensorboard")
        print("❌ TensorBoard not installed, training visualization will be unavailable")
    
    try:
        import monai
        print(f"✅ MONAI")
    except ImportError:
        missing_deps.append("monai")
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Please run the following command to install:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    print("✅ All dependencies installed")
    return True


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Cardiac function regression training example')
    parser.add_argument('--config', type=str, help='Configuration file path (JSON format)')
    parser.add_argument('--output_dir', type=str, help='Output directory path')
    parser.add_argument('--csv_path', type=str, help='CSV data file path')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--log_interval', type=int, help='Logging interval')
    parser.add_argument('--save_interval', type=int, help='Model saving interval')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], help='Training device')
    parser.add_argument('--use_pretrained', type=bool, default=True, help='Whether to use pretrained weights')
    parser.add_argument('--use_fast_loader', action='store_true', help='Use fast data loader')
    parser.add_argument('--preprocessed_data_dir', type=str, help='Preprocessed data directory')
    parser.add_argument('--preprocess_batch_size', type=int, help='Preprocessing batch size')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed, please install missing packages first")
        return
    
    # Load configuration
    if args.config:
        print(f"📋 Loading configuration from file: {args.config}")
        config = load_config_from_file(args.config)
    else:
        print("📋 Using default configuration")
        config = create_default_config()
    
    # Validate and handle Merlin pretrained weights
    if args.use_pretrained and config.get('pretrained_model_path'):
        print("\n🔍 Checking Merlin pretrained weights...")
        pretrained_path = config['pretrained_model_path']
        
        # If weights file doesn't exist, try to download automatically
        if not os.path.exists(pretrained_path):
            print(f"❌ Pretrained weights file not found: {pretrained_path}")
            print("🔄 Attempting to auto-download Merlin pretrained weights...")
            
            try:
                # Use Merlin's built-in weight download functionality
                from merlin import Merlin
                merlin_model = Merlin()  # This will automatically download weights
                
                # Get actual weights path
                actual_checkpoint_path = os.path.join(
                    merlin_model.current_path, 
                    'checkpoints', 
                    merlin_model.checkpoint_name
                )
                
                if os.path.exists(actual_checkpoint_path):
                    config['pretrained_model_path'] = actual_checkpoint_path
                    print(f"✅ Successfully downloaded and set pretrained weights: {actual_checkpoint_path}")
                else:
                    print("❌ Auto-download failed, will use randomly initialized weights")
                    config['pretrained_model_path'] = None
                    
            except Exception as e:
                print(f"❌ Auto-download weights failed: {e}")
                print("Will continue training with randomly initialized weights")
                config['pretrained_model_path'] = None
        else:
            print(f"✅ Found pretrained weights file: {pretrained_path}")
    
    # Check if using fast data loader
    use_fast_loader = config.get('use_fast_loader', False)
    if use_fast_loader:
        print("\n🚀 Using fast data loader mode")
        preprocessed_data_dir = config.get('preprocessed_data_dir')
        if not preprocessed_data_dir:
            print("❌ Fast data loader requires preprocessed_data_dir setting")
            print("Please run data preprocessing script first:")
            print("  python -m merlin.training.data_preprocessor --config config.json")
            return
        
        # Check preprocessed data files
        hdf5_path = Path(preprocessed_data_dir) / 'preprocessed_data.h5'
        metadata_path = Path(preprocessed_data_dir) / 'data_metadata.json'
        
        if not hdf5_path.exists():
            print(f"❌ Preprocessed data file not found: {hdf5_path}")
            print("Please run data preprocessing script first:")
            print("  python -m merlin.training.data_preprocessor --config config.json")
            return
        
        if not metadata_path.exists():
            print(f"❌ Metadata file not found: {metadata_path}")
            print("Please run data preprocessing script first:")
            print("  python -m merlin.training.data_preprocessor --config config.json")
            return
        
        print(f"✅ Found preprocessed data: {hdf5_path}")
        print(f"✅ Found metadata file: {metadata_path}")
    else:
        print("\n📁 Using standard data loader mode")
    
    # Command line arguments override configuration
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.csv_path:
        config['csv_path'] = args.csv_path
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.log_interval:
        config['log_interval'] = args.log_interval
    if args.save_interval:
        config['save_interval'] = args.save_interval
    if args.device:
        config['device'] = args.device
    if args.use_fast_loader:
        config['use_fast_loader'] = True
    if args.preprocessed_data_dir:
        config['preprocessed_data_dir'] = args.preprocessed_data_dir
    if args.preprocess_batch_size:
        config['preprocess_batch_size'] = args.preprocess_batch_size
    
    # Print configuration
    print_training_info(config)
    
    # Save configuration
    output_dir = Path(config['output_dir'])
    config_save_path = output_dir / 'config.json'
    save_config(config, config_save_path)
    print(f"📁 Configuration saved to: {config_save_path}")
    
    try:
        # Create data loaders
        print("\n📂 Creating data loaders...")
        use_fast_loader = config.get('use_fast_loader', False)
        
        if use_fast_loader:
            # Use fast data loader
            from merlin.training.fast_dataloader import create_fast_data_loaders
            train_loader, val_loader = create_fast_data_loaders(config)
            print(f"✅ Using fast data loader - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
        else:
            # Use standard data loader
            train_loader, val_loader = create_data_loaders(config)
            print(f"✅ Using standard data loader - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
        
        # Create trainer
        print("\n🤖 Initializing trainer...")
        trainer = CardiacTrainer(config)
        
        # Start training
        print("\n🚀 Starting training...")
        trainer.train(train_loader, val_loader)
        
        # Post-training information
        print("\n🎉 Training completed!")
        print("=" * 80)
        print("📁 Output files:")
        print(f"   🏆 Best model: {config['output_dir']}/best_model.pth")
        print(f"   📊 Training log: {config['output_dir']}/training.log")
        print(f"   ⚙️  Configuration: {config['output_dir']}/config.json")
        print(f"   📈 TensorBoard: {config['output_dir']}/tensorboard")
        print("\n💡 Next steps:")
        print("   1. Review training logs for detailed information")
        print("   2. Visualize training process with TensorBoard:")
        print(f"      tensorboard --logdir {config['output_dir']}/tensorboard")
        print("   3. Use trained model for prediction")
        print("   4. Training acceleration tips:")
        print("      - Preprocess data to accelerate future training:")
        print("        python -m merlin.training.data_preprocessor --config config.json")
        print("      - Use fast data loader:")
        print("        python cardiac_training_example.py --use_fast_loader --preprocessed_data_dir outputs/preprocessed_data")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Saved checkpoints can be used to resume training")
    except Exception as e:
        print(f"\n❌ Error occurred during training: {e}")
        print("\n🔍 Troubleshooting suggestions:")
        print("   1. Check if CSV file path is correct")
        print("   2. Check if data directory path exists")
        print("   3. Check if GPU memory is sufficient (try reducing batch_size)")
        print("   4. Check if disk space is sufficient")
        print("   5. See full error information:")
        raise


if __name__ == '__main__':
    main() 