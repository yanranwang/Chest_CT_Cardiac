# Examples

This directory contains example code and documentation for cardiac function prediction training.

## 🎯 Main Files

### cardiac_training_example.py
**Core training script** - Cardiac function prediction training with hybrid data loader support

**Features**:
- Supports hybrid data loading (CSV labels + HDF5 images)
- Supports standard data loading (CSV + raw images)
- Multi-task learning (LVEF regression + AS classification)
- TensorBoard visualization
- Automatic model saving and restoration

**Usage**:
```bash
# Train using hybrid data loader
python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json

# Use command line arguments
python examples/cardiac_training_example.py \
    --config configs/hybrid_cardiac_training_config.json \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4
```

**Command Line Arguments**:
- `--config`: Configuration file path
- `--output_dir`: Output directory
- `--csv_path`: CSV data file path
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--device`: Training device (cuda/cpu/auto)
- `--use_fast_loader`: Enable fast data loader
- `--use_hybrid_loader`: Enable hybrid data loader

## 📚 Documentation

### QUICK_START.md
Quick start guide containing:
- Environment configuration
- Data preparation
- Training launch
- Results viewing

### README_CARDIAC_TRAINING.md
Detailed training documentation containing:
- Complete configuration instructions
- Data format requirements
- Advanced training options
- Troubleshooting guide

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   - Ensure you have `merged_ct_echo_data.csv` label file
   - Ensure you have corresponding HDF5 image files

3. **Start Training**:
   ```bash
   python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json
   ```

4. **Monitor Training**:
   ```bash
   tensorboard --logdir outputs/hybrid_cardiac_training/tensorboard
   ```

## 📊 Output Files

After training completion, the following files will be generated in the output directory:
- `best_model.pth`: Best model weights
- `training.log`: Detailed training logs
- `config.json`: Configuration used
- `tensorboard/`: TensorBoard log files

## 🔧 Custom Training

### Modify Configuration File
Edit `configs/hybrid_cardiac_training_config.json`:

```json
{
  "epochs": 100,
  "batch_size": 24,
  "learning_rate": 5e-05,
  "regression_weight": 0.5,
  "classification_weight": 0.5
}
```

### Use Command Line Override
```bash
python examples/cardiac_training_example.py \
    --config configs/hybrid_cardiac_training_config.json \
    --epochs 200 \
    --batch_size 32
```

## 📋 Notes

- Ensure sufficient GPU memory (16GB+ recommended)
- First run will automatically download pretrained weights
- Checkpoints are automatically saved during training
- Use `Ctrl+C` to safely stop training 