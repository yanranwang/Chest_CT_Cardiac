# Cardiac Function Prediction - Hybrid Data Loader Training System

A cardiac function prediction model training system based on chest CT images and echocardiogram data.

## 🎯 Core Features

- **Hybrid Data Loading**: Read labels from CSV files and preprocessed image data from HDF5 files
- **Cardiac Function Prediction**: Simultaneous LVEF regression and aortic stenosis (AS) classification
- **Efficient Training**: Fast training using preprocessed HDF5 data

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Ensure the following files exist:
- **Label File**: `merged_ct_echo_data.csv` - Contains basename, folder, lvef, AS_maybe columns
- **Image Files**: Preprocessed image data in HDF5 format
- **Config File**: `configs/hybrid_cardiac_training_config.json`

### 3. Start Training

```bash
python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json
```

## 📊 Data Format

### CSV Label File Format
```csv
basename,folder,lvef,AS_maybe,patient_id
LA3dd33e5-LA3dd5b65,1A,61.47,0.0,patient_001
LA3dd74cb-LA3dd962e,1A,55.23,1.0,patient_002
```

### HDF5 Image File Format
- Path: `/path/to/preprocessed_data.h5`
- Structure: `images/` group contains image data with hash key names
- Metadata: `data_metadata.json` provides hash to basename/folder mapping

## ⚙️ Configuration

Key configuration parameters (`configs/hybrid_cardiac_training_config.json`):

```json
{
  "use_hybrid_loader": true,
  "csv_path": "/path/to/merged_ct_echo_data.csv",
  "hdf5_path": "/path/to/preprocessed_data.h5",
  "label_columns": ["lvef", "AS_maybe"],
  "epochs": 50,
  "batch_size": 24,
  "learning_rate": 5e-05
}
```

## 📁 Project Structure

```
├── configs/
│   ├── hybrid_cardiac_training_config.json  # Training configuration
│   └── README_hybrid_training.md            # Detailed usage guide
├── examples/
│   └── cardiac_training_example.py          # Main training script
├── merlin/
│   ├── training/
│   │   ├── fast_dataloader.py              # Hybrid data loader
│   │   └── cardiac_trainer.py              # Trainer
│   ├── models/                             # Model definitions
│   └── data/                               # Data processing tools
├── scripts/
│   └── merge_csv_data.py                   # CSV data merging tool
├── merged_ct_echo_data.csv                 # Merged label data
└── requirements.txt                        # Dependencies
```

## 🔧 Core Components

### HybridCardiacDataset
Hybrid data loader supporting:
- Reading label data from CSV
- Reading image data from HDF5
- Intelligent hash mapping matching
- Memory cache optimization

### CardiacTrainer
Trainer supporting:
- Multi-task learning (regression + classification)
- Class weight balancing
- TensorBoard visualization
- Automatic model saving

## 📈 Training Monitoring

### TensorBoard
```bash
tensorboard --logdir outputs/hybrid_cardiac_training/tensorboard
```

### Training Logs
- Location: `outputs/hybrid_cardiac_training/training.log`
- Contains: Loss curves, metric statistics, model saving information

## 🎯 Model Output

- **LVEF Regression**: Predict left ventricular ejection fraction (5-90%)
- **AS Classification**: Predict aortic stenosis risk (0: Normal, 1: Possible AS)

## 📋 System Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (GPU training)
- 16GB+ RAM (recommended)

## 🔍 Troubleshooting

### Common Issues

1. **Data Matching Failure**
   - Check basename/folder columns in CSV
   - Verify HDF5 file path and metadata file

2. **Out of Memory**
   - Reduce batch_size
   - Adjust cache_size
   - Set preload_data to false

3. **Slow Training**
   - Increase num_workers
   - Enable GPU training
   - Adjust cache settings

For detailed instructions, see: `configs/README_hybrid_training.md`

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve the project!
