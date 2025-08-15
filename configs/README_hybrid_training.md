# Hybrid Data Loader Usage Guide

## Overview

The hybrid data loader (`HybridCardiacDataset`) allows you to:
- Read label data from **CSV files**
- Read preprocessed image data from **HDF5 files**

This approach is particularly suitable for the following scenarios:
1. You already have preprocessed HDF5 image data
2. Label data is in CSV files and may be updated frequently
3. Avoid re-running time-consuming data preprocessing

## Configuration File

Use the `configs/hybrid_cardiac_training_config.json` configuration file with key settings:

```json
{
  "use_fast_loader": false,
  "use_hybrid_loader": true,
  "csv_path": "/path/to/your/labels.csv",
  "hdf5_path": "/path/to/your/preprocessed_data.h5",
  "label_columns": ["lvef", "AS_maybe"]
}
```

## Data Requirements

### CSV File Requirements
CSV files must contain the following columns:
- `basename`: File base name
- `folder`: Folder name  
- Label columns (e.g., `lvef`, `AS_maybe`)

Example:
```csv
basename,folder,lvef,AS_maybe,patient_id
LA3dd33e5-LA3dd5b65,1A,61.47,0.0,patient_001
LA3dd74cb-LA3dd962e,1A,55.23,1.0,patient_002
```

### HDF5 File Requirements
HDF5 files should contain an `images` group with:
- Key format: `{folder}_{basename}` (e.g., `1A_LA3dd33e5-LA3dd5b65`)
- Values: Preprocessed image data (numpy array)

## Usage Steps

### 1. Check Data Files
Ensure the following files exist and are properly formatted:
```bash
# Check CSV file
head /pasteur2/u/xhanwang/Chest_CT_Cardiac/filtered_echo_chestCT_data_filtered_chest_data.csv

# Check HDF5 file
ls -la /pasteur2/u/xhanwang/Chest_CT_Cardiac/outputs/cardiac_training_as_maybe/preprocessed_data.h5
```

### 2. Run Training
```bash
cd /pasteur2/u/xhanwang/Chest_CT_Cardiac
python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json
```

## Data Matching Logic

The hybrid data loader will:
1. Read all rows from the CSV file
2. For each row, construct `item_id = {folder}_{basename}`
3. Check if corresponding image data exists in the HDF5 file
4. Keep only data items that exist in both files
5. Display matching statistics

## Advantages

1. **Flexibility**: Can independently update labels without reprocessing images
2. **Efficiency**: Avoid redundant image preprocessing
3. **Compatibility**: Can use existing preprocessed data
4. **Debug-friendly**: Provides detailed data matching information

## Troubleshooting

### Common Issues

1. **"No matching data items found"**
   - Check the `basename` and `folder` columns in CSV
   - Confirm if the key format in HDF5 is `{folder}_{basename}`

2. **"Label columns missing"**
   - Confirm the CSV file contains the label columns specified in configuration
   - Check if column names match exactly (case-sensitive)

3. **"HDF5 file cannot be read"**
   - Check if the file path is correct
   - Confirm the file is not corrupted
   - Verify file permissions

### Debug Commands

```python
# Check HDF5 file structure in Python
import h5py
with h5py.File('/path/to/preprocessed_data.h5', 'r') as f:
    print("Groups:", list(f.keys()))
    if 'images' in f:
        print("Sample keys:", list(f['images'].keys())[:5])
```

## Configuration Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `use_hybrid_loader` | Enable hybrid data loader | `false` |
| `csv_path` | CSV label file path | Required |
| `hdf5_path` | HDF5 image file path | Required |
| `label_columns` | List of label column names | `["lvef", "AS_maybe"]` |
| `cache_size` | Image cache size | `200` |
| `enable_cache` | Whether to enable caching | `true` | 