# Scripts

This directory contains utility scripts for data processing and system maintenance.

## 📁 Script Descriptions

### merge_csv_data.py
**Function**: Merge CSV files of CT and Echo data

**Purpose**: 
- Merge CT data containing basename/folder with Echo data containing labels
- Generate complete label files for training

**Usage**:
```bash
python scripts/merge_csv_data.py
```

**Input Files**:
- `filtered_echo_chestCT_data_filtered_chest_data.csv` - CT data
- `filtered_echo_chestCT_data_filtered_echo_data.csv` - Echo data

**Output Files**:
- `merged_ct_echo_data.csv` - Merged complete data

### test_hybrid_loader.py
**Function**: Test hybrid data loader

**Purpose**:
- Verify if HybridCardiacDataset works properly
- Check data matching status
- Test sample loading

**Usage**:
```bash
python scripts/test_hybrid_loader.py
```

## 🔧 Use Cases

1. **Initial Setup**: Run `merge_csv_data.py` to create training data
2. **Debugging**: Use `test_hybrid_loader.py` to verify data loading
3. **Maintenance**: Re-run merge script when data sources are updated

## 📋 Notes

- Ensure input file paths are correct
- Check file permissions
- Backup important data files before running 