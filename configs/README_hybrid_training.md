# 混合数据加载器使用指南

## 概述

混合数据加载器 (`HybridCardiacDataset`) 允许您：
- 从 **CSV 文件** 读取标签数据
- 从 **HDF5 文件** 读取预处理的图像数据

这种方式特别适用于以下场景：
1. 您已经有预处理好的 HDF5 图像数据
2. 标签数据在 CSV 文件中，且可能经常更新
3. 避免重新运行耗时的数据预处理过程

## 配置文件

使用 `configs/hybrid_cardiac_training_config.json` 配置文件，关键设置：

```json
{
  "use_fast_loader": false,
  "use_hybrid_loader": true,
  "csv_path": "/path/to/your/labels.csv",
  "hdf5_path": "/path/to/your/preprocessed_data.h5",
  "label_columns": ["lvef", "AS_maybe"]
}
```

## 数据要求

### CSV 文件要求
CSV 文件必须包含以下列：
- `basename`: 文件基础名称
- `folder`: 文件夹名称  
- 标签列 (如 `lvef`, `AS_maybe`)

示例：
```csv
basename,folder,lvef,AS_maybe,patient_id
LA3dd33e5-LA3dd5b65,1A,61.47,0.0,patient_001
LA3dd74cb-LA3dd962e,1A,55.23,1.0,patient_002
```

### HDF5 文件要求
HDF5 文件应包含 `images` 组，其中：
- 键名格式：`{folder}_{basename}` (例如: `1A_LA3dd33e5-LA3dd5b65`)
- 值：预处理的图像数据 (numpy array)

## 使用步骤

### 1. 检查数据文件
确保以下文件存在且格式正确：
```bash
# 检查 CSV 文件
head /pasteur2/u/xhanwang/Chest_CT_Cardiac/filtered_echo_chestCT_data_filtered_chest_data.csv

# 检查 HDF5 文件
ls -la /pasteur2/u/xhanwang/Chest_CT_Cardiac/outputs/cardiac_training_as_maybe/preprocessed_data.h5
```

### 2. 运行训练
```bash
cd /pasteur2/u/xhanwang/Chest_CT_Cardiac
python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json
```

## 数据匹配逻辑

混合数据加载器会：
1. 读取 CSV 文件中的所有行
2. 对每一行，构建 `item_id = {folder}_{basename}`
3. 检查 HDF5 文件中是否存在对应的图像数据
4. 只保留在两个文件中都存在的数据项
5. 显示匹配统计信息

## 优势

1. **灵活性**: 可以独立更新标签而不需要重新处理图像
2. **效率**: 避免重复的图像预处理
3. **兼容性**: 可以使用现有的预处理数据
4. **调试友好**: 提供详细的数据匹配信息

## 故障排除

### 常见问题

1. **"没有找到匹配的数据项"**
   - 检查 CSV 中的 `basename` 和 `folder` 列
   - 确认 HDF5 中的键名格式是否为 `{folder}_{basename}`

2. **"标签列缺失"**
   - 确认 CSV 文件包含配置中指定的标签列
   - 检查列名是否完全匹配（区分大小写）

3. **"HDF5 文件无法读取"**
   - 检查文件路径是否正确
   - 确认文件没有损坏
   - 验证文件权限

### 调试命令

```python
# 在 Python 中检查 HDF5 文件结构
import h5py
with h5py.File('/path/to/preprocessed_data.h5', 'r') as f:
    print("Groups:", list(f.keys()))
    if 'images' in f:
        print("Sample keys:", list(f['images'].keys())[:5])
```

## 配置参数说明

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `use_hybrid_loader` | 启用混合数据加载器 | `false` |
| `csv_path` | CSV 标签文件路径 | 必需 |
| `hdf5_path` | HDF5 图像文件路径 | 必需 |
| `label_columns` | 标签列名列表 | `["lvef", "AS_maybe"]` |
| `cache_size` | 图像缓存大小 | `200` |
| `enable_cache` | 是否启用缓存 | `true` | 