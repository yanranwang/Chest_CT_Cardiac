# 心脏功能回归训练 - CSV数据支持

这个训练脚本已经修改为支持从CSV文件读取数据进行心脏功能回归训练。

## 主要改进

### 1. 灵活的CSV数据读取
- 支持从CSV文件读取训练数据
- 可配置的列名映射
- 数据验证和清理功能
- 支持真实心脏功能指标或模拟数据

### 2. 多种数据分割方式
- **随机分割** (`random`): 随机分配样本到训练/验证集
- **顺序分割** (`sequential`): 按顺序分配样本
- **基于患者的分割** (`patient_based`): 确保同一患者的数据不会同时出现在训练和验证集中

### 3. 数据质量控制
- 自动移除缺失数据
- 重复数据检测和移除
- 文件存在性检查（可选）
- 数据类型验证

### 4. 灵活的文件路径配置
- 可配置的图像文件路径模板
- 支持不同的目录结构

## 使用方法

### 1. 准备CSV文件

CSV文件至少需要包含以下列：
- `basename`: 图像文件的基础名称
- `folder`: 图像文件所在的文件夹

可选列：
- `patient_id`: 患者ID（用于基于患者的数据分割）
- 心脏功能指标列（如 `ejection_fraction`, `end_diastolic_volume` 等）
- 其他元数据列

示例CSV结构：
```csv
basename,folder,patient_id,ejection_fraction,end_diastolic_volume
CT_001,001,P001,55.2,120.5
CT_002,001,P001,58.1,115.3
CT_003,002,P002,62.0,110.8
```

### 2. 配置训练参数

使用 `cardiac_config_example.py` 中的配置模板：

```python
from cardiac_config_example import get_training_config, get_demo_config

# 基础配置
config = get_training_config()

# 修改CSV相关配置
config.update({
    'csv_path': 'your_data.csv',
    'cardiac_metric_columns': [
        'ejection_fraction',
        'end_diastolic_volume',
        'end_systolic_volume'
    ],
    'split_method': 'patient_based',
    'check_file_exists': True
})
```

### 3. 运行训练

#### 方法1: 直接运行训练脚本
```bash
python ../examples/cardiac_training_example.py
```

#### 方法2: 使用配置文件
```bash
python cardiac_config_example.py
```

#### 方法3: 在代码中调用
```python
from cardiac_trainer import create_data_loaders, CardiacTrainer
from cardiac_config_example import get_production_config

config = get_production_config()
train_loader, val_loader = create_data_loaders(config)
trainer = CardiacTrainer(config)
trainer.train(train_loader, val_loader)
```

## 配置参数详解

### CSV数据配置
```python
{
    'csv_path': 'path/to/your/data.csv',           # CSV文件路径
    'required_columns': ['basename', 'folder'],    # 必需的列
    'cardiac_metric_columns': [                    # 心脏功能指标列
        'ejection_fraction',
        'end_diastolic_volume'
    ],
    'metadata_columns': ['patient_id', 'age'],     # 额外的元数据列
}
```

### 文件路径配置
```python
{
    'base_path': '/path/to/images',                           # 图像文件根目录
    'image_path_template': '{base_path}/stanford_{folder}/{basename}.nii.gz',  # 文件路径模板
    'check_file_exists': True,                                # 是否检查文件存在
}
```

### 数据分割配置
```python
{
    'train_val_split': 0.8,              # 训练/验证分割比例
    'split_method': 'patient_based',     # 分割方式
    'seed': 42,                          # 随机种子
}
```

### 数据清理配置
```python
{
    'remove_missing_files': True,        # 移除缺失数据
    'remove_duplicates': True,           # 移除重复数据
}
```

## 输出文件

训练过程会在输出目录中生成以下文件：

- `data_info.json`: 数据统计信息
- `config.json`: 训练配置
- `training.log`: 训练日志
- `checkpoint_best.pth`: 最佳模型检查点
- `checkpoint_latest.pth`: 最新检查点
- `best_model.pth`: 最佳模型权重
- `tensorboard/`: TensorBoard日志

## 数据信息示例

`data_info.json` 包含以下信息：
```json
{
  "total_samples": 1000,
  "train_samples": 800,
  "val_samples": 200,
  "cardiac_metric_columns": ["ejection_fraction", "end_diastolic_volume"],
  "split_method": "patient_based",
  "split_ratio": 0.8
}
```

## 故障排除

### 常见问题

1. **CSV文件读取失败**
   - 检查文件路径是否正确
   - 确认文件编码为UTF-8
   - 验证必需的列是否存在

2. **图像文件找不到**
   - 检查 `base_path` 和 `image_path_template` 配置
   - 设置 `check_file_exists: true` 来验证文件
   - 检查文件权限

3. **心脏功能指标数据错误**
   - 确认列名是否正确
   - 检查数据类型（应为数值型）
   - 处理缺失值

4. **内存不足**
   - 减少 `batch_size`
   - 减少 `num_workers`
   - 使用 `drop_last: true`

### 调试建议

1. 先用小数据集测试（`get_demo_config()`）
2. 启用文件存在性检查（`check_file_exists: true`）
3. 查看训练日志文件
4. 检查 `data_info.json` 中的数据统计

## 进阶功能

### 自定义数据预处理
可以继承 `CardiacDataset` 类来实现自定义的数据预处理：

```python
class CustomCardiacDataset(CardiacDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # 添加自定义预处理
        return sample
```

### 自定义数据分割
可以实现自定义的数据分割逻辑：

```python
def custom_split_data(data_list, config):
    # 实现自定义分割逻辑
    return train_data, val_data
```

### 多GPU训练
脚本自动支持多GPU训练：
```python
config['device'] = 'cuda'  # 自动使用所有可用GPU
``` 