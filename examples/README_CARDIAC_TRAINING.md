# 心脏功能训练示例

这个目录包含了心脏功能回归训练的完整示例。

## 文件说明

### `cardiac_training_example.py`
完整的心脏功能训练示例，支持从CSV文件加载数据进行训练。

**功能特点：**
- 支持命令行参数配置
- 支持JSON配置文件
- 完整的训练流程
- 自动保存训练配置和结果
- 支持TensorBoard日志记录

### `cardiac_demo.py`
简化的心脏功能演示脚本，包含合成数据生成。

### `ct_cardiac_prediction_example.py`
心脏功能预测推理示例。

### `simple_cardiac_example.py`
简化的心脏功能训练示例。

## 使用方法

### 1. 基本训练

使用默认配置进行训练：

```bash
cd examples
python cardiac_training_example.py
```

### 2. 自定义参数训练

```bash
python cardiac_training_example.py \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --output_dir my_training_output \
    --csv_path my_data.csv
```

### 3. 使用配置文件

首先创建一个JSON配置文件 `my_config.json`：

```json
{
    "output_dir": "outputs/my_cardiac_training",
    "epochs": 100,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "csv_path": "my_cardiac_data.csv",
    "cardiac_metric_columns": [
        "ejection_fraction",
        "end_diastolic_volume"
    ]
}
```

然后使用配置文件训练：

```bash
python cardiac_training_example.py --config my_config.json
```

## 数据准备

### CSV文件格式

您的CSV文件至少需要包含以下列：
- `basename`: 图像文件的基础名称
- `folder`: 图像文件所在的文件夹

可选的心脏功能指标列：
- `ejection_fraction`: 射血分数
- `end_diastolic_volume`: 舒张末期容积
- `end_systolic_volume`: 收缩末期容积
- 其他心脏功能参数...

示例CSV格式：
```csv
basename,folder,patient_id,ejection_fraction,end_diastolic_volume
CT_001,001,P001,55.2,120.5
CT_002,001,P001,58.1,115.3
CT_003,002,P002,62.0,110.8
```

### 图像文件

确保您的图像文件按照配置的路径模板存储。默认模板为：
```
{base_path}/stanford_{folder}/{basename}.nii.gz
```

## 输出文件

训练完成后，在输出目录中会生成：
- `config.json`: 训练配置
- `best_model.pth`: 最佳模型权重
- `training.log`: 训练日志
- `data_info.json`: 数据统计信息
- `tensorboard/`: TensorBoard日志

## 配置参数说明

详细的配置参数说明请参考：`../merlin/training/README_CSV_Training.md`

## 故障排除

1. **找不到CSV文件**：检查文件路径是否正确
2. **找不到图像文件**：检查base_path和image_path_template配置
3. **内存不足**：减少batch_size或num_workers
4. **GPU不可用**：脚本会自动回退到CPU训练

## 相关文件

- 训练器核心代码：`../merlin/training/cardiac_trainer.py`
- 配置示例：`../merlin/training/cardiac_config_example.py`
- 详细文档：`../merlin/training/README_CSV_Training.md` 