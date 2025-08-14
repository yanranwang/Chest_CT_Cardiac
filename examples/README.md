# Examples 示例代码

这个目录包含心脏功能预测训练的示例代码和文档。

## 🎯 主要文件

### cardiac_training_example.py
**核心训练脚本** - 支持混合数据加载器的心脏功能预测训练

**功能特性**:
- 支持混合数据加载 (CSV标签 + HDF5图像)
- 支持标准数据加载 (CSV + 原始图像)
- 多任务学习 (LVEF回归 + AS分类)
- TensorBoard可视化
- 自动模型保存和恢复

**使用方法**:
```bash
# 使用混合数据加载器训练
python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json

# 使用命令行参数
python examples/cardiac_training_example.py \
    --config configs/hybrid_cardiac_training_config.json \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4
```

**命令行参数**:
- `--config`: 配置文件路径
- `--output_dir`: 输出目录
- `--csv_path`: CSV数据文件路径
- `--epochs`: 训练轮数
- `--batch_size`: 批量大小
- `--learning_rate`: 学习率
- `--device`: 训练设备 (cuda/cpu/auto)
- `--use_fast_loader`: 启用快速数据加载器
- `--use_hybrid_loader`: 启用混合数据加载器

## 📚 文档

### QUICK_START.md
快速开始指南，包含：
- 环境配置
- 数据准备
- 训练启动
- 结果查看

### README_CARDIAC_TRAINING.md
详细的训练文档，包含：
- 完整的配置说明
- 数据格式要求
- 高级训练选项
- 故障排除指南

## 🚀 快速开始

1. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

2. **准备数据**:
   - 确保有 `merged_ct_echo_data.csv` 标签文件
   - 确保有对应的 HDF5 图像文件

3. **开始训练**:
   ```bash
   python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json
   ```

4. **监控训练**:
   ```bash
   tensorboard --logdir outputs/hybrid_cardiac_training/tensorboard
   ```

## 📊 输出文件

训练完成后，在输出目录中会生成：
- `best_model.pth`: 最佳模型权重
- `training.log`: 详细训练日志
- `config.json`: 使用的配置
- `tensorboard/`: TensorBoard日志文件

## 🔧 自定义训练

### 修改配置文件
编辑 `configs/hybrid_cardiac_training_config.json`:

```json
{
  "epochs": 100,
  "batch_size": 24,
  "learning_rate": 5e-05,
  "regression_weight": 0.5,
  "classification_weight": 0.5
}
```

### 使用命令行覆盖
```bash
python examples/cardiac_training_example.py \
    --config configs/hybrid_cardiac_training_config.json \
    --epochs 200 \
    --batch_size 32
```

## 📋 注意事项

- 确保GPU内存足够（推荐16GB+）
- 首次运行会自动下载预训练权重
- 训练过程中会自动保存检查点
- 使用 `Ctrl+C` 可以安全停止训练 