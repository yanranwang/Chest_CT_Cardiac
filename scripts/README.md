# 心脏功能训练脚本使用说明

这个目录包含了用于心脏功能回归训练的shell脚本，提供了便捷的训练启动方式。

## 📁 文件结构

```
scripts/
├── train_cardiac.sh    # 主要训练脚本
├── quick_train.sh      # 快速启动脚本
└── README.md          # 使用说明（本文件）

configs/
└── cardiac_config.json # 配置文件示例
```

## 🚀 快速开始

### 方法1：使用快速启动脚本（推荐）

```bash
# 运行交互式快速启动
./scripts/quick_train.sh
```

然后根据提示选择训练模式：
- 1: 基础训练
- 2: 快速训练（使用预处理数据）
- 3: 调试模式
- 4: 生产模式
- 5: 自定义配置
- 6: 恢复训练

### 方法2：直接使用训练脚本

```bash
# 基础训练
./scripts/train_cardiac.sh basic

# 快速训练
./scripts/train_cardiac.sh fast

# 调试模式
./scripts/train_cardiac.sh debug

# 生产模式
./scripts/train_cardiac.sh production
```

## 📋 训练模式说明

### 1. 基础训练模式 (basic)
- **适用场景**: 标准训练，适合大多数情况
- **参数**: 100 epochs, batch_size=4, lr=1e-4
- **特点**: 平衡了训练效果和时间
- **命令**: `./scripts/train_cardiac.sh basic`

### 2. 快速训练模式 (fast)
- **适用场景**: 使用预处理数据的快速训练
- **前提**: 需要先运行数据预处理
- **特点**: 显著减少I/O时间
- **命令**: `./scripts/train_cardiac.sh fast`

### 3. 调试模式 (debug)
- **适用场景**: 代码调试和快速验证
- **参数**: 10 epochs, batch_size=2
- **特点**: 快速完成，适合测试
- **命令**: `./scripts/train_cardiac.sh debug`

### 4. 生产模式 (production)
- **适用场景**: 正式训练，追求最佳效果
- **参数**: 200 epochs, batch_size=8, lr=2e-4
- **特点**: 训练时间长，效果最好
- **命令**: `./scripts/train_cardiac.sh production`

### 5. 自定义配置模式 (custom)
- **适用场景**: 使用自定义配置文件
- **参数**: 从配置文件读取
- **命令**: `./scripts/train_cardiac.sh custom --config configs/cardiac_config.json`

### 6. 恢复训练模式 (resume)
- **适用场景**: 从检查点恢复训练
- **前提**: 需要有检查点文件
- **命令**: `./scripts/train_cardiac.sh resume --resume_from outputs/checkpoint.pth`

## ⚙️ 命令行参数

```bash
./scripts/train_cardiac.sh [模式] [选项]

选项:
  --config FILE        配置文件路径
  --epochs N           训练轮数
  --batch_size N       批量大小
  --learning_rate F    学习率
  --output_dir DIR     输出目录
  --csv_path FILE      CSV数据文件路径
  --device DEVICE      训练设备 (cuda/cpu)
  --resume_from FILE   恢复训练的检查点文件
  --use_fast_loader    使用快速数据加载器
  --preprocessed_dir DIR 预处理数据目录
  --dry_run            仅显示命令，不执行
  --help, -h           显示帮助信息
```

## 📊 使用示例

### 基础训练
```bash
# 默认参数训练
./scripts/train_cardiac.sh basic

# 自定义epochs和batch_size
./scripts/train_cardiac.sh basic --epochs 150 --batch_size 6

# 自定义输出目录
./scripts/train_cardiac.sh basic --output_dir /path/to/my/output
```

### 快速训练
```bash
# 首先预处理数据
python3 -m merlin.training.data_preprocessor --config configs/cardiac_config.json

# 然后使用快速训练
./scripts/train_cardiac.sh fast --epochs 50
```

### 调试模式
```bash
# 快速调试
./scripts/train_cardiac.sh debug

# 调试时查看命令但不执行
./scripts/train_cardiac.sh debug --dry_run
```

### 生产模式
```bash
# 完整生产训练
./scripts/train_cardiac.sh production

# 生产模式使用自定义数据路径
./scripts/train_cardiac.sh production --csv_path /path/to/my/data.csv
```

### 自定义配置
```bash
# 使用配置文件
./scripts/train_cardiac.sh custom --config configs/cardiac_config.json

# 配置文件 + 额外参数
./scripts/train_cardiac.sh custom --config configs/cardiac_config.json --epochs 80
```

### 恢复训练
```bash
# 从检查点恢复
./scripts/train_cardiac.sh resume --resume_from outputs/cardiac_training/checkpoint_epoch_50.pth
```

## 🔧 配置文件

使用 `configs/cardiac_config.json` 作为配置文件模板：

```json
{
  "output_dir": "outputs/cardiac_training",
  "epochs": 100,
  "batch_size": 4,
  "learning_rate": 1e-4,
  "device": "cuda",
  "use_fast_loader": false,
  ...
}
```

## 📈 训练输出

训练完成后，输出文件位于指定的输出目录：
- `best_model.pth`: 最佳模型权重
- `training.log`: 训练日志
- `config.json`: 使用的配置
- `tensorboard/`: TensorBoard日志

## 🔍 故障排除

### 常见问题

1. **权限问题**
   ```bash
   chmod +x scripts/train_cardiac.sh
   chmod +x scripts/quick_train.sh
   ```

2. **找不到训练脚本**
   - 确保在项目根目录运行
   - 检查 `examples/cardiac_training_example.py` 是否存在

3. **GPU内存不足**
   - 减少batch_size: `--batch_size 2`
   - 使用CPU: `--device cpu`

4. **数据文件不存在**
   - 检查CSV文件路径
   - 检查数据目录路径

5. **快速训练失败**
   - 确保已运行数据预处理
   - 检查预处理数据目录是否存在

### 获取帮助

```bash
# 显示帮助信息
./scripts/train_cardiac.sh --help

# 查看命令但不执行
./scripts/train_cardiac.sh debug --dry_run
```

## 🎯 训练建议

1. **首次使用**: 建议先运行调试模式验证环境
2. **数据预处理**: 对于重复训练，建议使用快速训练模式
3. **参数调优**: 可以从基础模式开始，逐步调整参数
4. **监控训练**: 使用TensorBoard查看训练过程
5. **检查点**: 定期保存检查点，便于恢复训练 