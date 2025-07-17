# SLURM 心脏功能训练指南

## 📋 概述

本指南介绍如何使用 SLURM 作业调度系统提交心脏功能训练任务，特别是如何使用 `submit_cardiac_training.sh` 脚本在多GPU环境中进行训练。

## 🚀 快速开始

### 1. 基本提交

```bash
# 使用默认配置提交训练任务
sbatch scripts/submit_cardiac_training.sh

# 使用自定义配置文件
sbatch scripts/submit_cardiac_training.sh --config configs/multi_gpu_training_config.json
```

### 2. 指定输出目录

```bash
# 指定自定义输出目录
sbatch scripts/submit_cardiac_training.sh --output_dir outputs/my_experiment_1

# 自动生成唯一目录（默认行为）
sbatch scripts/submit_cardiac_training.sh  # 会生成: outputs/fast_cardiac_training_20241201_143022_job12345
```

### 3. 完整参数示例

```bash
sbatch scripts/submit_cardiac_training.sh \
    --config configs/multi_gpu_training_config.json \
    --output_dir outputs/multi_gpu_experiment \
    --epochs 150 \
    --batch_size 36 \
    --learning_rate 1.5e-4 \
    --num_workers 20
```

## ⚙️ 脚本参数详解

### 必需参数
- **无必需参数**：脚本会使用默认配置

### 可选参数
| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--config` | 配置文件路径 | `configs/fast_training_config.json` | `--config configs/multi_gpu_training_config.json` |
| `--output_dir` | 输出目录 | 自动生成时间戳目录 | `--output_dir outputs/my_exp` |
| `--epochs` | 训练轮数 | 从配置文件读取 | `--epochs 100` |
| `--batch_size` | 批量大小 | 从配置文件读取 | `--batch_size 48` |
| `--learning_rate` | 学习率 | 从配置文件读取 | `--learning_rate 2e-4` |
| `--num_workers` | 工作进程数 | 从配置文件读取 | `--num_workers 24` |

## 🖥️ 资源配置

### 当前配置
```bash
#SBATCH --gres=gpu:rtx8000:3    # 3张RTX8000 GPU
#SBATCH --cpus-per-task=24      # 24个CPU核心
#SBATCH --mem=64G               # 64GB内存
#SBATCH --time=24:00:00         # 24小时时间限制
```

### 自定义资源配置
如需修改资源配置，编辑 `scripts/submit_cardiac_training.sh` 文件中的 `#SBATCH` 行：

```bash
# 修改GPU数量
#SBATCH --gres=gpu:rtx8000:2    # 改为2张GPU

# 修改内存
#SBATCH --mem=128G              # 改为128GB

# 修改时间限制
#SBATCH --time=48:00:00         # 改为48小时
```

## 📊 推荐配置

### 配置1：快速训练（单GPU等效）
```bash
sbatch scripts/submit_cardiac_training.sh \
    --config configs/fast_training_config.json \
    --output_dir outputs/fast_training \
    --batch_size 24 \
    --epochs 50
```

### 配置2：多GPU训练（推荐）
```bash
sbatch scripts/submit_cardiac_training.sh \
    --config configs/multi_gpu_training_config.json \
    --output_dir outputs/multi_gpu_training \
    --epochs 100
```

### 配置3：长时间训练
```bash
sbatch scripts/submit_cardiac_training.sh \
    --config configs/multi_gpu_training_config.json \
    --output_dir outputs/production_training \
    --epochs 200 \
    --batch_size 48 \
    --learning_rate 2e-4
```

## 🔧 环境配置

### 修改环境加载
根据您的集群环境，编辑脚本中的环境配置部分：

```bash
# 在 scripts/submit_cardiac_training.sh 中修改
# 模块加载
module load python/3.9
module load cuda/11.8
module load cudnn/8.6

# Python环境激活
source /path/to/your/conda/bin/activate your_env_name
# 或者
source /path/to/your/venv/bin/activate
```

## 📋 作业管理

### 查看作业状态
```bash
# 查看所有作业
squeue -u $USER

# 查看特定作业
squeue -j <job_id>

# 查看作业详情
scontrol show job <job_id>
```

### 取消作业
```bash
# 取消特定作业
scancel <job_id>

# 取消所有作业
scancel -u $USER
```

### 监控作业输出
```bash
# 实时查看输出日志
tail -f logs/cardiac_training_<job_id>.out

# 实时查看错误日志
tail -f logs/cardiac_training_<job_id>.err

# 查看训练日志
tail -f outputs/your_output_dir/training.log
```

## 📁 输出文件结构

训练完成后，输出目录包含：

```
outputs/your_output_dir/
├── best_model.pth              # 最佳模型权重
├── checkpoint_best.pth         # 最佳检查点
├── checkpoint_latest.pth       # 最新检查点
├── config.json                 # 训练配置
├── training.log                # 训练日志
├── tensorboard/                # TensorBoard日志
└── data_info.json             # 数据统计信息
```

## 🔍 故障排除

### 常见问题

1. **作业排队时间过长**
   ```bash
   # 查看资源可用性
   sinfo -p gpu
   
   # 查看队列状态
   squeue -p gpu
   ```

2. **GPU内存不足**
   - 减少 `batch_size` 参数
   - 调整 `cache_size` 在配置文件中
   - 检查是否有其他进程占用GPU

3. **文件权限问题**
   ```bash
   # 确保脚本有执行权限
   chmod +x scripts/submit_cardiac_training.sh
   
   # 检查输出目录权限
   ls -la outputs/
   ```

4. **环境问题**
   - 确保Python环境正确激活
   - 验证CUDA版本兼容性
   - 检查依赖包是否安装完整

### 日志检查

```bash
# 检查SLURM输出
cat logs/cardiac_training_<job_id>.out

# 检查SLURM错误
cat logs/cardiac_training_<job_id>.err

# 检查训练日志
cat outputs/your_output_dir/training.log
```

## 🚀 性能优化建议

### 1. 批量大小优化
```bash
# 根据GPU数量调整batch_size
# 单GPU: 16-24
# 双GPU: 32-48  
# 三GPU: 48-72
```

### 2. 工作进程数优化
```bash
# 根据CPU核心数调整num_workers
# 一般设为: min(CPU_cores, batch_size)
```

### 3. 内存缓存优化
```bash
# 在配置文件中调整cache_size
# 根据可用内存: 200-1000
```

## 📈 监控和可视化

### TensorBoard
```bash
# 启动TensorBoard
tensorboard --logdir outputs/your_output_dir/tensorboard --port 6006

# 通过SSH端口转发访问
ssh -L 6006:localhost:6006 user@cluster
```

### GPU监控
```bash
# 监控GPU使用率
watch -n 1 nvidia-smi

# 查看特定作业的GPU使用
nvidia-smi -i 0,1,2
```

## 🎯 最佳实践

1. **使用有意义的输出目录名称**
   ```bash
   --output_dir outputs/experiment_freeze_encoder_lr2e4_bs48
   ```

2. **保存多个检查点**
   - 设置较小的 `save_interval`
   - 定期备份重要检查点

3. **监控训练进度**
   - 使用TensorBoard可视化
   - 定期检查训练日志

4. **资源使用优化**
   - 根据数据集大小调整缓存设置
   - 监控内存使用情况
   - 合理设置工作进程数

## 📞 获取帮助

```bash
# 查看脚本帮助
sbatch scripts/submit_cardiac_training.sh --help

# 查看SLURM帮助
man sbatch
man squeue
man scontrol
``` 