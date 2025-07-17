# 快速训练使用指南

## 🚀 预处理数据加速训练

### 📊 配置概述

预处理数据已存储在训练机器上：
- **预处理数据目录**: `/data/joycewyr/cardiac_training_fast/`
- **HDF5数据文件**: `preprocessed_data.h5`
- **元数据文件**: `data_metadata.json`

### 🎯 快速训练的优势

使用预处理数据可以显著提升训练速度：
- ✅ **IO加速**: 直接读取HDF5文件，避免重复图像预处理
- ✅ **内存优化**: 高效的数据加载和缓存机制
- ✅ **批处理优化**: 预处理的数据已经优化为训练批次
- ✅ **时间节省**: 减少70-80%的数据加载时间

## 🔧 使用方法

### 1. 启用快速训练模式

#### 方法1：使用shell脚本（推荐）
```bash
# 使用快速训练模式
./scripts/train_cardiac.sh fast

# 自定义参数的快速训练
./scripts/train_cardiac.sh fast --epochs 100 --batch_size 8 --num_workers 16
```

#### 方法2：直接Python命令
```bash
# 启用快速数据加载器
python3 examples/cardiac_training_example.py \
    --use_fast_loader \
    --preprocessed_data_dir /data/joycewyr/cardiac_training_fast \
    --batch_size 8 \
    --epochs 100 \
    --num_workers 16
```

#### 方法3：使用配置文件
```json
{
  "use_fast_loader": true,
  "preprocessed_data_dir": "/data/joycewyr/cardiac_training_fast",
  "batch_size": 8,
  "num_workers": 16,
  "epochs": 100
}
```

然后运行：
```bash
python3 examples/cardiac_training_example.py --config configs/cardiac_config.json
```

### 2. 验证预处理数据

训练脚本会自动检查预处理数据：
```
🚀 Using fast data loader mode
✅ Found preprocessed data: /data/joycewyr/cardiac_training_fast/preprocessed_data.h5
✅ Found metadata file: /data/joycewyr/cardiac_training_fast/data_metadata.json
```

## 📈 性能优化建议

### Worker数量优化
基于你的88核CPU系统，推荐配置：

| 训练模式 | Worker数量 | 批次大小 | 预期速度提升 |
|----------|------------|----------|--------------|
| 开发测试 | 8 | 4 | 3-4x |
| 标准训练 | 16 | 8 | 5-6x |
| 高性能训练 | 32 | 16 | 7-8x |

### 内存优化
```json
{
  "cache_config": {
    "enable_cache": true,
    "cache_size": 2000,
    "preload_train_data": true,
    "preload_val_data": true
  }
}
```

## 🎯 完整训练命令示例

### 基础快速训练
```bash
./scripts/train_cardiac.sh fast \
    --epochs 100 \
    --batch_size 8 \
    --num_workers 16 \
    --learning_rate 1e-4
```

### 高性能训练
```bash
./scripts/train_cardiac.sh fast \
    --epochs 200 \
    --batch_size 16 \
    --num_workers 32 \
    --learning_rate 2e-4 \
    --output_dir outputs/fast_training_high_perf
```

### 自定义配置训练
```bash
./scripts/train_cardiac.sh custom \
    --config configs/cardiac_config.json \
    --num_workers 16
```

## 🔍 故障排除

### 常见问题及解决方案

1. **预处理数据文件不存在**
   ```
   ❌ Preprocessed data file not found: /data/joycewyr/cardiac_training_fast/preprocessed_data.h5
   ```
   - 确保在训练机器上运行
   - 检查文件路径是否正确

2. **元数据文件缺失**
   ```
   ❌ Metadata file not found: /data/joycewyr/cardiac_training_fast/data_metadata.json
   ```
   - 检查预处理过程是否完整
   - 确保两个文件都存在

3. **内存不足**
   ```
   CUDA out of memory
   ```
   - 减少batch_size: `--batch_size 4`
   - 减少worker数量: `--num_workers 8`
   - 禁用数据预加载: `"preload_train_data": false`

4. **数据加载慢**
   ```
   GPU utilization low
   ```
   - 增加worker数量: `--num_workers 32`
   - 启用数据缓存: `"enable_cache": true`
   - 预加载数据: `"preload_train_data": true`

## 📊 性能监控

### 训练过程监控
```bash
# 监控GPU使用情况
nvidia-smi -l 1

# 监控系统资源
htop

# 查看训练日志
tail -f outputs/cardiac_training/training.log
```

### 性能指标
- **数据加载时间**: 应该显著减少
- **GPU利用率**: 应该保持在80%以上
- **内存使用**: 监控避免OOM
- **训练速度**: 每个epoch的时间应该减少50%以上

## 🎉 预期效果

使用预处理数据的快速训练模式，你可以期待：

1. **训练速度提升**: 70-80%的时间节省
2. **资源利用率**: 更高的GPU利用率
3. **稳定性**: 更稳定的训练过程
4. **扩展性**: 支持更大的batch size

## 🚀 开始训练

准备好开始快速训练了吗？使用以下命令：

```bash
# 快速开始
./scripts/quick_train.sh
# 然后选择 "2. 快速训练 (使用预处理数据)"

# 或者直接命令行
./scripts/train_cardiac.sh fast --num_workers 16 --batch_size 8
```

享受极速的心脏功能训练体验！🏃‍♂️💨 