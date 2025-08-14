# 安装和配置指南

## 🔧 系统要求

- **操作系统**: Linux/macOS/Windows
- **Python**: 3.8 或更高版本
- **GPU**: NVIDIA GPU with CUDA 11.0+ (推荐)
- **内存**: 16GB RAM (推荐)
- **存储**: 10GB+ 可用空间

## 📦 安装步骤

### 1. 克隆项目
```bash
git clone <repository-url>
cd Chest_CT_Cardiac
```

### 2. 创建虚拟环境 (推荐)
```bash
# 使用conda
conda create -n cardiac python=3.8
conda activate cardiac

# 或使用venv
python -m venv cardiac_env
source cardiac_env/bin/activate  # Linux/macOS
# cardiac_env\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import monai; print(f'MONAI: {monai.__version__}')"
```

## 📊 数据准备

### 1. 准备CSV标签文件
确保您的CSV文件包含以下必需列：
- `basename`: 文件基础名称
- `folder`: 文件夹名称
- `lvef`: 左心室射血分数
- `AS_maybe`: 主动脉狭窄标签

### 2. 准备HDF5图像文件
- 预处理的图像数据存储在HDF5格式
- 包含对应的 `data_metadata.json` 元数据文件

### 3. 合并数据 (如果需要)
```bash
python scripts/merge_csv_data.py
```

## ⚙️ 配置文件设置

编辑 `configs/hybrid_cardiac_training_config.json`:

```json
{
  "csv_path": "/path/to/your/merged_ct_echo_data.csv",
  "hdf5_path": "/path/to/your/preprocessed_data.h5",
  "output_dir": "outputs/my_training",
  "epochs": 50,
  "batch_size": 24,
  "learning_rate": 5e-05
}
```

### 重要配置项说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `csv_path` | CSV标签文件路径 | 绝对路径 |
| `hdf5_path` | HDF5图像文件路径 | 绝对路径 |
| `batch_size` | 批量大小 | 16-32 (根据GPU内存) |
| `num_workers` | 数据加载进程数 | CPU核心数的一半 |
| `learning_rate` | 学习率 | 1e-5 到 1e-4 |

## 🚀 运行训练

### 基本训练
```bash
python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json
```

### 自定义参数
```bash
python examples/cardiac_training_example.py \
    --config configs/hybrid_cardiac_training_config.json \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4
```

### GPU设置
```bash
# 指定GPU
CUDA_VISIBLE_DEVICES=0 python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json

# 使用CPU (如果没有GPU)
python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json --device cpu
```

## 📈 监控训练

### TensorBoard
```bash
tensorboard --logdir outputs/hybrid_cardiac_training/tensorboard --port 6006
```
然后在浏览器中访问 `http://localhost:6006`

### 查看日志
```bash
tail -f outputs/hybrid_cardiac_training/training.log
```

## 🔍 故障排除

### 常见问题

1. **CUDA out of memory**
   ```bash
   # 减少批量大小
   --batch_size 8
   ```

2. **数据加载慢**
   ```bash
   # 增加工作进程
   --num_workers 8
   ```

3. **找不到数据文件**
   - 检查文件路径是否正确
   - 使用绝对路径
   - 确认文件权限

4. **导入错误**
   ```bash
   # 重新安装依赖
   pip install -r requirements.txt --force-reinstall
   ```

### 测试数据加载器
```bash
python scripts/test_hybrid_loader.py
```

### 验证配置
```bash
python -c "
import json
with open('configs/hybrid_cardiac_training_config.json') as f:
    config = json.load(f)
print('配置验证通过')
"
```

## 📋 性能优化建议

### GPU优化
- 使用混合精度训练: `--use_amp`
- 调整批量大小以充分利用GPU内存
- 使用多GPU: `--device cuda`

### 数据加载优化
- 增加 `num_workers` (通常为CPU核心数的一半)
- 启用 `pin_memory=True`
- 调整缓存大小 `cache_size`

### 内存优化
- 减少批量大小
- 设置 `preload_data=False`
- 调整缓存配置

## 🎯 下一步

1. **开始训练**: 使用提供的配置开始第一次训练
2. **监控结果**: 使用TensorBoard查看训练进度
3. **调优参数**: 根据结果调整学习率和批量大小
4. **评估模型**: 使用验证集评估模型性能

## 📞 获取帮助

- 查看 `README.md` 了解项目概述
- 阅读 `configs/README_hybrid_training.md` 了解详细配置
- 参考 `examples/README.md` 了解使用示例 