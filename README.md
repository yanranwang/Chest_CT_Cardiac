# 心脏功能预测 - 混合数据加载器训练系统

基于胸部CT影像和超声心动图数据的心脏功能预测模型训练系统。

## 🎯 核心功能

- **混合数据加载**: 从CSV文件读取标签，从HDF5文件读取预处理的图像数据
- **心脏功能预测**: 同时进行LVEF回归和主动脉狭窄(AS)分类
- **高效训练**: 利用预处理的HDF5数据实现快速训练

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保以下文件存在：
- **标签文件**: `merged_ct_echo_data.csv` - 包含basename, folder, lvef, AS_maybe列
- **图像文件**: HDF5格式的预处理图像数据
- **配置文件**: `configs/hybrid_cardiac_training_config.json`

### 3. 开始训练

```bash
python examples/cardiac_training_example.py --config configs/hybrid_cardiac_training_config.json
```

## 📊 数据格式

### CSV标签文件格式
```csv
basename,folder,lvef,AS_maybe,patient_id
LA3dd33e5-LA3dd5b65,1A,61.47,0.0,patient_001
LA3dd74cb-LA3dd962e,1A,55.23,1.0,patient_002
```

### HDF5图像文件格式
- 路径: `/path/to/preprocessed_data.h5`
- 结构: `images/` 组包含哈希键名的图像数据
- 元数据: `data_metadata.json` 提供哈希到basename/folder的映射

## ⚙️ 配置说明

关键配置参数 (`configs/hybrid_cardiac_training_config.json`):

```json
{
  "use_hybrid_loader": true,
  "csv_path": "/path/to/merged_ct_echo_data.csv",
  "hdf5_path": "/path/to/preprocessed_data.h5",
  "label_columns": ["lvef", "AS_maybe"],
  "epochs": 50,
  "batch_size": 24,
  "learning_rate": 5e-05
}
```

## 📁 项目结构

```
├── configs/
│   ├── hybrid_cardiac_training_config.json  # 训练配置
│   └── README_hybrid_training.md            # 详细使用说明
├── examples/
│   └── cardiac_training_example.py          # 主训练脚本
├── merlin/
│   ├── training/
│   │   ├── fast_dataloader.py              # 混合数据加载器
│   │   └── cardiac_trainer.py              # 训练器
│   ├── models/                             # 模型定义
│   └── data/                               # 数据处理工具
├── scripts/
│   └── merge_csv_data.py                   # CSV数据合并工具
├── merged_ct_echo_data.csv                 # 合并的标签数据
└── requirements.txt                        # 依赖包
```

## 🔧 核心组件

### HybridCardiacDataset
混合数据加载器，支持：
- 从CSV读取标签数据
- 从HDF5读取图像数据
- 智能哈希映射匹配
- 内存缓存优化

### CardiacTrainer
训练器，支持：
- 多任务学习 (回归+分类)
- 类别权重平衡
- TensorBoard可视化
- 自动模型保存

## 📈 训练监控

### TensorBoard
```bash
tensorboard --logdir outputs/hybrid_cardiac_training/tensorboard
```

### 训练日志
- 位置: `outputs/hybrid_cardiac_training/training.log`
- 包含: 损失曲线、指标统计、模型保存信息

## 🎯 模型输出

- **LVEF回归**: 预测左心室射血分数 (5-90%)
- **AS分类**: 预测主动脉狭窄风险 (0: 正常, 1: 可能AS)

## 📋 系统要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (GPU训练)
- 16GB+ RAM (推荐)

## 🔍 故障排除

### 常见问题

1. **数据匹配失败**
   - 检查CSV中的basename/folder列
   - 验证HDF5文件路径和元数据文件

2. **内存不足**
   - 减少batch_size
   - 调整cache_size
   - 设置preload_data为false

3. **训练速度慢**
   - 增加num_workers
   - 启用GPU训练
   - 调整缓存设置

详细说明请参考: `configs/README_hybrid_training.md`

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！
