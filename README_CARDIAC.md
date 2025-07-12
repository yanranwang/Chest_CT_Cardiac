# Merlin 心脏功能预测扩展

本文档介绍如何使用扩展后的 Merlin 模型进行心脏功能回归预测。

## 功能概述

基于原有的 Merlin 3D CT视觉语言模型，我们添加了专门的心脏功能回归预测能力，包括：

### 🫀 心脏功能指标预测

模型可以预测以下10个关键心脏功能指标：

1. **射血分数 (Ejection Fraction)** - 心脏每次跳动时泵出的血液百分比
2. **每搏输出量 (Stroke Volume)** - 每次心跳泵出的血液量 (mL)
3. **心输出量 (Cardiac Output)** - 每分钟心脏泵出的血液量 (L/min)
4. **心率变异性 (Heart Rate Variability)** - 心跳间隔的变化 (ms)
5. **左心室质量 (Left Ventricular Mass)** - 左心室肌肉的重量 (g)
6. **室壁厚度 (Wall Thickness)** - 心室壁的厚度 (mm)
7. **心室容积 (Chamber Volume)** - 心室的容积 (mL)
8. **收缩性指数 (Contractility Index)** - 心肌收缩能力指标
9. **舒张功能 (Diastolic Function)** - 心脏舒张期功能指标
10. **瓣膜功能 (Valvular Function)** - 心脏瓣膜功能指标

## 🚀 快速开始

### 安装依赖

确保安装了必要的依赖包：

```bash
pip install torch torchvision monai transformers scikit-learn tensorboard
```

### 基本使用

#### 1. 训练心脏功能预测模型

```python
from merlin.training.cardiac_trainer import CardiacTrainer, create_data_loaders

# 配置训练参数
config = {
    'output_dir': 'outputs/cardiac_training',
    'pretrained_model_path': 'path/to/merlin_weights.pth',  # Merlin预训练权重
    'num_cardiac_metrics': 10,
    'epochs': 100,
    'batch_size': 4,
    'learning_rate': 1e-4,
    'device': 'cuda',
    'freeze_encoder': True  # 冻结预训练编码器进行微调
}

# 创建数据加载器（需要您的心脏CT数据）
train_loader, val_loader = create_data_loaders(config)

# 创建训练器并开始训练
trainer = CardiacTrainer(config)
trainer.train(train_loader, val_loader)
```

#### 2. 推理心脏功能

```python
from merlin.inference.cardiac_inference import CardiacInference

# 加载训练好的模型
predictor = CardiacInference('outputs/cardiac_training/best_model.pth')

# 预测单个CT扫描
predictions = predictor.predict_single('path/to/ct_scan.nii.gz')

print("心脏功能预测结果:")
for metric, value in predictions.items():
    print(f"  {metric}: {value:.2f}")

# 生成详细报告
report = predictor.generate_report(predictions, patient_id="PATIENT_001")
print(report)
```

#### 3. 批量预测

```python
# 批量预测多个CT扫描
image_paths = ['scan1.nii.gz', 'scan2.nii.gz', 'scan3.nii.gz']
results = predictor.predict_batch(image_paths)

# 保存结果
predictor.save_predictions(results, 'outputs/batch_predictions.json')
```

#### 4. 不确定性估计

```python
# 使用蒙特卡洛dropout获得预测不确定性
uncertainty_results = predictor.predict_with_confidence(
    'path/to/ct_scan.nii.gz', 
    num_samples=20
)

for metric, stats in uncertainty_results.items():
    print(f"{metric}: {stats['mean']:.2f} ± {stats['std']:.2f}")
```

## 📊 演示脚本

我们提供了完整的演示脚本来展示训练和推理流程：

```bash
# 运行完整演示（训练+推理）
python examples/cardiac_demo.py --mode all

# 仅训练
python examples/cardiac_demo.py --mode train

# 仅推理
python examples/cardiac_demo.py --mode inference --model_path outputs/cardiac_training/best_model.pth
```

## 🏗️ 模型架构

### CardiacFunctionModel

扩展的模型架构包含以下组件：

1. **CardiacImageEncoder**: 基于预训练Merlin的3D图像编码器
2. **CardiacRegressionHead**: 专门的回归预测头
3. **CardiacMetricsCalculator**: 心脏指标计算和标准化工具

```python
from merlin.models.cardiac_regression import CardiacFunctionModel

# 创建模型
model = CardiacFunctionModel(
    pretrained_model_path='path/to/merlin_weights.pth',
    num_cardiac_metrics=10
)

# 前向传播
cardiac_preds, ehr_preds = model(ct_images)
```

### 权重加载

模型支持从预训练的Merlin权重初始化：

```python
# 自动加载Merlin预训练权重
model = CardiacFunctionModel(pretrained_model_path='merlin_weights.pth')

# 冻结图像编码器进行微调
model.freeze_encoder(freeze=True)
```

## 📈 训练配置

### 推荐的训练参数

```python
config = {
    # 模型参数
    'pretrained_model_path': 'path/to/merlin_weights.pth',
    'num_cardiac_metrics': 10,
    'freeze_encoder': True,  # 推荐冻结预训练编码器
    
    # 训练参数
    'epochs': 100,
    'batch_size': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'optimizer': 'adamw',
    'loss_function': 'mse',
    
    # 学习率调度
    'scheduler': {
        'type': 'cosine',
        'eta_min': 1e-6
    },
    
    # 训练设置
    'grad_clip': 1.0,
    'device': 'cuda',
    'num_workers': 4,
    'save_interval': 10
}
```

### 损失函数选择

支持多种回归损失函数：

- `mse`: 均方误差（默认）
- `mae`: 平均绝对误差
- `smooth_l1`: 平滑L1损失
- `huber`: Huber损失

### 学习率调度器

支持多种学习率调度策略：

- `cosine`: 余弦退火（推荐）
- `step`: 阶梯式衰减
- `plateau`: 基于验证损失的自适应调整

## 🔧 数据格式

### 训练数据格式

数据应组织为包含以下字段的字典列表：

```python
data_sample = {
    'image': 'path/to/ct_scan.nii.gz',  # CT图像文件路径
    'cardiac_metrics': np.array([...]),  # 10个心脏功能指标的标准化值
    'patient_id': 'PATIENT_001'  # 患者ID（可选）
}
```

### 心脏功能指标标准化

指标值应标准化到合理的生理范围：

```python
from merlin.models.cardiac_regression import CardiacMetricsCalculator

# 获取指标名称
metric_names = CardiacMetricsCalculator.get_metric_names()

# 标准化预测值到生理范围
normalized_preds = CardiacMetricsCalculator.normalize_predictions(raw_predictions)
```

## 📋 评估指标

训练过程中会计算以下评估指标：

- **MSE**: 均方误差
- **MAE**: 平均绝对误差  
- **R²**: 决定系数
- 每个心脏功能指标的单独MSE、MAE、R²

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```python
   # 减少batch size
   config['batch_size'] = 2
   
   # 使用梯度累积
   config['gradient_accumulation_steps'] = 2
   ```

2. **预训练权重加载失败**
   ```python
   # 确保Merlin预训练权重路径正确
   config['pretrained_model_path'] = 'correct/path/to/merlin_weights.pth'
   ```

3. **数据加载错误**
   ```python
   # 检查CT图像文件格式和路径
   # 确保图像为.nii.gz格式
   ```

### 调试模式

启用详细日志进行调试：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 训练时会输出详细信息
trainer = CardiacTrainer(config)
```

## 📚 参考文献

基于以下研究的实现：

1. Merlin: A Vision Language Foundation Model for 3D Computed Tomography
2. 相关心脏功能评估的医学研究

## 📞 技术支持

如有问题或建议，请：

1. 查看本文档的故障排除部分
2. 检查示例代码 `examples/cardiac_demo.py`
3. 提交Issue到项目仓库

---

**注意**: 本模型预测结果仅供研究和临床参考使用，不能替代专业医学诊断。在临床应用前请确保充分验证模型性能。 