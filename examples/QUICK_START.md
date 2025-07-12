# 快速开始指南

## 心脏功能训练示例

现在您可以立即运行心脏功能训练示例，无需准备真实的心脏功能标签数据。

### 快速运行

从项目根目录运行：

```bash
cd examples
python cardiac_training_example.py --epochs 2 --batch_size 2
```

这将：
- 使用模拟的心脏功能标签进行训练
- 运行2个epoch（用于快速演示）
- 使用小批量大小以减少内存使用

### 关键信息

- **模型输出**: 训练的是 LVEF（左室射血分数）回归 + AS（主动脉瓣狭窄）分类的双任务模型
- **数据**: 使用现有的CT扫描元数据文件，但心脏功能标签会自动生成模拟数据
- **设备**: 代码会自动检测并使用GPU（如果可用），否则使用CPU

### 预期输出

您应该看到类似以下的输出：

```
使用默认配置
配置已保存到: outputs/cardiac_training/config.json
创建数据加载器...
从 ../filtered_echo_chestCT_data_filtered_chest_data.csv 读取数据...
原始数据集大小: 4950 行
注意: 配置中未指定心脏功能指标列，将使用模拟标签进行训练演示
移除了 2475 行缺失basename或folder的数据
清理后数据集大小: 2475 行
成功构建 2475 个数据项
数据分割完成:
  训练集: 1980 个样本
  验证集: 495 个样本
初始化训练器...
开始训练...
```

### 输出文件

训练完成后，您将在 `outputs/cardiac_training/` 目录中找到：

- `config.json` - 训练配置
- `best_model.pth` - 最佳模型权重
- `training.log` - 训练日志
- `data_info.json` - 数据统计信息
- `tensorboard/` - TensorBoard日志（可选）

### 下一步

1. **查看训练日志**: 检查 `outputs/cardiac_training/training.log`
2. **可视化训练过程**: 使用 TensorBoard 查看训练曲线
3. **测试模型**: 使用 `simple_cardiac_example.py` 测试训练好的模型

### 故障排除

如果遇到问题：

1. **CUDA内存不足**: 减少 `--batch_size` 参数
2. **找不到CSV文件**: 确保您在 `examples` 目录中运行命令
3. **权限错误**: 确保有写入 `outputs/` 目录的权限

### 自定义训练

您可以通过以下参数自定义训练：

```bash
# 更长的训练
python cardiac_training_example.py --epochs 20 --batch_size 4

# 自定义输出目录
python cardiac_training_example.py --output_dir my_cardiac_model

# 使用配置文件
python cardiac_training_example.py --config my_config.json
``` 