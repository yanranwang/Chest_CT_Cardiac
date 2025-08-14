# Scripts 工具脚本

这个目录包含用于数据处理和系统维护的辅助脚本。

## 📁 脚本说明

### merge_csv_data.py
**功能**: 合并CT和Echo数据的CSV文件

**用途**: 
- 将包含basename/folder的CT数据与包含标签的Echo数据合并
- 生成用于训练的完整标签文件

**使用方法**:
```bash
python scripts/merge_csv_data.py
```

**输入文件**:
- `filtered_echo_chestCT_data_filtered_chest_data.csv` - CT数据
- `filtered_echo_chestCT_data_filtered_echo_data.csv` - Echo数据

**输出文件**:
- `merged_ct_echo_data.csv` - 合并后的完整数据

### test_hybrid_loader.py
**功能**: 测试混合数据加载器

**用途**:
- 验证HybridCardiacDataset是否正常工作
- 检查数据匹配情况
- 测试样本加载

**使用方法**:
```bash
python scripts/test_hybrid_loader.py
```

## 🔧 使用场景

1. **初次设置**: 运行 `merge_csv_data.py` 来创建训练数据
2. **调试**: 使用 `test_hybrid_loader.py` 验证数据加载
3. **维护**: 当数据源更新时重新运行合并脚本

## 📋 注意事项

- 确保输入文件路径正确
- 检查文件权限
- 运行前备份重要数据文件 