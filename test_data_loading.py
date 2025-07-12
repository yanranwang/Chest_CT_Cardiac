#!/usr/bin/env python3
"""
测试数据读取功能的脚本
从CSV文件中读取basename和folder列，并构建文件路径
"""

import pandas as pd
import os
from pathlib import Path

def test_data_loading():
    """测试数据读取功能"""
    # 配置参数
    csv_path = 'filtered_echo_chestCT_data_filtered_chest_data.csv'
    base_path = '/dataNAS/data/ct_data/ct_scans'
    
    print(f"读取CSV文件: {csv_path}")
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件 {csv_path} 不存在")
        return
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"CSV文件包含 {len(df)} 行数据")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查是否包含必要的列
    required_cols = ['basename', 'folder']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误: CSV文件缺少必要的列: {missing_cols}")
        return
    
    print("\n前5行数据的basename和folder列:")
    print(df[['basename', 'folder']].head())
    
    # 构建文件路径并检查前几个
    print("\n构建的文件路径示例:")
    valid_files = 0
    invalid_files = 0
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if i >= 10:  # 只检查前10个文件
            break
            
        basename = row['basename']
        folder = row['folder']
        
        # 构建文件路径
        image_path = f"{base_path}/stanford_{folder}/{basename}.nii.gz"
        
        # 检查文件是否存在
        exists = os.path.exists(image_path)
        if exists:
            valid_files += 1
        else:
            invalid_files += 1
        
        print(f"  {i+1:2d}. {image_path} {'✓' if exists else '✗'}")
    
    print(f"\n在前10个样本中:")
    print(f"  存在的文件: {valid_files}")
    print(f"  不存在的文件: {invalid_files}")
    
    # 统计不同文件夹的数量
    print(f"\n文件夹分布:")
    folder_counts = df['folder'].value_counts()
    print(folder_counts)

if __name__ == '__main__':
    test_data_loading() 