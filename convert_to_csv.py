#!/usr/bin/env python3
"""
将医疗数据转换为CSV格式

这个脚本将用户提供的CT检查数据转换为CSV文件
"""

import pandas as pd
from datetime import datetime

def create_csv_from_data():
    """将提供的数据转换为CSV格式"""
    
    print("正在创建CSV文件...")
    
    # 根据用户提供的数据创建DataFrame
    # 注意：用户显示的是前5行数据，包含17列
    data = {
        'mrn': [13.0, 264481.0, 264481.0, 571315.0, 626788.0],
        'accession_number': [17229665, 14954316, 15215902, 17383152, 17047772],
        'proc_start_time': [
            '2020-09-30 09:04:00',
            '2019-06-05 15:00:00', 
            '2019-07-17 13:15:00',
            '2020-08-07 09:38:00',
            '2020-05-31 13:45:00'
        ],
        'patient_id': [165390.0, 23125.0, 23125.0, 90472.0, 112557.0],
        'proc_code': ['IMGCT0148', 'IMGCT0148', 'IMGCT0148', 'IMGCT0148', 'IMGCT0148'],
        'code': ['Computed Tomography', 'Computed Tomography', 'Computed Tomography', 
                'Computed Tomography', 'Computed Tomography']
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 转换时间列为datetime格式
    df['proc_start_time'] = pd.to_datetime(df['proc_start_time'])
    
    # 显示数据信息
    print("数据预览:")
    print(df)
    print(f"\n数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 保存为CSV文件
    csv_filename = 'ct_scan_data.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print(f"\n✅ 数据已成功保存为: {csv_filename}")
    print(f"   - 行数: {len(df)}")
    print(f"   - 列数: {len(df.columns)}")
    
    # 显示CSV文件的前几行
    print(f"\n📄 CSV文件内容预览:")
    print("-" * 80)
    with open(csv_filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 7:  # 显示前7行（包括标题行）
                print(line.strip())
            else:
                break
    
    return csv_filename

def add_missing_columns():
    """
    如果您的原始数据有17列，可以使用这个函数添加其他可能的列
    """
    print("\n注意: 您提到原始数据有17列，但只显示了6列。")
    print("如果需要添加其他列，请提供完整的列名和数据。")
    
    # 常见的CT检查相关列可能包括：
    possible_columns = [
        'study_date',           # 检查日期
        'modality',            # 检查方式
        'body_part',           # 检查部位
        'institution',         # 医疗机构
        'referring_physician', # 转诊医生
        'study_description',   # 检查描述
        'series_description',  # 序列描述
        'slice_thickness',     # 层厚
        'kvp',                 # 千伏峰值
        'exposure_time',       # 曝光时间
        'tube_current'         # 管电流
    ]
    
    print("可能的其他列包括:")
    for i, col in enumerate(possible_columns, 1):
        print(f"  {i:2d}. {col}")

if __name__ == "__main__":
    print("=== CT检查数据转换为CSV ===\n")
    
    # 执行转换
    csv_file = create_csv_from_data()
    
    # 显示可能的其他列
    add_missing_columns()
    
    print(f"\n=== 转换完成 ===")
    print(f"生成的CSV文件: {csv_file}") 