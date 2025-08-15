#!/usr/bin/env python3
"""
Convert medical data to CSV format

This script converts user-provided CT examination data to CSV files
"""

import pandas as pd
from datetime import datetime

def create_csv_from_data():
    """Convert provided data to CSV format"""
    
    print("Creating CSV file...")
    
    # Create DataFrame based on user-provided data
    # Note: User shows first 5 rows of data, including 17 columns
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
    
    # CreateDataFrame
    df = pd.DataFrame(data)
    
    # Convert time column to datetime format
    df['proc_start_time'] = pd.to_datetime(df['proc_start_time'])
    
    # showdatainfo
    print("Data preview:")
    print(df)
    print(f"\nData shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    
    # Save as CSV file
    csv_filename = 'ct_scan_data.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print(f"\n✅ Data successfully saved as: {csv_filename}")
    print(f"   - Row count: {len(df)}")
    print(f"   - 列数: {len(df.columns)}")
    
    # Show first few rows of CSV file
    print(f"\n📄 CSV文件内容预览:")
    print("-" * 80)
    with open(csv_filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 7:  # Show first 7 rows (including header)
                print(line.strip())
            else:
                break
    
    return csv_filename

def add_missing_columns():
    """
    如果您的原始数据有17列，可以使用这个函数Add其他可能的列
    """
    print("\n注意: 您提到原始数据有17列，但只Show了6列。")
    print("如果需要Add其他列，请提供完整的列名和数据。")
    
    # Common CT examination related columns may include:
    possible_columns = [
        'study_date',           # Checkdate
        'modality',            # Examination method
        'body_part',           # Examination body part
        'institution',         # Medical institution
        'referring_physician', # Referring physician
        'study_description',   # Study description
        'series_description',  # Series description
        'slice_thickness',     # Slice thickness
        'kvp',                 # Peak kilovoltage
        'exposure_time',       # Exposure time
        'tube_current'         # Tube current
    ]
    
    print("可能的其他列包括:")
    for i, col in enumerate(possible_columns, 1):
        print(f"  {i:2d}. {col}")

if __name__ == "__main__":
    print("=== CTCheck数据Convert为CSV ===\n")
    
    # Execute conversion
    csv_file = create_csv_from_data()
    
    # Show other possible columns
    add_missing_columns()
    
    print(f"\n=== Convert完成 ===")
    print(f"生成的CSV文件: {csv_file}") 