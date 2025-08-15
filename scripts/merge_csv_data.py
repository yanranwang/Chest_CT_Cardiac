#!/usr/bin/env python3
"""
MergeCT和Echo数据的CSV文件
将包含basename/folder的CT数据与包含标签的Echo数据Merge
"""

import pandas as pd
import os

def merge_csv_data():
    """MergeCSV数据"""
    
    # filepath
    ct_csv_path = "filtered_echo_chestCT_data_filtered_chest_data.csv"
    echo_csv_path = "filtered_echo_chestCT_data_filtered_echo_data.csv"
    output_path = "merged_ct_echo_data.csv"
    
    print("🔍 ReadCSV文件...")
    
    # ReadCTdata (include basename, folder)
    if not os.path.exists(ct_csv_path):
        print(f"❌ CT CSV file does not exist: {ct_csv_path}")
        return
    
    ct_df = pd.read_csv(ct_csv_path)
    print(f"📊 CT数据: {len(ct_df)} 行, 列: {list(ct_df.columns)[:5]}...")
    
    # ReadEchodata (include lvef, AS_maybe)
    if not os.path.exists(echo_csv_path):
        print(f"❌ Echo CSV file does not exist: {echo_csv_path}")
        return
    
    echo_df = pd.read_csv(echo_csv_path)
    print(f"📊 Echo数据: {len(echo_df)} 行, 包含标签列: {'lvef' in echo_df.columns}, {'AS_maybe' in echo_df.columns}")
    
    # CheckMergekey
    if 'mrn' in ct_df.columns and 'mrn' in echo_df.columns:
        merge_key = 'mrn'
    elif 'accession_number' in ct_df.columns and 'accession_number' in echo_df.columns:
        merge_key = 'accession_number'
    else:
        print("❌ 找不到合适的Merge键 (mrn 或 accession_number)")
        return
    
    print(f"🔗 使用 '{merge_key}' 作为Merge键")
    
    # Mergedata
    print("🔄 Merge数据...")
    merged_df = pd.merge(ct_df, echo_df, on=merge_key, how='inner', suffixes=('_ct', '_echo'))
    
    print(f"✅ Merge完成: {len(merged_df)} 行")
    
    # Check必需的列
    required_columns = ['basename', 'folder', 'lvef', 'AS_maybe']
    missing_columns = [col for col in required_columns if col not in merged_df.columns]
    
    if missing_columns:
        print(f"❌ Merge后仍缺少列: {missing_columns}")
        print(f"可用列: {list(merged_df.columns)}")
        return
    
    # SaveMergeresults
    print(f"💾 Save到: {output_path}")
    merged_df.to_csv(output_path, index=False)
    
    # showstatisticsinfo
    print("\n📈 Merge数据Statistics:")
    print(f"总行数: {len(merged_df)}")
    print(f"有效lvef值: {merged_df['lvef'].notna().sum()}")
    print(f"有效AS_maybe值: {merged_df['AS_maybe'].notna().sum()}")
    
    # showlabels分布
    if 'AS_maybe' in merged_df.columns:
        print("\nAS_maybe分布:")
        print(merged_df['AS_maybe'].value_counts().sort_index())
    
    print(f"\n✅ Merge完成! 输出文件: {output_path}")
    return output_path

if __name__ == "__main__":
    merge_csv_data() 