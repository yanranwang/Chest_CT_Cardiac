#!/usr/bin/env python3
"""
将完整的Excel文件转换为CSV格式

这个脚本将filtered_echo_chestCT_data.xls文件的所有sheet分别转换为CSV文件
"""

import pandas as pd
import numpy as np
from pathlib import Path

def convert_excel_to_csv(excel_file='filtered_echo_chestCT_data.xls'):
    """将Excel文件的所有sheet转换为CSV格式"""
    
    print(f"正在读取Excel文件: {excel_file}")
    
    try:
        # 首先获取所有sheet名称
        try:
            # 尝试不同的引擎来读取.xls文件
            sheet_names = pd.ExcelFile(excel_file, engine='xlrd').sheet_names
            engine = 'xlrd'
            print("✅ 使用xlrd引擎成功读取文件")
        except Exception as e1:
            print(f"xlrd引擎失败，尝试openpyxl: {e1}")
            try:
                sheet_names = pd.ExcelFile(excel_file, engine='openpyxl').sheet_names
                engine = 'openpyxl'
                print("✅ 使用openpyxl引擎成功读取文件")
            except Exception as e2:
                print(f"openpyxl引擎也失败，尝试默认引擎: {e2}")
                sheet_names = pd.ExcelFile(excel_file).sheet_names
                engine = None
                print("✅ 使用默认引擎成功读取文件")
        
        print(f"\n📋 发现 {len(sheet_names)} 个工作表:")
        for i, sheet_name in enumerate(sheet_names, 1):
            print(f"   {i}. {sheet_name}")
        
        results = []
        
        # 遍历每个sheet
        for i, sheet_name in enumerate(sheet_names):
            print(f"\n{'='*80}")
            print(f"📄 正在处理工作表 {i+1}/{len(sheet_names)}: '{sheet_name}'")
            print('='*80)
            
            # 读取当前sheet
            if engine:
                df = pd.read_excel(excel_file, sheet_name=sheet_name, engine=engine)
            else:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # 生成CSV文件名
            base_name = Path(excel_file).stem
            csv_file = f"{base_name}_{sheet_name}.csv"
            # 清理文件名中的特殊字符
            csv_file = csv_file.replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_').replace('*', '_')
            
            # 显示当前sheet的基本信息
            print(f"\n📊 工作表 '{sheet_name}' 信息:")
            print(f"   - 行数: {len(df):,}")
            print(f"   - 列数: {len(df.columns)}")
            print(f"   - 内存使用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # 显示列名
            print(f"\n📋 列名 ({len(df.columns)} 列):")
            for j, col in enumerate(df.columns, 1):
                print(f"   {j:2d}. {col}")
            
            # 显示数据类型
            print(f"\n🔍 数据类型:")
            for col, dtype in df.dtypes.items():
                print(f"   {col:25}: {dtype}")
            
            # 显示前几行数据
            print(f"\n📄 前3行数据预览:")
            print("-" * 100)
            print(df.head(3))
            
            # 检查是否有缺失值
            missing_data = df.isnull().sum()
            if missing_data.any():
                print(f"\n⚠️  缺失值统计:")
                for col, missing_count in missing_data.items():
                    if missing_count > 0:
                        percentage = (missing_count / len(df)) * 100
                        print(f"   {col:25}: {missing_count:,} ({percentage:.1f}%)")
            else:
                print(f"\n✅ 无缺失值")
            
            # 数据质量检查
            check_data_quality(df, sheet_name)
            
            # 保存为CSV文件
            print(f"\n💾 正在保存为CSV文件: {csv_file}")
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            # 检查生成的CSV文件
            csv_size = Path(csv_file).stat().st_size / 1024 / 1024
            print(f"✅ CSV文件保存成功!")
            print(f"   - 文件名: {csv_file}")
            print(f"   - 文件大小: {csv_size:.2f} MB")
            print(f"   - 行数: {len(df):,}")
            print(f"   - 列数: {len(df.columns)}")
            
            # 显示CSV文件的前几行
            print(f"\n📄 CSV文件内容预览 (前2行):")
            print("-" * 100)
            with open(csv_file, 'r', encoding='utf-8') as f:
                for j, line in enumerate(f):
                    if j < 3:  # 显示标题行 + 前2行数据
                        if len(line.strip()) > 120:
                            print(line.strip()[:120] + "...")
                        else:
                            print(line.strip())
                    else:
                        break
            
            results.append({
                'sheet_name': sheet_name,
                'dataframe': df,
                'csv_file': csv_file,
                'rows': len(df),
                'columns': len(df.columns)
            })
        
        return results
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        print("请确保:")
        print("1. Excel文件存在且可读")
        print("2. 已安装必要的库: pip install pandas xlrd openpyxl")
        return None

def check_data_quality(df, sheet_name):
    """检查数据质量"""
    if df is None:
        return
    
    print(f"\n🔍 工作表 '{sheet_name}' 数据质量检查:")
    
    # 检查重复行
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"   重复行数: {duplicates:,}")
    else:
        print(f"   ✅ 无重复行")
    
    # 检查数值列的统计信息
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 数值列统计 ({len(numeric_cols)} 列):")
        for col in numeric_cols[:3]:  # 只显示前3个数值列
            if df[col].count() > 0:  # 确保列不全为空
                print(f"   {col:25}: 最小值={df[col].min():.2f}, 最大值={df[col].max():.2f}, 均值={df[col].mean():.2f}")
        if len(numeric_cols) > 3:
            print(f"   ... 还有 {len(numeric_cols) - 3} 个数值列")

if __name__ == "__main__":
    print("=== Excel文件转换为CSV (多工作表支持) ===\n")
    
    # 检查源文件是否存在
    excel_file = 'filtered_echo_chestCT_data.xls'
    if not Path(excel_file).exists():
        print(f"❌ 错误: 找不到文件 {excel_file}")
        print("请确保文件在当前目录中")
        exit(1)
    
    # 执行转换
    results = convert_excel_to_csv(excel_file)
    
    if results:
        print(f"\n{'='*80}")
        print(f"=== 转换完成总结 ===")
        print('='*80)
        print(f"输入文件: {excel_file}")
        print(f"共处理了 {len(results)} 个工作表:")
        
        total_rows = 0
        total_size = 0
        
        for i, result in enumerate(results, 1):
            csv_size = Path(result['csv_file']).stat().st_size / 1024 / 1024
            total_rows += result['rows']
            total_size += csv_size
            
            print(f"\n  {i}. 工作表: {result['sheet_name']}")
            print(f"     └─ 输出文件: {result['csv_file']}")
            print(f"     └─ 数据规模: {result['rows']:,} 行 × {result['columns']} 列")
            print(f"     └─ 文件大小: {csv_size:.2f} MB")
        
        print(f"\n📊 总计:")
        print(f"   - 总行数: {total_rows:,}")
        print(f"   - 总文件大小: {total_size:.2f} MB")
        print(f"\n✅ 所有工作表已成功转换为CSV文件，现在可以分别进行数据分析了！")
    else:
        print(f"\n=== 转换失败 ===")
        print("请检查错误信息并重试") 