#!/usr/bin/env python3
"""
Convert complete Excel file to CSV format

This script converts all sheets of the filtered_echo_chestCT_data.xls file to separate CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path

def convert_excel_to_csv(excel_file='filtered_echo_chestCT_data.xls'):
    """Convert all sheets of Excel file to CSV format"""
    
    print(f"Reading Excel file: {excel_file}")
    
    try:
        # First get all sheet names
        try:
            # Try different engines to read .xls files
            sheet_names = pd.ExcelFile(excel_file, engine='xlrd').sheet_names
            engine = 'xlrd'
            print("✅ Successfully read file using xlrd engine")
        except Exception as e1:
            print(f"xlrd engine failed, trying openpyxl: {e1}")
            try:
                sheet_names = pd.ExcelFile(excel_file, engine='openpyxl').sheet_names
                engine = 'openpyxl'
                print("✅ Successfully read file using openpyxl engine")
            except Exception as e2:
                print(f"openpyxl engine also failed, trying default engine: {e2}")
                sheet_names = pd.ExcelFile(excel_file).sheet_names
                engine = None
                print("✅ Successfully read file using default engine")
        
        print(f"\n📋 Found {len(sheet_names)} worksheets:")
        for i, sheet_name in enumerate(sheet_names, 1):
            print(f"   {i}. {sheet_name}")
        
        results = []
        
        # Process each sheet
        for i, sheet_name in enumerate(sheet_names):
            print(f"\n{'='*80}")
            print(f"📄 Processing worksheet {i+1}/{len(sheet_names)}: '{sheet_name}'")
            print('='*80)
            
            # Read current sheet
            if engine:
                df = pd.read_excel(excel_file, sheet_name=sheet_name, engine=engine)
            else:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Generate CSV filename
            base_name = Path(excel_file).stem
            csv_file = f"{base_name}_{sheet_name}.csv"
            # Clean special characters in filename
            csv_file = csv_file.replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_').replace('*', '_')
            
            # Show basic info for current sheet
            print(f"\n📊 Worksheet '{sheet_name}' info:")
            print(f"   - Rows: {len(df):,}")
            print(f"   - Columns: {len(df.columns)}")
            print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Show column names
            print(f"\n📋 Column names ({len(df.columns)} columns):")
            for j, col in enumerate(df.columns, 1):
                print(f"   {j:2d}. {col}")
            
            # Show data types
            print(f"\n🔍 Data types:")
            for col, dtype in df.dtypes.items():
                print(f"   {col:25}: {dtype}")
            
            # Show preview of first few rows
            print(f"\n📄 Preview of first 3 rows:")
            print("-" * 100)
            print(df.head(3))
            
            # Check for missing values
            missing_data = df.isnull().sum()
            if missing_data.any():
                print(f"\n⚠️  Missing values statistics:")
                for col, missing_count in missing_data.items():
                    if missing_count > 0:
                        percentage = (missing_count / len(df)) * 100
                        print(f"   {col:25}: {missing_count:,} ({percentage:.1f}%)")
            else:
                print(f"\n✅ No missing values")
            
            # Data quality check
            check_data_quality(df, sheet_name)
            
            # Save as CSV file
            print(f"\n💾 Saving as CSV file: {csv_file}")
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            # Check generated CSV file
            csv_size = Path(csv_file).stat().st_size / 1024 / 1024
            print(f"✅ CSV file saved successfully!")
            print(f"   - File name: {csv_file}")
            print(f"   - File size: {csv_size:.2f} MB")
            print(f"   - Rows: {len(df):,}")
            print(f"   - Columns: {len(df.columns)}")
            
            # Show CSV file preview
            print(f"\n📄 CSV file content preview (first 2 rows):")
            print("-" * 100)
            with open(csv_file, 'r', encoding='utf-8') as f:
                for j, line in enumerate(f):
                    if j < 3:  # Show header + first 2 data rows
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
        print(f"❌ Conversion failed: {e}")
        print("Please ensure:")
        print("1. Excel file exists and is readable")
        print("2. Required libraries are installed: pip install pandas xlrd openpyxl")
        return None

def check_data_quality(df, sheet_name):
    """Check data quality"""
    if df is None:
        return
    
    print(f"\n🔍 Data quality check for worksheet '{sheet_name}':")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"   Duplicate rows: {duplicates:,}")
    else:
        print(f"   ✅ No duplicate rows")
    
    # Check statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 Numeric column statistics ({len(numeric_cols)} columns):")
        for col in numeric_cols[:3]:  # Show only first 3 numeric columns
            if df[col].count() > 0:  # Ensure column is not all empty
                print(f"   {col:25}: Min={df[col].min():.2f}, Max={df[col].max():.2f}, Mean={df[col].mean():.2f}")
        if len(numeric_cols) > 3:
            print(f"   ... and {len(numeric_cols) - 3} more numeric columns")

if __name__ == "__main__":
    print("=== Excel file to CSV conversion (Multi-sheet support) ===\n")
    
    # Check if source file exists
    excel_file = 'filtered_echo_chestCT_data.xls'
    if not Path(excel_file).exists():
        print(f"❌ Error: File not found {excel_file}")
        print("Please ensure the file is in the current directory")
        exit(1)
    
    # Execute conversion
    results = convert_excel_to_csv(excel_file)
    
    if results:
        print(f"\n{'='*80}")
        print(f"=== Conversion Summary ===")
        print('='*80)
        print(f"Input file: {excel_file}")
        print(f"Processed {len(results)} worksheets:")
        
        total_rows = 0
        total_size = 0
        
        for i, result in enumerate(results, 1):
            csv_size = Path(result['csv_file']).stat().st_size / 1024 / 1024
            total_rows += result['rows']
            total_size += csv_size
            
            print(f"\n  {i}. Worksheet: {result['sheet_name']}")
            print(f"     └─ Output file: {result['csv_file']}")
            print(f"     └─ Data scale: {result['rows']:,} rows × {result['columns']} columns")
            print(f"     └─ File size: {csv_size:.2f} MB")
        
        print(f"\n📊 Total:")
        print(f"   - Total rows: {total_rows:,}")
        print(f"   - Total file size: {total_size:.2f} MB")
        print(f"\n✅ All worksheets successfully converted to CSV files, ready for data analysis!")
    else:
        print(f"\n=== Conversion Failed ===")
        print("Please check error messages and retry") 