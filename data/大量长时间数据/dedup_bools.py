import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(r"c:\backup\doubleml\大量长时间数据")
    input_file = base_dir / "bool_variables_table.csv"
    output_file = base_dir / "unique_bool_variables.csv"

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # 去重
    unique_vars = df['Variable'].drop_duplicates().sort_values().reset_index(drop=True)
    
    # 保存结果
    unique_vars.to_csv(output_file, index=False, header=["Variable"], encoding='utf-8-sig')
    
    print(f"Original records: {len(df)}")
    print(f"Unique boolean variables: {len(unique_vars)}")
    print(f"Saved unique variables to {output_file}")

if __name__ == "__main__":
    main()
