import pandas as pd
import numpy as np
import os

def ensemble_voting():
    results_dir = "因果发现结果"
    output_file = os.path.join(results_dir, "integrated_causal_voting_full_report.csv")
    
    # 待融合的文件及其得分列名
    algorithms = {
        "cuts_plus": ("cuts_plus_effects_on_y.csv", "Causal_Score"),
        "tcdf": ("tcdf_space_time_effects_on_y.csv", "Attention_To_Y"),
        "nts_notears": ("nts_notears_effects_on_y.csv", "Causal_Score")
    }
    
    W_k = 1.0 / len(algorithms)
    all_dfs = []
    
    print("正在加载并解析结果文件...")
    for name, (file, score_col) in algorithms.items():
        file_path = os.path.join(results_dir, file)
        if os.path.exists(file_path):
            try:
                # 强制编码以防乱码
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                if not df.empty:
                    # 去除列名空格
                    df.columns = [c.strip() for c in df.columns]
                    
                    if 'Source_Var' in df.columns and score_col in df.columns:
                        # 去除变量名空格
                        df['Source_Var'] = df['Source_Var'].astype(str).str.strip()
                        
                        # 归一化得分
                        s_min = df[score_col].min()
                        s_max = df[score_col].max()
                        if s_max > s_min:
                            df[f'{name}_norm_score'] = (df[score_col] - s_min) / (s_max - s_min)
                        else:
                            df[f'{name}_norm_score'] = 1.0
                        
                        # 只保留核心列用于合并
                        cols_to_keep = ['Source_Var', f'{name}_norm_score']
                        if 'Stage' in df.columns:
                            cols_to_keep.append('Stage')
                            
                        small_df = df[cols_to_keep].drop_duplicates(subset=['Source_Var'])
                        all_dfs.append(small_df)
                        print(f" - {name}: 成功加载 {len(small_df)} 条记录")
                    else:
                        print(f" - {name}: 缺失必要列 ('Source_Var' 或 '{score_col}')")
            except Exception as e:
                print(f" - {name}: 读取失败: {e}")
                
    if not all_dfs:
        print("错误：没有可合并的数据。")
        return

    # 使用外连接依次合并所有 DataFrame
    final_df = all_dfs[0]
    for i in range(1, len(all_dfs)):
        # 合并时保留 Stage 
        final_df = pd.merge(final_df, all_dfs[i], on='Source_Var', how='outer', suffixes=('', '_extra'))
        
        # 合并 Stage 信息（如果一边的 Stage 为空，则填充另一边的）
        if 'Stage_extra' in final_df.columns:
            final_df['Stage'] = final_df['Stage'].fillna(final_df['Stage_extra'])
            final_df.drop(columns=['Stage_extra'], inplace=True)

    # 计算总分和投票数
    score_cols = [c for c in final_df.columns if '_norm_score' in c]
    hit_cols = []
    
    final_df['Total_Weighted_Score'] = 0.0
    final_df['Vote_Count'] = 0
    
    for sc in score_cols:
        # 加权求和
        final_df['Total_Weighted_Score'] += final_df[sc].fillna(0) * W_k
        # 记录命中数
        hit_name = sc.replace('_norm_score', '_Hit')
        final_df[hit_name] = final_df[sc].notna().astype(int)
        final_df['Vote_Count'] += final_df[hit_name]

    # 排序
    final_df = final_df.sort_values(by=['Total_Weighted_Score', 'Vote_Count'], ascending=False)
    
    # 调整列顺序，让关键信息靠前
    cols = ['Source_Var', 'Total_Weighted_Score', 'Vote_Count', 'Stage'] + \
           [c for c in final_df.columns if c not in ['Source_Var', 'Total_Weighted_Score', 'Vote_Count', 'Stage']]
    final_df = final_df[cols]

    # 保存结果
    final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n合并完成！共计 {len(final_df)} 条记录。")
    print(f"完整报告保存至: {output_file}")
    print("\n前10名预览:")
    print(final_df.head(10).to_string(index=False))

if __name__ == "__main__":
    ensemble_voting()
