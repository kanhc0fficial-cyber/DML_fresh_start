import csv
import json
import os
import time
import urllib.request
import ssl
import re

# ==========================================
# ⚙️ 路径与配置
# ==========================================
INPUT_CSV = r"C:\DML_fresh_start\数据预处理\数据与处理结果-分阶段-去共线性后\non_collinear_representative_vars_annotated.csv"
OUTPUT_CSV = r"C:\DML_fresh_start\数据预处理\数据与处理结果-分阶段-去共线性后\non_collinear_representative_vars_operability.csv"
EXPERT_KNOWLEDGE_FILE = r"C:\DML_fresh_start\东鞍山烧结厂选矿专家知识.txt"

API_KEY = "sk-501b377173a1446ba01c86bc0b89b4b6"
API_BASE_URL = "https://api.deepseek.com/chat/completions" 
MODEL_NAME = "deepseek-chat" # 使用 chat 模型进行快速分类
BATCH_SIZE = 30

# 图片中的硬性可控设备清单（作为硬约束提示词）
CONTROLLABLE_DEVICES_KNOWLEDGE = """
根据工厂实际考察，仅以下设备/参数属于可操作变量 (operable)：
1. 破碎：皮带启停、皮带频率。
2. 球磨：补水阀门、摆式给矿、给矿量。
3. 磁选：励磁电压、励磁电流、尾矿阀门。
4. 塔磨：旋流器开闭、泵池补水阀门、旋流器补水阀门、泵池给矿（人工控）。
5. 浮选：分矿给矿量、加药阀门（药剂阀门）、连通阀门、浮选充气。

凡是不属于上述清单的，一律标记为 observable。
特别是：所有‘实际值’、‘液位’、‘浓度’、‘品位’、‘频率反馈’等反馈信号均为 observable。
只有‘设定值’、‘给定’、‘开度命令’等且属于上述设备范围的才可能是 operable。
"""

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"错误: 找不到输入文件 '{INPUT_CSV}'")
        return

    # 1. 加载专家知识
    expert_content = ""
    if os.path.exists(EXPERT_KNOWLEDGE_FILE):
        try:
            with open(EXPERT_KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
                expert_content = f.read().strip()
            print("已加载专家知识库。")
        except:
            print("警告：未能读取专家知识库，将仅依赖内置规则。")

    # 2. 读取输入 CSV
    data = []
    fieldnames = []
    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        data = list(reader)
    print(f"加载了 {len(data)} 个待分类变量。")

    for idx, row in enumerate(data):
        row['_RowIndex'] = idx

    output_results = []

    # 3. 分批处理
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]
        batch_json = json.dumps(batch, ensure_ascii=False)

        prompt = f"""你是一个工业过程控制专家。你的任务是将给出的工控变量分为 'operable' (可操作变量/控制量) 或 'observable' (仅可观测变量/状态量)。

【硬性判定准则】
{CONTROLLABLE_DEVICES_KNOWLEDGE}

【辅助专家知识】
{expert_content}

【判定逻辑】
1. 检查变量对应的设备是否在【可控设备清单】中。
2. 即使设备在清单中，也要检查变量的性质：
   - 如果是反馈信号（如：实际频率、实际电流、实际液位、实际值），必须设为 observable。
   - 如果是控制指令（如：给定频率、设定电流、阀门开度设定、给定值），且设备在清单中，设为 operable。
3. 任何不在清单中的设备（如：电机温度、轴承振动、浓密机压力等）统统设为 observable。

请严格返回 JSON 格式：
{{
    "results": [
        {{
            "_RowIndex": <对应输入的 _RowIndex>,
            "operability": "operable" 或 "observable",
            "reason": "简短说明理由"
        }}
    ]
}}

待处理变量数据：
{batch_json}
"""

        req_body = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a specialized industrial data classifier. Return ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        req = urllib.request.Request(
            API_BASE_URL, 
            data=json.dumps(req_body).encode('utf-8'), 
            headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
        )

        ssl_context = ssl._create_unverified_context()

        try:
            print(f"正在处理批次 {i//BATCH_SIZE + 1}...")
            with urllib.request.urlopen(req, timeout=60, context=ssl_context) as response:
                res_body = response.read().decode('utf-8')
                res_json = json.loads(res_body)
                reply = json.loads(res_json['choices'][0]['message']['content'].strip())
                
                for res in reply['results']:
                    original_row = data[res['_RowIndex']]
                    out_row = {k: v for k, v in original_row.items() if k != '_RowIndex'}
                    out_row['Operability'] = res['operability']
                    out_row['Operability_Reason'] = res['reason']
                    output_results.append(out_row)
        except Exception as e:
            print(f"批次处理失败: {e}")

    # 4. 写入输出
    if output_results:
        output_fieldnames = [f for f in fieldnames if f != '_RowIndex'] + ['Operability', 'Operability_Reason']
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(output_results)
        print(f"🎉 处理完成！结果保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
