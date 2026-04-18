import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

print("\n" + "="*60)
print(f" [{time.strftime('%H:%M:%S')}] 开始专属临时 TCDF 单跑任务 ")
print(" 注意：变量维度猛增至 480 维附近，TCDF 为逐节点并行训练模型，预计将耗费数小时！")
print("="*60 + "\n")

# 直接调用我们已经高度优化的 TCDF 脚本，跑双线
cmd = f'python "{os.path.join(parent_dir, "run_tcdf_space_time_dag.py")}" --line both'
exit_code = os.system(cmd)

if exit_code == 0:
    print("\n>>> TCDF 单独跑批任务圆满结束。跳过 NTS/CUTS+ 及集成投票流程！<<<")
    print(f">>> 结果已保存至: {os.path.join(parent_dir, '因果发现结果')} <<<")
else:
    print(f"\n>>> [错误] TCDF 任务异常退出 (Code: {exit_code}) <<<")
