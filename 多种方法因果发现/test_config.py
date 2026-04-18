import sys
sys.path.insert(0, '多种方法因果发现')
from causal_discovery_config import prepare_data

print('=== prepare_data xin1 ===')
df1, cols1, vs1, vg1 = prepare_data('xin1')
print(f'xin1: {df1.shape}, y_grade min={df1["y_grade"].min():.2f} max={df1["y_grade"].max():.2f}')

print('')
print('=== prepare_data xin2 ===')
df2, cols2, vs2, vg2 = prepare_data('xin2')
print(f'xin2: {df2.shape}, y_grade min={df2["y_grade"].min():.2f} max={df2["y_grade"].max():.2f}')

overlap = set(cols1).intersection(set(cols2))
print(f'xin1 vs xin2 变量集重叠数 (公用C变量): {len(overlap)}')
xin1_only = set(cols1) - set(cols2)
xin2_only = set(cols2) - set(cols1)
print(f'xin1专有变量(A组): {len(xin1_only)}')
print(f'xin2专有变量(B组): {len(xin2_only)}')
print('配置验证通过！')
