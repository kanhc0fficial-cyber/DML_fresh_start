@echo off
chcp 65001 >nul
cd /d "C:\DML_fresh_start\数据预处理"
python annotate_variables.py "C:\DML_fresh_start\双重机器学习\结果\joint_causal_xin2\joint_causal_dml_xin2_fixed.csv" "操作节点"
pause
