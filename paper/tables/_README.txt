
此目录下表格为 scripts/export_latex_tables.py 自动生成，请勿手动编辑。

在论文中引用：
    \input{paper/tables/table1_main_three_methods}
    \input{paper/tables/table2_mbpp_es_mr_split}
    \input{paper/tables/table3_gsm8k_ablation}
    \input{paper/tables/table4_sensitivity}   % Appendix B: hyperparameter sensitivity
    \input{paper/tables/table5_crossmodel}    % Appendix C: cross-family transfer
    \input{paper/tables/table6_gating_controls}  % Appendix D: naive-gating controls
    \input{paper/tables/table7_pre_generation_scheduling}      % pre-generation scheduling
    \input{paper/tables/table7_pre_generation_scheduling_zh}   % 中文版

若需重生成：
    python3 scripts/export_latex_tables.py

table4 需要 results/sweep_v1/sweep_summary.json 覆盖全部 15 个网格点，否则自动跳过。
table6 需要 results/gating_sweep/ 下 6 个点 (anchor + random + 4×top-k) 全部跑完，否则自动跳过。
table7 需要 results/pre_generation_scheduling_20261117/gsm8k_train/summary.json 含 5 个条件
       与 4 组配对对比，否则自动跳过。
