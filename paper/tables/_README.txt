
此目录下表格为 scripts/export_latex_tables.py 自动生成，请勿手动编辑。

在论文中引用：
    \input{paper/tables/table1_main_three_methods}
    \input{paper/tables/table2_mbpp_es_mr_split}
    \input{paper/tables/table3_gsm8k_ablation}
    \input{paper/tables/table4_sensitivity}   % Appendix B: hyperparameter sensitivity
    \input{paper/tables/table5_crossmodel}    % Appendix C: cross-family transfer

若需重生成：
    python3 scripts/export_latex_tables.py

table4 需要 results/sweep_v1/sweep_summary.json 覆盖全部 15 个网格点，否则自动跳过。
