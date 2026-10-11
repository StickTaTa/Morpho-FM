@echo off
cd /d E:\Morpho-FM
if not exist results\benchmark_audit\runner_process_logs mkdir results\benchmark_audit\runner_process_logs
C:\ProgramData\anaconda3\python.exe -u scripts\run_prostate_protocol_a_benchmark_local.py --methods HisToGene --fold-tags test_INT25_val_INT26 >> results\benchmark_audit\runner_process_logs\histogene_protocol_a_test_fold.log 2>&1
