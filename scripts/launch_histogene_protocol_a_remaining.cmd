@echo off
cd /d E:\Morpho-FM
if not exist results\benchmark_audit\runner_process_logs mkdir results\benchmark_audit\runner_process_logs
C:\ProgramData\anaconda3\python.exe -u scripts\run_prostate_protocol_a_benchmark_local.py --methods HisToGene --fold-tags test_INT26_val_INT27 test_INT27_val_INT28 test_INT28_val_INT25 >> results\benchmark_audit\runner_process_logs\histogene_protocol_a_remaining.log 2>&1
