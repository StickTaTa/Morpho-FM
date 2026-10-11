@echo off
cd /d E:\Morpho-FM
if not exist results\benchmark_audit\runner_process_logs mkdir results\benchmark_audit\runner_process_logs
C:\ProgramData\anaconda3\python.exe -u scripts\run_benchmark_missing_sequential.py --cohort Prostate --protocol multi >> results\benchmark_audit\runner_process_logs\prostate_multi_queue.log 2>&1
