@echo off
cd /d E:\Morpho-FM
C:\ProgramData\anaconda3\python.exe -u scripts\run_prostate_protocol_a_benchmark_local.py --methods HisToGene --max-runs 1 >> results\benchmark_audit\protocol_a_runner_process_logs\smoke_hisToGene.log 2>&1
