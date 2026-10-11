@echo off
cd /d E:\Morpho-FM
C:\ProgramData\anaconda3\python.exe -u scripts\run_prostate_protocol_a_benchmark_local.py --methods HisToGene HiST sCellST iStar mclSTExp >> results\benchmark_audit\protocol_a_runner_process_logs\full_queue.log 2>&1
C:\ProgramData\anaconda3\python.exe scripts\build_prostate_protocol_a_benchmark_manifest.py >> results\benchmark_audit\protocol_a_runner_process_logs\full_queue.log 2>&1
