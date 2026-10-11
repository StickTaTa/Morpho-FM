param(
    [int]$WaitPid = 0
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$AuditDir = Join-Path $Root "results\benchmark_audit"
$ProcessLogDir = Join-Path $AuditDir "runner_process_logs"
New-Item -ItemType Directory -Force -Path $ProcessLogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $ProcessLogDir "local_queue_$Stamp.log"

function Write-QueueLog {
    param([string]$Message)
    $Line = "[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')] $Message"
    Add-Content -LiteralPath $LogPath -Value $Line -Encoding UTF8
}

Set-Location -LiteralPath $Root
Write-QueueLog "queue started root=$Root wait_pid=$WaitPid"

if ($WaitPid -gt 0) {
    try {
        $Process = Get-Process -Id $WaitPid -ErrorAction Stop
        Write-QueueLog "waiting for existing prostate runner pid=$WaitPid"
        Wait-Process -Id $WaitPid
        Write-QueueLog "existing prostate runner pid=$WaitPid finished"
    } catch {
        Write-QueueLog "wait pid $WaitPid not active; continuing"
    }
}

Write-QueueLog "refreshing manifest before kidney queue"
& C:\ProgramData\anaconda3\python.exe scripts\build_benchmark_task_manifest.py *>> $LogPath

Write-QueueLog "starting kidney sequential benchmark queue"
& C:\ProgramData\anaconda3\python.exe scripts\run_benchmark_missing_sequential.py --cohort Kidney *>> $LogPath
$KidneyCode = $LASTEXITCODE
Write-QueueLog "kidney queue finished exit_code=$KidneyCode"

Write-QueueLog "refreshing manifest and unified metrics"
& C:\ProgramData\anaconda3\python.exe scripts\build_benchmark_task_manifest.py *>> $LogPath
& C:\ProgramData\anaconda3\python.exe scripts\evaluate_benchmark_predictions.py *>> $LogPath
Write-QueueLog "queue finished"

