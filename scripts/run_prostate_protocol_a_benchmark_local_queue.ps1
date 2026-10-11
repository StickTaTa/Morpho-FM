param(
    [string[]]$Methods = @("HisToGene", "HiST", "sCellST", "iStar", "mclSTExp"),
    [int]$MaxRuns = 0
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$AuditDir = Join-Path $Root "results\benchmark_audit"
$ProcessLogDir = Join-Path $AuditDir "protocol_a_runner_process_logs"
New-Item -ItemType Directory -Force -Path $ProcessLogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $ProcessLogDir "prostate_protocol_a_local_queue_$Stamp.log"

function Write-QueueLog {
    param([string]$Message)
    $Line = "[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')] $Message"
    Add-Content -LiteralPath $LogPath -Value $Line -Encoding UTF8
}

Set-Location -LiteralPath $Root
Write-QueueLog "queue started root=$Root methods=$($Methods -join ',') max_runs=$MaxRuns"

Write-QueueLog "refreshing Protocol-A manifest"
& C:\ProgramData\anaconda3\python.exe scripts\build_prostate_protocol_a_benchmark_manifest.py *>> $LogPath

$ArgsList = @("scripts\run_prostate_protocol_a_benchmark_local.py", "--methods") + $Methods
if ($MaxRuns -gt 0) {
    $ArgsList += @("--max-runs", [string]$MaxRuns)
}

Write-QueueLog "starting Protocol-A local benchmark queue"
& C:\ProgramData\anaconda3\python.exe @ArgsList *>> $LogPath
$RunCode = $LASTEXITCODE
Write-QueueLog "Protocol-A queue finished exit_code=$RunCode"

Write-QueueLog "refreshing Protocol-A manifest after queue"
& C:\ProgramData\anaconda3\python.exe scripts\build_prostate_protocol_a_benchmark_manifest.py *>> $LogPath
Write-QueueLog "queue finished"
