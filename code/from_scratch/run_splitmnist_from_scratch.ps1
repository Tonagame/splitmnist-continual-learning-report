param(
    [ValidateSet("class", "domain", "task")]
    [string]$Scenario = "class",

    [int]$Iters = 2000,
    [int]$Batch = 128,
    [int]$AccN = 1024,
    [int]$EvalEvery = 100,
    [int]$MemoryPerClass = 100,
    [int]$Seed = 0,

    [string]$Python = "E:\conda-envs\continual\python.exe",
    [string]$Root = "E:\Codex\continual-learning-setup\splitmnist-continual-learning-report",
    [string]$DataDir = "E:\Codex\continual-learning-setup\data"
)

$ErrorActionPreference = "Continue"

$Script = Join-Path $Root "code\from_scratch\splitmnist_cl.py"
$Out = Join-Path $Root "results_from_scratch\splitmnist_${Scenario}_${Iters}"
$Logs = Join-Path $Out "logs"
New-Item -ItemType Directory -Force -Path $Out, $Logs | Out-Null

if ($Scenario -eq "task") {
    $Methods = @(
        "none",
        "ewc",
        "lwf",
        "agem",
        "separate",
        "lsr-lite",
        "lsr-lite-fourier",
        "lsr-lite-asw",
        "lsr-lite-fourier-asw",
        "joint"
    )
} else {
    $Methods = @(
        "none",
        "ewc",
        "lwf",
        "agem",
        "gen-classifier",
        "lsr-lite",
        "lsr-lite-fourier",
        "lsr-lite-asw",
        "lsr-lite-fourier-asw",
        "joint"
    )
}

$StatusPath = Join-Path $Out "run_status.csv"
"method,status,exit_code,log_file" | Set-Content -LiteralPath $StatusPath -Encoding UTF8

foreach ($Method in $Methods) {
    $SafeMethod = $Method.Replace("-", "_")
    $LogFile = Join-Path $Logs "${Scenario}_${Method}.log"
    $RunId = "${Scenario}_${SafeMethod}_seed${Seed}"

    $Args = @(
        $Script,
        "--method", $Method,
        "--scenario", $Scenario,
        "--contexts", "5",
        "--iters", "$Iters",
        "--batch", "$Batch",
        "--acc-n", "$AccN",
        "--eval-every", "$EvalEvery",
        "--memory-per-class", "$MemoryPerClass",
        "--seed", "$Seed",
        "--data-dir", $DataDir,
        "--results-dir", $Out,
        "--run-id", $RunId
    )

    "START $Method $(Get-Date -Format s)" | Out-File -LiteralPath $LogFile -Encoding UTF8
    & $Python @Args *>> $LogFile
    $ExitCode = $LASTEXITCODE
    $Status = if ($ExitCode -eq 0) { "success" } else { "failed" }
    "`"$Method`",$Status,$ExitCode,`"$LogFile`"" | Add-Content -LiteralPath $StatusPath -Encoding UTF8
}

Write-Host "Saved from-scratch results to $Out"
