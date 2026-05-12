param(
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
$Aggregate = Join-Path $Root "code\from_scratch\aggregate_from_scratch.py"
$OutRoot = Join-Path $Root "results_from_scratch\classic_no_lsr_$Iters"
$Logs = Join-Path $OutRoot "logs"
New-Item -ItemType Directory -Force -Path $OutRoot, $Logs | Out-Null

$StatusPath = Join-Path $OutRoot "run_status.csv"
"scenario,method,status,exit_code,start_time,end_time,runtime_seconds,log_file" | Set-Content -LiteralPath $StatusPath -Encoding UTF8

$Plan = @(
    @{
        Scenario = "class"
        Methods = @("none", "ewc", "lwf", "agem", "gen-classifier", "joint")
    },
    @{
        Scenario = "domain"
        Methods = @("none", "ewc", "lwf", "agem", "gen-classifier", "joint")
    },
    @{
        Scenario = "task"
        Methods = @("none", "ewc", "lwf", "agem", "gen-classifier", "separate", "joint")
    }
)

foreach ($ScenarioPlan in $Plan) {
    $Scenario = $ScenarioPlan.Scenario
    $ScenarioOut = Join-Path $OutRoot "splitmnist_${Scenario}_${Iters}"
    New-Item -ItemType Directory -Force -Path $ScenarioOut | Out-Null

    foreach ($Method in $ScenarioPlan.Methods) {
        $SafeMethod = $Method.Replace("-", "_")
        $LogFile = Join-Path $Logs "${Scenario}_${SafeMethod}.log"
        $RunId = "${Scenario}_${SafeMethod}_seed${Seed}"
        $Start = Get-Date

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
            "--results-dir", $ScenarioOut,
            "--run-id", $RunId
        )

        "START $Scenario $Method $($Start.ToString('s'))" | Out-File -LiteralPath $LogFile -Encoding UTF8
        & $Python @Args *>> $LogFile
        $ExitCode = $LASTEXITCODE
        $End = Get-Date
        $Runtime = [int](New-TimeSpan -Start $Start -End $End).TotalSeconds
        $Status = if ($ExitCode -eq 0) { "success" } else { "failed" }
        "`"$Scenario`",`"$Method`",$Status,$ExitCode,$($Start.ToString('s')),$($End.ToString('s')),$Runtime,`"$LogFile`"" |
            Add-Content -LiteralPath $StatusPath -Encoding UTF8
    }
}

& $Python $Aggregate --root $OutRoot --title "From-scratch Split MNIST classic methods, no LSR, $Iters iterations" *>> (Join-Path $Logs "aggregate.log")

Write-Host "Saved from-scratch classic/no-LSR results to $OutRoot"
