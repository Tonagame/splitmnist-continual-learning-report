$ErrorActionPreference = "Continue"
$env:TEMP = "E:\Temp"
$env:TMP = "E:\Temp"

$Root = "E:\Codex\continual-learning-setup\continual-learning"
$Python = "E:\conda-envs\continual\python.exe"
$Out = Join-Path $Root "results\splitmnist_task_2000"
$Logs = Join-Path $Out "logs"
$Plots = Join-Path $Out "plots"
$Models = Join-Path $Root "results\models"
$LearningCurve = Join-Path $Out "learning_curve.csv"
$Status = Join-Path $Out "run_status.csv"

New-Item -ItemType Directory -Force -Path $Out, $Logs, $Plots, $Models, "E:\Temp" | Out-Null
Set-Location $Root
if (Test-Path -LiteralPath $LearningCurve) { Remove-Item -LiteralPath $LearningCurve -Force }
"method,status,exit_code,start_time,end_time,runtime_seconds,command,log_file" | Set-Content -LiteralPath $Status -Encoding UTF8

function Run-PhaseMethod {
    param(
        [string]$Method,
        [string]$Script,
        [string[]]$ArgsList,
        [string]$LogFile
    )
    $start = Get-Date
    "===== START $Method $($start.ToString('s')) =====" | Out-File -LiteralPath $LogFile -Encoding UTF8
    & $Python $Script @ArgsList *>> $LogFile
    $exitCode = $LASTEXITCODE
    $end = Get-Date
    $runtime = [int](New-TimeSpan -Start $start -End $end).TotalSeconds
    $state = if ($exitCode -eq 0) { "success" } else { "failed" }
    $command = ('& "{0}" {1} {2}' -f $Python, $Script, ($ArgsList -join ' '))
    $row = '"{0}",{1},{2},{3},{4},{5},"{6}","{7}"' -f (
        $Method, $state, $exitCode, $start.ToString('s'), $end.ToString('s'), $runtime,
        $command.Replace('"', '""'), $LogFile.Replace('"', '""')
    )
    Add-Content -LiteralPath $Status -Encoding UTF8 -Value $row
}

$CommonMain = @(
    "--experiment=splitMNIST", "--scenario=task", "--contexts=5", "--iters=2000",
    "--batch=128", "--acc-n=1024", "--acc-log=100", "--time", "--no-save",
    "--budget=100", "--results-dir=$Out", "--plot-dir=$Plots", "--model-dir=$Models",
    "--eval-history-file=$LearningCurve"
)
$CommonLSR = @(
    "--experiment=splitMNIST", "--scenario=task", "--contexts=5", "--iters=2000",
    "--batch=128", "--acc-n=1024", "--budget=100", "--eval-every=100",
    "--results-dir=$Out", "--model-dir=$Models", "--eval-history-file=$LearningCurve"
)

$Runs = @(
    @{ Method="None"; Script="main.py"; Args=$CommonMain + @("--eval-history-method=None"); Log=Join-Path $Logs "splitMNIST_task_2000_none.log" },
    @{ Method="EWC"; Script="main.py"; Args=$CommonMain + @("--ewc", "--eval-history-method=EWC"); Log=Join-Path $Logs "splitMNIST_task_2000_ewc.log" },
    @{ Method="LwF"; Script="main.py"; Args=$CommonMain + @("--lwf", "--eval-history-method=LwF"); Log=Join-Path $Logs "splitMNIST_task_2000_lwf.log" },
    @{ Method="A-GEM"; Script="main.py"; Args=$CommonMain + @("--agem", "--eval-history-method=A-GEM"); Log=Join-Path $Logs "splitMNIST_task_2000_agem.log" },
    @{ Method="Separate Networks"; Script="main.py"; Args=$CommonMain + @("--separate-networks", "--eval-history-method=Separate Networks"); Log=Join-Path $Logs "splitMNIST_task_2000_separate_networks.log" },
    @{ Method="LSR-lite"; Script="train_lsr_lite.py"; Args=$CommonLSR + @("--eval-history-method=LSR-lite"); Log=Join-Path $Logs "splitMNIST_task_2000_lsr_lite.log" },
    @{ Method="LSR-lite + Fourier"; Script="train_lsr_lite.py"; Args=$CommonLSR + @("--fourier", "--eval-history-method=LSR-lite + Fourier"); Log=Join-Path $Logs "splitMNIST_task_2000_lsr_lite_fourier.log" },
    @{ Method="LSR-lite + ASW"; Script="train_lsr_lite.py"; Args=$CommonLSR + @("--asw", "--distill-weight=1.0", "--feature-weight=0.5", "--temp=2", "--eval-history-method=LSR-lite + ASW"); Log=Join-Path $Logs "splitMNIST_task_2000_lsr_lite_asw.log" },
    @{ Method="LSR-lite + Fourier + ASW"; Script="train_lsr_lite.py"; Args=$CommonLSR + @("--fourier", "--asw", "--distill-weight=1.0", "--feature-weight=0.5", "--fourier-weight=0.05", "--temp=2", "--eval-history-method=LSR-lite + Fourier + ASW"); Log=Join-Path $Logs "splitMNIST_task_2000_lsr_lite_fourier_asw.log" },
    @{ Method="Joint"; Script="main.py"; Args=@(
        "--experiment=splitMNIST", "--scenario=task", "--contexts=5", "--iters=10000",
        "--batch=128", "--acc-n=1024", "--acc-log=100", "--time", "--no-save",
        "--budget=100", "--joint", "--results-dir=$Out", "--plot-dir=$Plots",
        "--model-dir=$Models", "--eval-history-file=$LearningCurve", "--eval-history-method=Joint"
    ); Log=Join-Path $Logs "splitMNIST_task_2000_joint.log" }
)

foreach ($run in $Runs) {
    Run-PhaseMethod -Method $run.Method -Script $run.Script -ArgsList $run.Args -LogFile $run.Log
}

& $Python phase3_summarize.py --results-dir "$Out"
