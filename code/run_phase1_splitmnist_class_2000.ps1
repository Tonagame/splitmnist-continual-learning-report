$ErrorActionPreference = "Continue"
$env:TEMP = "E:\Temp"
$env:TMP = "E:\Temp"

$Root = "E:\Codex\continual-learning-setup\continual-learning"
$Python = "E:\conda-envs\continual\python.exe"
$Out = Join-Path $Root "results\splitmnist_class_2000"
$Logs = Join-Path $Out "logs"
$Plots = Join-Path $Out "plots"
$Models = Join-Path $Root "results\models"
$LearningCurve = Join-Path $Out "learning_curve.csv"
$Status = Join-Path $Out "run_status.csv"

New-Item -ItemType Directory -Force -Path $Out, $Logs, $Plots, $Models, "E:\Temp" | Out-Null
Set-Location $Root

if (Test-Path -LiteralPath $LearningCurve) {
    Remove-Item -LiteralPath $LearningCurve -Force
}
"method,status,exit_code,start_time,end_time,runtime_seconds,command,log_file" | Set-Content -LiteralPath $Status -Encoding UTF8

function Run-PhaseMethod {
    param(
        [string]$Method,
        [string]$Command,
        [string]$LogFile
    )
    $start = Get-Date
    "===== START $Method $start =====" | Out-File -LiteralPath $LogFile -Encoding UTF8
    $script = @"
`$env:TEMP='E:\Temp'
`$env:TMP='E:\Temp'
Set-Location '$Root'
$Command
"@
    powershell -NoProfile -ExecutionPolicy Bypass -Command $script *>> $LogFile
    $exitCode = $LASTEXITCODE
    $end = Get-Date
    $runtime = [int](New-TimeSpan -Start $start -End $end).TotalSeconds
    $status = if ($exitCode -eq 0) { "success" } else { "failed" }
    $escapedCommand = $Command.Replace('"', '""')
    $escapedLog = $LogFile.Replace('"', '""')
    "`"$Method`",$status,$exitCode,$($start.ToString('s')),$($end.ToString('s')),$runtime,`"$escapedCommand`",`"$escapedLog`"" |
        Add-Content -LiteralPath $Status -Encoding UTF8
}

$CommonMain = "--experiment=splitMNIST --scenario=class --contexts=5 --iters=2000 --batch=128 --acc-n=1024 --acc-log=10 --time --no-save --budget=100 --results-dir=`"$Out`" --plot-dir=`"$Plots`" --model-dir=`"$Models`" --eval-history-file=`"$LearningCurve`""
$CommonLSR = "--experiment=splitMNIST --scenario=class --contexts=5 --iters=2000 --batch=128 --acc-n=1024 --budget=100 --eval-every=10 --results-dir=`"$Out`" --model-dir=`"$Models`" --eval-history-file=`"$LearningCurve`""

$Runs = @(
    @{
        Method = "None"
        Command = "& `"$Python`" main.py $CommonMain --eval-history-method=`"None`""
        Log = Join-Path $Logs "splitMNIST_class_2000_none.log"
    },
    @{
        Method = "EWC"
        Command = "& `"$Python`" main.py $CommonMain --ewc --eval-history-method=`"EWC`""
        Log = Join-Path $Logs "splitMNIST_class_2000_ewc.log"
    },
    @{
        Method = "LwF"
        Command = "& `"$Python`" main.py $CommonMain --lwf --eval-history-method=`"LwF`""
        Log = Join-Path $Logs "splitMNIST_class_2000_lwf.log"
    },
    @{
        Method = "A-GEM"
        Command = "& `"$Python`" main.py $CommonMain --agem --eval-history-method=`"A-GEM`""
        Log = Join-Path $Logs "splitMNIST_class_2000_agem.log"
    },
    @{
        Method = "Generative Classifier"
        Command = "& `"$Python`" main.py $CommonMain --gen-classifier --eval-history-method=`"Generative Classifier`""
        Log = Join-Path $Logs "splitMNIST_class_2000_generative_classifier.log"
    },
    @{
        Method = "LSR-lite"
        Command = "& `"$Python`" train_lsr_lite.py $CommonLSR --eval-history-method=`"LSR-lite`""
        Log = Join-Path $Logs "splitMNIST_class_2000_lsr_lite.log"
    },
    @{
        Method = "LSR-lite + Fourier"
        Command = "& `"$Python`" train_lsr_lite.py $CommonLSR --fourier --eval-history-method=`"LSR-lite + Fourier`""
        Log = Join-Path $Logs "splitMNIST_class_2000_lsr_lite_fourier.log"
    },
    @{
        Method = "LSR-lite + ASW"
        Command = "& `"$Python`" train_lsr_lite.py $CommonLSR --asw --distill-weight=1.0 --feature-weight=0.5 --temp=2 --eval-history-method=`"LSR-lite + ASW`""
        Log = Join-Path $Logs "splitMNIST_class_2000_lsr_lite_asw.log"
    },
    @{
        Method = "LSR-lite + Fourier + ASW"
        Command = "& `"$Python`" train_lsr_lite.py $CommonLSR --fourier --asw --distill-weight=1.0 --feature-weight=0.5 --fourier-weight=0.05 --temp=2 --eval-history-method=`"LSR-lite + Fourier + ASW`""
        Log = Join-Path $Logs "splitMNIST_class_2000_lsr_lite_fourier_asw.log"
    },
    @{
        Method = "Joint"
        Command = "& `"$Python`" main.py --experiment=splitMNIST --scenario=class --contexts=5 --iters=10000 --batch=128 --acc-n=1024 --acc-log=10 --time --no-save --budget=100 --joint --results-dir=`"$Out`" --plot-dir=`"$Plots`" --model-dir=`"$Models`" --eval-history-file=`"$LearningCurve`" --eval-history-method=`"Joint`""
        Log = Join-Path $Logs "splitMNIST_class_2000_joint.log"
    }
)

foreach ($run in $Runs) {
    Run-PhaseMethod -Method $run.Method -Command $run.Command -LogFile $run.Log
}

& $Python phase1_summarize.py --results-dir "$Out"
