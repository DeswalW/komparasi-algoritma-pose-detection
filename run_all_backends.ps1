param(
	[string]$WorkspaceRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path),
	[string[]]$Basename = @(),
	[string[]]$Backend = @("all"),
	[string]$VideoDir = "Video_Uji",
	[string]$GtDir = "Ground_Truth_Keypoint",
	[string]$OutputRoot = "results",
	[string]$AlphaPoseCfg = "",
	[string]$AlphaPoseCheckpoint = "",
	[string]$AlphaPoseDetector = "yolo",
	[string]$EfficientPoseCfg = "",
	[string]$EfficientPoseCheckpoint = "",
	[switch]$StopOnError
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
	param([string]$Base, [string]$PathValue)

	if ([string]::IsNullOrWhiteSpace($PathValue)) {
		return $PathValue
	}
	if ([System.IO.Path]::IsPathRooted($PathValue)) {
		return $PathValue
	}
	return (Join-Path $Base $PathValue)
}

function First-ExistingPath {
	param([string[]]$Candidates)

	foreach ($c in $Candidates) {
		if (-not [string]::IsNullOrWhiteSpace($c) -and (Test-Path $c -PathType Leaf)) {
			return $c
		}
	}
	return $null
}

function First-ExistingFileByPattern {
	param(
		[string]$BaseDir,
		[string[]]$Patterns
	)

	if (-not (Test-Path $BaseDir -PathType Container)) {
		return $null
	}

	foreach ($p in $Patterns) {
		$match = Get-ChildItem -Path $BaseDir -File -Filter $p -ErrorAction SilentlyContinue |
			Sort-Object Length -Descending |
			Select-Object -First 1
		if ($match) {
			return $match.FullName
		}
	}
	return $null
}

$WorkspaceRoot = Resolve-AbsolutePath -Base (Get-Location).Path -PathValue $WorkspaceRoot
$VideoDirAbs = Resolve-AbsolutePath -Base $WorkspaceRoot -PathValue $VideoDir
$GtDirAbs = Resolve-AbsolutePath -Base $WorkspaceRoot -PathValue $GtDir
$OutputRootAbs = Resolve-AbsolutePath -Base $WorkspaceRoot -PathValue $OutputRoot

if (-not (Test-Path $WorkspaceRoot -PathType Container)) {
	throw "Workspace root not found: $WorkspaceRoot"
}
if (-not (Test-Path $VideoDirAbs -PathType Container)) {
	throw "Video directory not found: $VideoDirAbs"
}
if (-not (Test-Path $GtDirAbs -PathType Container)) {
	throw "GT directory not found: $GtDirAbs"
}

$pyMediaPipe = Join-Path $WorkspaceRoot "MediaPipe Pose\.venv\Scripts\python.exe"
$pyAlphaPose = Join-Path $WorkspaceRoot "venvAlphapose\Scripts\python.exe"
$pyOpenPose = Join-Path $WorkspaceRoot "venvOpenPose\Scripts\python.exe"
$pyHRNet = Join-Path $WorkspaceRoot "venvHRNet\Scripts\python.exe"
$pyRoot = Join-Path $WorkspaceRoot ".venv\Scripts\python.exe"

$alphaCfgAuto = First-ExistingPath @(
	(Join-Path $WorkspaceRoot "AlphaPose\configs\coco\resnet\256x192_res50_lr1e-3_1x-simple.yaml"),
	(Join-Path $WorkspaceRoot "AlphaPose\configs\coco\resnet\256x192_res50_lr1e-3_1x.yaml"),
	(Join-Path $WorkspaceRoot "AlphaPose\configs\halpe_26\resnet\256x192_res50_lr1e-3_1x.yaml")
)
$alphaCkptAuto = First-ExistingFileByPattern -BaseDir (Join-Path $WorkspaceRoot "AlphaPose\pretrained_models") -Patterns @("pose_resnet_50_256x192*.pth", "fast_res50_256x192*.pth", "*.pth")
$effCfgAuto = First-ExistingPath @(
	(Join-Path $WorkspaceRoot "EfficientPose-master\experiments\coco\efficientpose\nasnet_192x256_adam_lr1e-3_efficientpose-c.yaml"),
	(Join-Path $WorkspaceRoot "EfficientPose-master\experiments\coco\efficientpose\nasnet_192x256_adam_lr1e-3_efficientpose-b.yaml"),
	(Join-Path $WorkspaceRoot "EfficientPose-master\experiments\coco\efficientpose\nasnet_192x256_adam_lr1e-3_efficientpose-a.yaml")
)
$effCkptAuto = First-ExistingFileByPattern -BaseDir (Join-Path $WorkspaceRoot "EfficientPose-master\model_checkpoint") -Patterns @("final_state.pth", "model_best.pth", "*.pth")

# Environment mapping per backend
$backendPython = @{
	"mediapipe"    = First-ExistingPath @($pyMediaPipe)
	"blazepose"    = First-ExistingPath @($pyMediaPipe)
	"alphapose"    = First-ExistingPath @($pyAlphaPose)
	"openpose"     = First-ExistingPath @($pyOpenPose)
	"hrnet"        = First-ExistingPath @($pyHRNet)
	"yolopose"     = First-ExistingPath @($pyHRNet)
	"efficientpose"= First-ExistingPath @($pyHRNet)
	# Sesuai preferensi Anda: PoseNet utamakan venvHRNet
	"posenet"      = First-ExistingPath @($pyHRNet, $pyRoot)
	"movenet"      = First-ExistingPath @($pyRoot, $pyHRNet)
}

$allBackends = @(
	"mediapipe",
	"blazepose",
	"movenet",
	"posenet",
	"alphapose",
	"openpose",
	"hrnet",
	"yolopose",
	"efficientpose"
)

$requested = @($Backend | ForEach-Object { $_.ToLowerInvariant() })
if ($requested -contains "all") {
	$backends = $allBackends
}
else {
	$backends = @()
	foreach ($b in $requested) {
		if ($allBackends -contains $b) {
			$backends += $b
		}
		else {
			Write-Warning "Unknown backend ignored: $b"
		}
	}
	if ($backends.Count -eq 0) {
		throw "No valid backend selected. Valid values: $($allBackends -join ', ')"
	}
}

Push-Location $WorkspaceRoot
try {
	$results = @()

	Write-Host "Workspace   : $WorkspaceRoot"
	Write-Host "Backend(s)  : $($backends -join ', ')"
	Write-Host "Video dir   : $VideoDirAbs"
	Write-Host "GT dir      : $GtDirAbs"
	Write-Host "Output root : $OutputRootAbs"
	if ($Basename.Count -gt 0) {
		Write-Host "Basename(s) : $($Basename -join ', ')"
	}
	Write-Host ""

	foreach ($backend in $backends) {
		$pythonExe = $backendPython[$backend]
		if (-not $pythonExe) {
			Write-Warning "[$backend] Skip: python env tidak ditemukan"
			$results += [PSCustomObject]@{ Backend = $backend; Status = "skipped"; ExitCode = ""; Note = "python env not found" }
			continue
		}

		$args = @(
			"-m", "keypoint_evaluator",
			"--backend", $backend,
			"--video-dir", $VideoDirAbs,
			"--gt-dir", $GtDirAbs,
			"--output-root", $OutputRootAbs
		)

		if ($Basename.Count -gt 0) {
			$args += "--basename"
			$args += $Basename
		}

		if ($backend -eq "alphapose") {
			$cfgPath = if ([string]::IsNullOrWhiteSpace($AlphaPoseCfg)) { $alphaCfgAuto } else { Resolve-AbsolutePath -Base $WorkspaceRoot -PathValue $AlphaPoseCfg }
			$ckptPath = if ([string]::IsNullOrWhiteSpace($AlphaPoseCheckpoint)) { $alphaCkptAuto } else { Resolve-AbsolutePath -Base $WorkspaceRoot -PathValue $AlphaPoseCheckpoint }

			if ([string]::IsNullOrWhiteSpace($cfgPath) -or -not (Test-Path $cfgPath -PathType Leaf)) {
				Write-Warning "[alphapose] Skip: config YAML tidak ditemukan. Gunakan -AlphaPoseCfg"
				$results += [PSCustomObject]@{ Backend = $backend; Status = "skipped"; ExitCode = ""; Note = "missing AlphaPose cfg" }
				continue
			}
			if ([string]::IsNullOrWhiteSpace($ckptPath) -or -not (Test-Path $ckptPath -PathType Leaf)) {
				Write-Warning "[alphapose] Skip: checkpoint .pth tidak ditemukan. Gunakan -AlphaPoseCheckpoint"
				$results += [PSCustomObject]@{ Backend = $backend; Status = "skipped"; ExitCode = ""; Note = "missing AlphaPose checkpoint" }
				continue
			}

			$args += "--alphapose-cfg"
			$args += $cfgPath
			$args += "--alphapose-checkpoint"
			$args += $ckptPath
			$args += "--alphapose-detector"
			$args += $AlphaPoseDetector
		}

		if ($backend -eq "efficientpose") {
			$cfgAbs = if ([string]::IsNullOrWhiteSpace($EfficientPoseCfg)) { $effCfgAuto } else { Resolve-AbsolutePath -Base $WorkspaceRoot -PathValue $EfficientPoseCfg }
			$ckptAbs = if ([string]::IsNullOrWhiteSpace($EfficientPoseCheckpoint)) { $effCkptAuto } else { Resolve-AbsolutePath -Base $WorkspaceRoot -PathValue $EfficientPoseCheckpoint }
			if (-not (Test-Path $cfgAbs -PathType Leaf)) {
				Write-Warning "[efficientpose] Skip: config YAML tidak ditemukan. Isi -EfficientPoseCfg"
				$results += [PSCustomObject]@{ Backend = $backend; Status = "skipped"; ExitCode = ""; Note = "config not found" }
				continue
			}
			if (-not (Test-Path $ckptAbs -PathType Leaf)) {
				Write-Warning "[efficientpose] Skip: checkpoint tidak ditemukan. Isi -EfficientPoseCheckpoint"
				$results += [PSCustomObject]@{ Backend = $backend; Status = "skipped"; ExitCode = ""; Note = "checkpoint not found" }
				continue
			}
			$args += "--efficientpose-cfg"
			$args += $cfgAbs
			$args += "--efficientpose-checkpoint"
			$args += $ckptAbs
		}

		Write-Host "==== RUN $backend ====" -ForegroundColor Cyan
		Write-Host "Python: $pythonExe"
		& $pythonExe @args
		$exitCode = $LASTEXITCODE

		if ($exitCode -eq 0) {
			Write-Host "[$backend] DONE" -ForegroundColor Green
			$results += [PSCustomObject]@{ Backend = $backend; Status = "ok"; ExitCode = 0; Note = "" }
		}
		else {
			Write-Warning "[$backend] FAILED (exit code: $exitCode)"
			$results += [PSCustomObject]@{ Backend = $backend; Status = "failed"; ExitCode = $exitCode; Note = "" }
			if ($StopOnError) {
				throw "StopOnError aktif. Backend gagal: $backend"
			}
		}

		Write-Host ""
	}

	Write-Host "===== SUMMARY =====" -ForegroundColor Yellow
	$results | Format-Table -AutoSize
}
finally {
	Pop-Location
}

