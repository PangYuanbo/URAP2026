$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

function Sanitize-FileName([string]$name) {
  $invalid = [System.IO.Path]::GetInvalidFileNameChars()
  foreach ($c in $invalid) { $name = $name.Replace($c, " ") }
  $name = $name -replace "\s+", " "
  $name = $name.Trim()
  if ($name.Length -gt 180) { $name = $name.Substring(0, 180).Trim() }
  return $name
}

function Download-Pdf([string]$title, [string]$url, [string]$fallbackUrl = "") {
  $base = Sanitize-FileName $title
  $out = Join-Path (Get-Location) ($base + ".pdf")
  $outShortcut = Join-Path (Get-Location) ($base + ".url")

  if (Test-Path $out) {
    $len = (Get-Item $out).Length
    if ($len -gt 1024) {
      Write-Host "SKIP (exists): $title" -ForegroundColor DarkGray
      return @{ title = $title; url = $url; file = (Split-Path $out -Leaf); status = "ok"; bytes = $len }
    }
    Remove-Item $out -Force -ErrorAction SilentlyContinue
  }

  Write-Host "Downloading: $title" -ForegroundColor Cyan
  & curl.exe -L --fail --retry 3 --retry-delay 2 -o $out $url
  if ($LASTEXITCODE -ne 0) {
    if (Test-Path $out) { Remove-Item $out -Force -ErrorAction SilentlyContinue }
    if ($fallbackUrl) {
      Write-Host "FAILED (saving .url): $title" -ForegroundColor Yellow
      @(
        "[InternetShortcut]"
        "URL=$fallbackUrl"
      ) | Out-File -Encoding ascii $outShortcut
      return @{
        title = $title
        url = $url
        file = (Split-Path $outShortcut -Leaf)
        status = "partial"
        note = "PDF not downloadable automatically; saved URL shortcut for manual access."
      }
    }

    Write-Host "FAILED: $title" -ForegroundColor Yellow
    return @{ title = $title; url = $url; file = ""; status = "failed" }
  }

  $len = (Get-Item $out).Length
  return @{ title = $title; url = $url; file = (Split-Path $out -Leaf); status = "ok"; bytes = $len }
}

function Print-Website-ToPdf([string]$title, [string]$url) {
  $base = Sanitize-FileName $title
  $out = Join-Path (Get-Location) ($base + ".pdf")

  if (Test-Path $out) {
    $len = (Get-Item $out).Length
    if ($len -gt 1024) {
      Write-Host "SKIP (exists): $title" -ForegroundColor DarkGray
      return @{ title = $title; url = $url; file = (Split-Path $out -Leaf); status = "ok"; bytes = $len; note = "printed from website" }
    }
    Remove-Item $out -Force -ErrorAction SilentlyContinue
  }

  $edgeCandidates = @(
    "$env:ProgramFiles (x86)\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe"
  )
  $msedge = $edgeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $msedge) {
    Write-Host "NO_EDGE: cannot print website to PDF ($title). Saving URL shortcut instead." -ForegroundColor Yellow
    $shortcut = Join-Path (Get-Location) ($base + ".url")
    @(
      "[InternetShortcut]"
      "URL=$url"
    ) | Out-File -Encoding ascii $shortcut
    return @{ title = $title; url = $url; file = (Split-Path $shortcut -Leaf); status = "partial"; note = "saved .url (no msedge.exe found)" }
  }

  Write-Host "Printing website to PDF: $title" -ForegroundColor Cyan
  & $msedge --headless --disable-gpu --run-all-compositor-stages-before-draw --virtual-time-budget=10000 --print-to-pdf="$out" "$url" | Out-Null
  if (-not (Test-Path $out)) {
    Write-Host "FAILED_PRINT: $title" -ForegroundColor Yellow
    return @{ title = $title; url = $url; file = ""; status = "failed"; note = "msedge print-to-pdf failed" }
  }
  $len = (Get-Item $out).Length
  return @{ title = $title; url = $url; file = (Split-Path $out -Leaf); status = "ok"; bytes = $len; note = "printed from website" }
}

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
  throw "curl.exe not found (needed for downloads)."
}

$items = @(
  # arXiv / CVF OpenAccess: direct PDFs
  @{ kind = "pdf"; title = "Visible and Clear: Finding Tiny Objects in Difference Map (SR-TOD)"; url = "https://arxiv.org/pdf/2405.11276.pdf"; fallback = "" },
  @{ kind = "pdf"; title = "Anti-UAV410: A Thermal Infrared Benchmark and Customized Scheme for Tracking Drones in the Wild"; url = "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10325629"; fallback = "https://doi.org/10.1109/TPAMI.2023.3335338" },
  @{ kind = "pdf"; title = "Event-based Tiny Object Detection: A Benchmark Dataset and Baseline"; url = "https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Event-based_Tiny_Object_Detection_A_Benchmark_Dataset_and_Baseline_ICCV_2025_paper.pdf"; fallback = "" },
  @{ kind = "pdf"; title = "SynDroneVision: A Synthetic Dataset for Image-Based Drone Detection"; url = "https://openaccess.thecvf.com/content/WACV2025/papers/Lenhard_SynDroneVision_A_Synthetic_Dataset_for_Image-Based_Drone_Detection_WACV_2025_paper.pdf"; fallback = "" },
  @{ kind = "pdf"; title = "DrIFT: Autonomous Drone Dataset with Integrated Real and Synthetic Data"; url = "https://openaccess.thecvf.com/content/WACV2025/papers/Dadboud_DrIFT_Autonomous_Drone_Dataset_with_Integrated_Real_and_Synthetic_Data_WACV_2025_paper.pdf"; fallback = "" },
  @{ kind = "pdf"; title = "TransVisDrone: Spatio-Temporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos"; url = "https://arxiv.org/pdf/2210.08423.pdf"; fallback = "" },
  @{ kind = "pdf"; title = "Evidential Detection and Tracking Collaboration: New Problem, Benchmark and Algorithm for Robust Anti-UAV System"; url = "https://arxiv.org/pdf/2306.15767.pdf"; fallback = "" },
  @{ kind = "pdf"; title = "Multi-Modal UAV Detection, Classification and Tracking Algorithm -- Technical Report for CVPR 2024 UG2 Challenge"; url = "https://arxiv.org/pdf/2405.16464.pdf"; fallback = "" },
  @{ kind = "pdf"; title = "SimD3: A Synthetic drone Dataset with Payload and Bird Distractor Modeling for Robust Detection"; url = "https://arxiv.org/pdf/2601.14742.pdf"; fallback = "" },
  @{ kind = "pdf"; title = "YOLOBirDrone: Dataset for Bird vs Drone Detection and Classification and a YOLO based enhanced learning architecture"; url = "https://arxiv.org/pdf/2601.08319.pdf"; fallback = "" },

  # Website (not a paper PDF): print-to-pdf if Edge exists
  @{ kind = "web"; title = "4th Anti-UAV Challenge"; url = "https://anti-uav.github.io/" }
)

$results = @()
foreach ($it in $items) {
  if ($it.kind -eq "pdf") {
    $results += Download-Pdf -title $it.title -url $it.url -fallbackUrl $it.fallback
  } elseif ($it.kind -eq "web") {
    $results += Print-Website-ToPdf -title $it.title -url $it.url
  } else {
    $results += @{ title = $it.title; url = $it.url; file = ""; status = "skipped"; note = "unknown kind: $($it.kind)" }
  }
}

$indexPath = Join-Path (Get-Location) "index.json"
$results | ConvertTo-Json -Depth 4 | Out-File -Encoding utf8 $indexPath
Write-Host "Wrote: $indexPath" -ForegroundColor Green
