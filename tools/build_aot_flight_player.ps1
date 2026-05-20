param(
    [string]$FlightDir = "D:\URAP_datasets\AOT\part1\Images\0001ba865c8e410e88609541b8f55ffc",
    [string]$OutFile = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\doc\aot_flight_player.html",
    [int]$DefaultFps = 15
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $FlightDir)) {
    throw "Flight directory not found: $FlightDir"
}

$flight = Get-Item $FlightDir
$frames = Get-ChildItem -Path $flight.FullName -File | Sort-Object Name
if ($frames.Count -eq 0) {
    throw "No frames found in: $FlightDir"
}

$frameUrls = $frames | ForEach-Object {
    "file:///" + ($_.FullName -replace "\\", "/")
}

$frameJson = ConvertTo-Json $frameUrls -Compress
$title = "AOT Flight Player - $($flight.Name)"

$html = @"
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>$title</title>
  <style>
    :root {
      --bg: #0f1115;
      --panel: #171a21;
      --muted: #98a2b3;
      --text: #f5f7fa;
      --accent: #6ee7ff;
      --accent2: #8b5cf6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: linear-gradient(135deg, #0f1115, #131722 55%, #1a1f2b);
      color: var(--text);
    }
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }
    .card {
      background: rgba(23, 26, 33, 0.95);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }
    h1 {
      margin: 0 0 10px;
      font-size: 28px;
      line-height: 1.15;
    }
    .meta {
      color: var(--muted);
      margin-bottom: 18px;
      word-break: break-all;
    }
    img {
      width: 100%;
      max-height: 70vh;
      object-fit: contain;
      border-radius: 12px;
      background: #000;
      display: block;
    }
    .toolbar {
      display: grid;
      grid-template-columns: auto auto auto 1fr auto;
      gap: 12px;
      align-items: center;
      margin-top: 16px;
    }
    button, select {
      background: #232836;
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 10px;
      padding: 10px 14px;
      cursor: pointer;
    }
    input[type="range"] {
      width: 100%;
    }
    .status {
      margin-top: 12px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
    .hint {
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>$title</h1>
      <div class="meta">$($flight.FullName)</div>
      <img id="frame" alt="AOT frame" />
      <div class="toolbar">
        <button id="playPause">Play</button>
        <button id="prev">Prev</button>
        <button id="next">Next</button>
        <input id="scrubber" type="range" min="0" max="$($frames.Count - 1)" value="0" />
        <select id="speed">
          <option value="5">5 FPS</option>
          <option value="10">10 FPS</option>
          <option value="$DefaultFps" selected>$DefaultFps FPS</option>
          <option value="24">24 FPS</option>
          <option value="30">30 FPS</option>
        </select>
      </div>
      <div class="status">
        <div id="counter"></div>
        <div id="filename"></div>
      </div>
      <div class="hint">If the browser blocks local image access, open this HTML in Edge/Chrome and allow local file loading, or serve it with a local static server.</div>
    </div>
  </div>

  <script>
    const frames = $frameJson;
    const frameNames = frames.map(p => p.split('/').pop());
    let index = 0;
    let timer = null;

    const img = document.getElementById('frame');
    const scrubber = document.getElementById('scrubber');
    const counter = document.getElementById('counter');
    const filename = document.getElementById('filename');
    const playPause = document.getElementById('playPause');
    const speed = document.getElementById('speed');

    function render() {
      img.src = frames[index];
      scrubber.value = index;
      counter.textContent = `Frame ${index + 1} / ${frames.length}`;
      filename.textContent = frameNames[index];
    }

    function stopPlayback() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
      playPause.textContent = 'Play';
    }

    function startPlayback() {
      stopPlayback();
      const fps = Number(speed.value);
      timer = setInterval(() => {
        index = (index + 1) % frames.length;
        render();
      }, 1000 / fps);
      playPause.textContent = 'Pause';
    }

    playPause.addEventListener('click', () => {
      if (timer) {
        stopPlayback();
      } else {
        startPlayback();
      }
    });

    document.getElementById('prev').addEventListener('click', () => {
      index = Math.max(0, index - 1);
      render();
    });

    document.getElementById('next').addEventListener('click', () => {
      index = Math.min(frames.length - 1, index + 1);
      render();
    });

    scrubber.addEventListener('input', (e) => {
      index = Number(e.target.value);
      render();
    });

    speed.addEventListener('change', () => {
      if (timer) startPlayback();
    });

    document.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        playPause.click();
      } else if (e.code === 'ArrowRight') {
        index = Math.min(frames.length - 1, index + 1);
        render();
      } else if (e.code === 'ArrowLeft') {
        index = Math.max(0, index - 1);
        render();
      }
    });

    render();
  </script>
</body>
</html>
"@

$outDir = Split-Path -Parent $OutFile
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
Set-Content -Path $OutFile -Value $html -Encoding UTF8

Write-Host "Wrote player to: $OutFile"
Write-Host "Frames: $($frames.Count)"
Write-Host "Flight: $($flight.FullName)"
