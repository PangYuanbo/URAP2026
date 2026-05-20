param(
    [string]$FlightDir,
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

if (-not $FlightDir) {
    throw "FlightDir is required."
}
if (-not (Test-Path $FlightDir)) {
    throw "Flight directory not found: $FlightDir"
}

$flight = Get-Item $FlightDir
$frames = Get-ChildItem -Path $flight.FullName -File | Sort-Object Name
if ($frames.Count -eq 0) {
    throw "No frames found in: $FlightDir"
}

$frameNames = $frames | ForEach-Object { $_.Name }
$frameLookup = @{}
foreach ($frame in $frames) {
    $frameLookup[$frame.Name] = $frame.FullName
}

$listener = [System.Net.HttpListener]::new()
$prefix = "http://localhost:$Port/"
$listener.Prefixes.Add($prefix)
$listener.Start()

Write-Host "AOT flight server listening on $prefix"
Write-Host "Flight: $($flight.FullName)"
Write-Host "Frames: $($frames.Count)"

$html = @"
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AOT Flight Player</title>
  <style>
    :root {
      --bg: #0f1115;
      --panel: #171a21;
      --muted: #98a2b3;
      --text: #f5f7fa;
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
    h1 { margin: 0 0 10px; font-size: 28px; line-height: 1.15; }
    .meta { color: var(--muted); margin-bottom: 18px; word-break: break-all; }
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
    input[type="range"] { width: 100%; }
    .status {
      margin-top: 12px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1 id="title">AOT Flight Player</h1>
      <div class="meta" id="meta"></div>
      <img id="frame" alt="AOT frame" />
      <div class="toolbar">
        <button id="playPause">Play</button>
        <button id="prev">Prev</button>
        <button id="next">Next</button>
        <input id="scrubber" type="range" min="0" value="0" />
        <select id="speed">
          <option value="5">5 FPS</option>
          <option value="10">10 FPS</option>
          <option value="15" selected>15 FPS</option>
          <option value="24">24 FPS</option>
          <option value="30">30 FPS</option>
        </select>
      </div>
      <div class="status">
        <div id="counter"></div>
        <div id="filename"></div>
      </div>
    </div>
  </div>

  <script>
    let frames = [];
    let index = 0;
    let timer = null;
    let flightDir = "";

    const img = document.getElementById('frame');
    const scrubber = document.getElementById('scrubber');
    const counter = document.getElementById('counter');
    const filename = document.getElementById('filename');
    const playPause = document.getElementById('playPause');
    const speed = document.getElementById('speed');
    const meta = document.getElementById('meta');

    function render() {
      if (!frames.length) return;
      const name = frames[index];
      img.src = '/frames/' + encodeURIComponent(name);
      scrubber.value = index;
      counter.textContent = `Frame ${index + 1} / ${frames.length}`;
      filename.textContent = name;
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
      if (timer) stopPlayback(); else startPlayback();
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

    async function init() {
      const data = await fetch('/api/frames').then(r => r.json());
      frames = data.frames;
      flightDir = data.flight_dir;
      document.getElementById('title').textContent = 'AOT Flight Player - ' + data.flight_name;
      meta.textContent = flightDir;
      scrubber.max = Math.max(0, frames.length - 1);
      render();
    }

    init();
  </script>
</body>
</html>
"@

try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response
        $path = $request.Url.AbsolutePath

        try {
            if ($path -eq "/") {
                $bytes = [System.Text.Encoding]::UTF8.GetBytes($html)
                $response.ContentType = "text/html; charset=utf-8"
                $response.OutputStream.Write($bytes, 0, $bytes.Length)
            }
            elseif ($path -eq "/api/frames") {
                $payload = @{
                    flight_name = $flight.Name
                    flight_dir  = $flight.FullName
                    frames      = $frameNames
                } | ConvertTo-Json -Compress
                $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
                $response.ContentType = "application/json; charset=utf-8"
                $response.OutputStream.Write($bytes, 0, $bytes.Length)
            }
            elseif ($path.StartsWith("/frames/")) {
                $name = [System.Uri]::UnescapeDataString($path.Substring(8))
                if (-not $frameLookup.ContainsKey($name)) {
                    $response.StatusCode = 404
                }
                else {
                    $file = $frameLookup[$name]
                    $bytes = [System.IO.File]::ReadAllBytes($file)
                    $response.ContentType = "image/png"
                    $response.OutputStream.Write($bytes, 0, $bytes.Length)
                }
            }
            else {
                $response.StatusCode = 404
            }
        }
        finally {
            $response.OutputStream.Close()
        }
    }
}
finally {
    if ($listener.IsListening) {
        $listener.Stop()
    }
    $listener.Close()
}
