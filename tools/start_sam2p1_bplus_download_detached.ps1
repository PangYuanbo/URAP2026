param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='sam2p1_bplus_download')
$ErrorActionPreference='Stop'
$root=Join-Path $RepoRoot 'artifacts\detached_sam2p1_bplus_download'
$logs=Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$target=Join-Path $RepoRoot 'third_party\samurai\sam2\checkpoints\sam2.1_hiera_base_plus.pt'
New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
$url='https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt'
$timestamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout=Join-Path $logs ($RunId+'_'+$timestamp+'.out.txt')
$stderr=Join-Path $logs ($RunId+'_'+$timestamp+'.err.txt')
$arguments=@('-L','--fail','--retry','5','--retry-delay','5','--continue-at','-','--output',$target,$url)
$process=Start-Process -FilePath 'curl.exe' -ArgumentList $arguments -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content (Join-Path $root ($RunId+'.pid')) $process.Id
@{pid=$process.Id;started=(Get-Date -Format o);command=@('curl.exe')+$arguments;url=$url;target=$target;stdout=$stdout;stderr=$stderr}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $root ($RunId+'.meta.json'))
Write-Output "started pid=$($process.Id) target=$target stdout=$stdout stderr=$stderr"
