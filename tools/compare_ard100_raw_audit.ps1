param(
    [string]$LocalBaseline = "C:\Users\aaron\Desktop\URAP\artifacts\ard100_local_baseline.json",
    [string]$RemoteAudit = "C:\Users\aaron\Desktop\URAP\artifacts\ard100_remote_audit.json"
)

$ErrorActionPreference = "Stop"
$local = Get-Content -LiteralPath $LocalBaseline -Raw | ConvertFrom-Json -AsHashtable
$remote = Get-Content -LiteralPath $RemoteAudit -Raw | ConvertFrom-Json -AsHashtable
$failures = [System.Collections.Generic.List[string]]::new()
foreach ($field in @("train_count", "test_count", "train_bytes", "test_bytes", "annotations_bytes", "annotations_sha256", "extract_frames_bytes", "extract_frames_sha256")) {
    if ($local[$field] -ne $remote[$field]) { $failures.Add("$field local=$($local[$field]) remote=$($remote[$field])") }
}
foreach ($group in @("train_files", "test_files")) {
    foreach ($name in $local[$group].Keys) {
        if (-not $remote[$group].ContainsKey($name)) { $failures.Add("missing remote $group/$name"); continue }
        if ([int64]$local[$group][$name] -ne [int64]$remote[$group][$name]) { $failures.Add("size mismatch $group/$name") }
    }
    foreach ($name in $remote[$group].Keys) {
        if (-not $local[$group].ContainsKey($name)) { $failures.Add("unexpected remote $group/$name") }
    }
}
if (-not $remote.complete) { $failures.Add("remote audit complete=false") }
if ($failures.Count) {
    $failures | ForEach-Object { Write-Error $_ }
    exit 1
}
Write-Host "ARD100 RAW AUDIT PASSED"
Write-Host "train=$($remote.train_count) test=$($remote.test_count) total_bytes=$([int64]$remote.train_bytes + [int64]$remote.test_bytes)"
Write-Host "annotations_sha256=$($remote.annotations_sha256)"
Write-Host "extract_frames_sha256=$($remote.extract_frames_sha256)"
