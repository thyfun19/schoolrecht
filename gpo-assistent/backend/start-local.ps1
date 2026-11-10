param(
    [int]$Port = 8001,
    [string]$DependencyPath = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($DependencyPath) {
    $env:PYTHONPATH = (Resolve-Path -LiteralPath $DependencyPath).Path
}
if (!(Test-Path -LiteralPath 'data/e5/bundle.json')) {
    throw 'Importeer eerst de lokale E5-cache; zie VECTOR_BACKEND.md.'
}
$gpoProcess = Start-Process -FilePath "$PSScriptRoot/.venv/Scripts/python.exe" -ArgumentList @(
    '-m','uvicorn','gpo_local.api:app','--host','127.0.0.1','--port',"$Port",'--no-access-log'
) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$PSScriptRoot/data/e5-server.stdout.log" `
    -RedirectStandardError "$PSScriptRoot/data/e5-server.stderr.log"
@{pid=$gpoProcess.Id; url="http://127.0.0.1:$Port/docs"} | ConvertTo-Json | Set-Content -LiteralPath 'data/e5-server.json'
Write-Output "http://127.0.0.1:$Port/docs (PID $($gpoProcess.Id))"
