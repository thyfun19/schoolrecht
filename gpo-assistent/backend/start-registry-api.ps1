param([ValidateRange(1024,65535)][int]$Port=8002,[string]$DependencyPath='')
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    if($DependencyPath){$env:PYTHONPATH=(Resolve-Path -LiteralPath $DependencyPath).Path}
    & ./.venv/Scripts/python.exe -m gpo_app.cli counts
    if($LASTEXITCODE -ne 0){throw 'Bronregister is niet beschikbaar.'}
    & ./.venv/Scripts/python.exe -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', $Port)); s.close()"
    if($LASTEXITCODE -ne 0){throw 'Poort is bezet; bestaande server blijft ongemoeid.'}
    $gpoRegistryProcess=Start-Process -FilePath "$PSScriptRoot/.venv/Scripts/python.exe" -ArgumentList @(
        '-m','uvicorn','gpo_app.api:app','--host','127.0.0.1','--port',"$Port",'--no-access-log'
    ) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput "$PSScriptRoot/data/registry/server.stdout.log" `
      -RedirectStandardError "$PSScriptRoot/data/registry/server.stderr.log"
    @{pid=$gpoRegistryProcess.Id;url="http://127.0.0.1:$Port/docs"} | ConvertTo-Json |
        Set-Content -LiteralPath data/registry/server.json
    Write-Output "GPO-bronregister: http://127.0.0.1:$Port/docs"
} finally {Pop-Location}
