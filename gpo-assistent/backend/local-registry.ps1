param([switch]$SkipMigrations)
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    & ./.venv/Scripts/python.exe -m gpo_app.local_database
    if($LASTEXITCODE -ne 0){throw 'Initialiseren databaseconfiguratie mislukt.'}
    & docker compose -p gpo-source-registry -f compose-registry.yaml up -d --wait
    if($LASTEXITCODE -ne 0){throw 'Starten PostgreSQL mislukt.'}
    if(-not $SkipMigrations){
        & ./.venv/Scripts/python.exe -m alembic -c alembic-registry.ini upgrade head
        if($LASTEXITCODE -ne 0){throw 'Databasemigratie mislukt.'}
    }
} finally {Pop-Location}
