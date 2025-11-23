<#
setup_env.ps1
Script para crear un virtualenv `.venv`, instalar TensorFlow y los requirements del proyecto.
Ejecutar desde la raíz del repositorio en PowerShell (como administrador si es necesario):
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    .\setup_env.ps1
#>

Write-Host "== Setup del entorno para AgenteDeIA ==" -ForegroundColor Cyan

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

# Encuentra un intérprete Python disponible
$pyCmd = "python"
try {
    $pyPath = (& $pyCmd -c "import sys; print(sys.executable)") 2>$null
} catch {
    # intentar con `py -3`
    try {
        $pyCmd = "py -3"
        $pyPath = (& $pyCmd -c "import sys; print(sys.executable)") 2>$null
    } catch {
        Write-Error "No se encontró Python en PATH. Instala Python 3 y vuelve a intentarlo."; Pop-Location; exit 1
    }
}

Write-Host "Usando intérprete para crear venv: $pyCmd" -ForegroundColor Green

if (-not (Test-Path "$root\.venv")) {
    Write-Host "Creando virtualenv en $root\.venv ..." -ForegroundColor Yellow
    & $pyCmd -m venv .venv
} else {
    Write-Host "Virtualenv ya existe en .venv" -ForegroundColor Green
}

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "No se encontró $venvPython. Asegúrate de que el virtualenv se creó correctamente."; Pop-Location; exit 1
}

Write-Host "Actualizando pip, setuptools y wheel en el venv..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Instalando TensorFlow (puede tardar)..." -ForegroundColor Yellow
& $venvPython -m pip install tensorflow

# Instalar requirements si existen
$reqs = @(
    Join-Path $root 'lstm_notebook\requirements.txt',
    Join-Path $root 'web_demo\requirements.txt'
)
foreach ($r in $reqs) {
    if (Test-Path $r) {
        Write-Host "Instalando paquetes desde $r ..." -ForegroundColor Yellow
        & $venvPython -m pip install -r $r
    } else {
        Write-Host "No existe $r, lo omito." -ForegroundColor DarkYellow
    }
}

Write-Host "\nListo. Para activar el venv y ejecutar la demo en PowerShell:" -ForegroundColor Cyan
Write-Host "    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process" -ForegroundColor Gray
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "    python .\web_demo\app.py" -ForegroundColor Gray

Pop-Location
