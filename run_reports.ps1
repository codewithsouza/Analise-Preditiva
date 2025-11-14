Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Ir para a pasta do script
Set-Location -Path $PSScriptRoot

# Criar venv se não existir
if (!(Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Criando ambiente virtual (.venv)..."
    py -m venv .venv
}

# Ativar venv
Write-Host "Ativando .venv..."
. ".\.venv\Scripts\Activate.ps1"

# Instalar dependências
Write-Host "Instalando dependências..."
python -m pip install --upgrade pip
python -m pip install -r ".\requirements.txt"

# Executar o gerador de relatórios
Write-Host "Executando gerar_relatorios_series.py..."
python ".\gerar_relatorios_series.py"

Write-Host "Concluído."


