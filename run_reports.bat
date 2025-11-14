@echo off
setlocal enabledelayedexpansion

REM Ir para a pasta do script
cd /d "%~dp0"

REM Criar venv se não existir
if not exist ".venv\Scripts\python.exe" (
  echo Criando ambiente virtual (.venv)...
  py -m venv .venv
)

REM Ativar venv
call ".venv\Scripts\activate"

REM Instalar dependencias
python -m pip install --upgrade pip
python -m pip install -r "requirements.txt"

REM Executar script
python "gerar_relatorios_series.py"

pause


