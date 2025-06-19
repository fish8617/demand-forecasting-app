# PowerShell script to download and install Python on Windows 10
# This script downloads the latest Python 3.11 installer and installs it silently with PATH update

$pythonInstallerUrl = "https://www.python.org/ftp/python/3.11.4/python-3.11.4-amd64.exe"
$installerPath = "$env:TEMP\python-3.11.4-amd64.exe"

Write-Host "Downloading Python installer..."
Invoke-WebRequest -Uri $pythonInstallerUrl -OutFile $installerPath

Write-Host "Installing Python silently..."
Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" -Wait

Write-Host "Cleaning up installer..."
Remove-Item $installerPath

Write-Host "Python installation completed."

Write-Host "Verifying Python installation..."
$pythonVersion = & python --version
Write-Host "Installed Python version: $pythonVersion"

Write-Host "Installing required packages for Streamlit app..."
& python -m pip install --upgrade pip
& python -m pip install streamlit pandas scikit-learn matplotlib openpyxl

Write-Host "Setup complete. You can now run the Streamlit app with:"
Write-Host "streamlit run streamlit_beer_production_system.py"
