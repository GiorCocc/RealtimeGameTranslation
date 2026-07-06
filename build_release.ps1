Write-Host "Inizio build del progetto Game Translation Overlay per i Beta Tester..."

# Chiusura forzata eventuali processi rimasti appesi e pulizia log
Stop-Process -Name "GameTranslationOverlay" -Force -ErrorAction SilentlyContinue
if (Test-Path "dist\GameTranslationOverlay\app.log") { Remove-Item -Force "dist\GameTranslationOverlay\app.log" -ErrorAction SilentlyContinue }
if (Test-Path "dist\GameTranslationOverlay\crash.log") { Remove-Item -Force "dist\GameTranslationOverlay\crash.log" -ErrorAction SilentlyContinue }

# Pulizia cartelle vecchie
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }

# IMPORTANTE: si passa il file .spec, MAI run_beta.py direttamente.
# Se passi uno script .py a pyinstaller, lui IGNORA qualunque .spec
# esistente con lo stesso nome e ne rigenera uno nuovo da zero,
# perdendo silenziosamente collect_all/hiddenimports configurati a mano
# (questo era probabilmente il motivo dei risultati incoerenti tra una
# build e l'altra).
& .\.venv\Scripts\pyinstaller.exe --noconfirm GameTranslationOverlay.spec

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build completata con successo! Trovi l'eseguibile in 'dist\GameTranslationOverlay'."
    Write-Host "Nota: per questa beta la console resta visibile (console=True nello .spec)"
    Write-Host "cosi' eventuali errori di libreria (Paddle/OCR) che avvengono PRIMA che"
    Write-Host "app.log sia configurato restano visibili invece di crashare in silenzio."
} else {
    Write-Host "Errore durante la compilazione. Codice: $LASTEXITCODE"
}