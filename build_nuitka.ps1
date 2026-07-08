Write-Host "Inizio build con Nuitka del progetto Game Translation Overlay (beta)..."

# Pulizia build precedenti
param([switch]$Clean)

if ($Clean -and (Test-Path "dist_nuitka")) { Remove-Item -Recurse -Force "dist_nuitka" }

# NOTE sui flag:
#   --standalone              cartella con exe + tutte le dipendenze (equivalente --onedir)
#   --windows-console-mode=force  console SEMPRE visibile per la beta (vedi note precedenti
#                                  su --windowed/sys.stdout=None con PyInstaller: stesso principio,
#                                  vogliamo vedere subito eventuali errori invece di scoprirli da app.log)
#   --enable-plugin=pyqt6      necessario per Qt (plugin ufficiale Nuitka)
#   --enable-plugin=torch      necessario per torch (gestisce le estensioni compilate .pyd di torch)
#   --include-package=X        include l'INTERO albero del pacchetto X leggendo il filesystem,
#                               senza bisogno di importarlo con successo in fase di build
#                               (a differenza di collect_all/collect_submodules di PyInstaller) —
#                               questo e' il motivo per cui proviamo Nuitka: elimina la classe di
#                               problemi avuta con 'soundfile' che rompeva l'introspezione di paddlex
#   --include-package-data=X   include i file NON .py del pacchetto (yaml/json di config, modelli, ecc.)
#   --assume-yes-for-downloads consente a Nuitka di scaricare un compilatore (MinGW64) se non ne trova uno

& .\.venv\Scripts\python.exe -m nuitka `
    --standalone `
    --assume-yes-for-downloads `
    --windows-console-mode=force `
    --enable-plugin=pyqt6 `
    --enable-plugin=torch `
    --include-package=paddle `
    --include-package=paddleocr `
    --include-package=paddlex `
    --include-package=easyocr `
    --include-package=transformers `
    --include-package=scipy `
    --nofollow-import-to=transformers.cli `
    --nofollow-import-to=faiss `
    --nofollow-import-to=matplotlib `
    --include-package-data=paddleocr `
    --include-package-data=paddlex `
    --include-package-data=scipy `
    --include-data-files=settings.json=settings.json `
    --output-dir=dist_nuitka `
    --output-filename=GameTranslationOverlay.exe `
    --company-name="Giorgio" `
    --product-name="Game Translator" `
    --file-version="0.1.0.0" `
    --product-version="0.1.0.0" `
    run_beta.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Build completata! Eseguibile in:"
    Write-Host "  dist_nuitka\run_beta.dist\GameTranslationOverlay.exe"
    Write-Host ""
    Write-Host "Lancialo da un terminale (non doppio click) per vedere subito eventuali errori."
} else {
    Write-Host "Errore durante la compilazione Nuitka. Codice: $LASTEXITCODE"
}