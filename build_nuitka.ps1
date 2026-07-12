<#
    build_nuitka.ps1 - Build eseguibile standalone con Nuitka (Beta)
    Game Translation Overlay

    Uso:
      .\build_nuitka.ps1            # build incrementale (riusa la cache Nuitka)
      .\build_nuitka.ps1 -Clean     # cancella dist_nuitka prima di ricompilare
#>

param([switch]$Clean)

Write-Host "Inizio build con Nuitka del progetto Game Translation Overlay (beta)..."

# --- Pulizia build precedenti (opzionale) ------------------------------
# NOTA: cancellare dist_nuitka distrugge la cache incrementale di Nuitka
# (il singolo ottimizzo piu' impattante sui tempi di build, vedi memoria
# di progetto) -- usare -Clean solo quando serve davvero (es. dopo aver
# cambiato molte dipendenze o in caso di build corrotta).
if ($Clean -and (Test-Path "dist_nuitka")) {
    Write-Host "Pulizia cartella dist_nuitka (cache incrementale persa)..."
    Remove-Item -Recurse -Force "dist_nuitka"
}

# --- pywin32: DLL di sistema non individuate automaticamente da Nuitka -
# pythoncomXXX.dll / pywintypesXXX.dll vivono in
# .venv\Lib\site-packages\pywin32_system32 e NON vengono copiate in
# modalita' --standalone (bug noto: github.com/Nuitka/Nuitka/issues/696,
# /issues/740). Senza queste DLL, il primo "import win32api" nell'exe
# fallisce con ImportError/"DLL load failed" -- probabile causa
# dell'errore visto finora, dato che ui/overlay.py importa win32api e
# win32con a livello di modulo (quindi fallisce appena l'app prova a
# creare l'OverlayWindow, cioe' ad ogni avvio).
$pywin32Dir = ".\.venv\Lib\site-packages\pywin32_system32"
$pywin32Args = @()
if (Test-Path $pywin32Dir) {
    Get-ChildItem $pywin32Dir -Filter "*.dll" | ForEach-Object {
        $pywin32Args += "--include-data-files=$($_.FullName)=$($_.Name)"
    }
    Write-Host ("pywin32: {0} DLL di sistema trovate e incluse." -f $pywin32Args.Count)
} else {
    Write-Warning "Cartella pywin32_system32 non trovata ($pywin32Dir) - win32api rischia di fallire nell'exe. Verifica il path del venv."
}

# --- Modelli MarianMT gia' scaricati in locale (opzionale ma consigliato)
# Se models/ esiste gia' (coppie di lingue scaricate durante lo sviluppo),
# la includiamo nell'exe cosi' i beta tester non devono scaricare nulla
# al primo avvio (coerente con "nessuna API esterna / tutto offline",
# vedere KNOWLEDGE_BASE.md par. 13). Se la ometti, il primo avvio del
# tester scarichera' da HuggingFace (serve Internet).
$modelsArgs = @()
if (Test-Path ".\models") {
    $modelsArgs += "--include-data-dir=models=models"
    Write-Host "Cartella models/ trovata - verra' inclusa nell'exe."
} else {
    Write-Warning "Cartella models/ non trovata - i beta tester dovranno scaricare il modello MarianMT al primo avvio (serve Internet)."
}

# NOTE sui flag principali:
# --standalone                    cartella con exe + tutte le dipendenze (~ --onedir)
# --windows-console-mode=force    console SEMPRE visibile per la beta (vogliamo vedere
#                                  subito eventuali errori invece di scoprirli da app.log)
# --enable-plugin=pyqt6           necessario per Qt (plugin ufficiale Nuitka)
# --enable-plugin=torch           necessario per torch (gestisce le estensioni .pyd)
# --include-package=X             include l'INTERO albero del pacchetto X leggendo il
#                                  filesystem, senza doverlo importare con successo in
#                                  fase di build (a differenza di collect_all di PyInstaller)
# --include-package-data=X        include i file NON .py del pacchetto (yaml/json config, ecc.)
# --nofollow-import-to=paddle.distributed / paddle.incubate
#                                  sottopacchetti "solo training" di paddlepaddle, pesanti
#                                  e mai usati a runtime -- esclusi per velocizzare la build
#                                  e ridurre la superficie di errore
# --assume-yes-for-downloads      consente a Nuitka di scaricare un compilatore (MinGW64)
#                                  se non ne trova uno

$nuitkaArgs = @(
    "--standalone"
    "--assume-yes-for-downloads"
    "--windows-console-mode=force"
    "--enable-plugin=pyqt6"
    "--enable-plugin=torch"
    "--include-package=paddle"
    "--include-package=paddleocr"
    "--include-package=paddlex"
    "--include-package=easyocr"
    "--include-package=transformers"
    "--include-package=scipy"
    "--nofollow-import-to=transformers.cli"
    "--nofollow-import-to=faiss"
    "--nofollow-import-to=matplotlib"
    "--nofollow-import-to=paddle.distributed"
    "--nofollow-import-to=paddle.incubate"
    "--include-package-data=paddle"
    "--include-package-data=paddleocr"
    "--include-package-data=paddlex"
    "--include-package-data=scipy"
    "--include-data-files=settings.json=settings.json"
) + $pywin32Args + $modelsArgs + @(
    "--output-dir=dist_nuitka"
    "--output-filename=GameTranslationOverlay.exe"
    "--company-name=Giorgio"
    "--product-name=Game Translator"
    "--file-version=0.1.0.0"
    "--product-version=0.1.0.0"
    "run_beta.py"
)

& .\.venv\Scripts\python.exe -m nuitka @nuitkaArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Build completata! Eseguibile in:"
    Write-Host "  dist_nuitka\run_beta.dist\GameTranslationOverlay.exe"
    Write-Host ""
    Write-Host "Lancialo da un terminale (non doppio click) per vedere subito eventuali errori."
} else {
    Write-Host "Errore durante la compilazione Nuitka. Codice: $LASTEXITCODE"
}