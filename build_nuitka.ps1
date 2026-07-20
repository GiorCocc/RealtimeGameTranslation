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

# --- pywin32: gestito automaticamente da Nuitka (dll-files plugin) -----
# Nelle versioni recenti di Nuitka (qui: 4.1.3), pythoncomXXX.dll e
# pywintypesXXX.dll vengono individuate e copiate in automatico dal
# dll-files plugin quando scansiona le dipendenze binarie dei moduli
# nativi inclusi (win32api.pyd, pywintypes.pyd, ecc.). NON includerle
# di nuovo a mano con --include-data-files: causa un conflitto in fase
# di link ("data file ... conflicts with dll ...") perche' Nuitka le ha
# gia' messe li' da sola. Se in futuro dovessi tornare a un errore di
# ImportError su win32api nell'exe, e' un segnale che vale la pena
# ricontrollare, ma con questa versione non serve intervento manuale.
$pywin32Args = @()

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
# --nofollow-import-to=sympy.polys.polyquinticconst
#                                  modulo enorme (costanti precalcolate per equazioni di 5o
#                                  grado), tirato dentro da torch/sympy ma mai usato da questa
#                                  app -- da solo puo' richiedere 30-40 minuti di compilazione C
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
    "--nofollow-import-to=sympy.polys.polyquinticconst"
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