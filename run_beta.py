import multiprocessing
import sys
import main

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Inietta automaticamente il flag --debug in modo che i beta tester 
    # generino sempre log avanzati su file (app.log) senza dover passare argomenti.
    sys.argv = [sys.argv[0], "--debug"]
    
    # Avvia l'applicazione principale
    main.main()
