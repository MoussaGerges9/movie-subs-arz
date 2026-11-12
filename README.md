# movie-subs-arz

Genera sottotitoli **inglesi e italiani** per un film in **arabo egiziano** e li incorpora nel video.
Tutto gira **in locale e gratis**: nessuna API key, nessun upload del film.

```bash
python movie_subs.py film.mkv
```

Risultato: `film.ar.srt`, `film.en.srt`, `film.it.srt` e `film.subs.mkv` (video originale + 3 tracce sottotitoli selezionabili dal player).

## Come funziona

| Fase | Strumento | Cosa fa |
|------|-----------|---------|
| 1. Trascrizione | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3`, lingua `ar` | Ascolta l'audio e produce testo arabo con il tempo di ogni parola |
| 2. Sottotitoli brevi | `chunk_words` | Spezza il parlato in righe di max ~5 s / 70 caratteri, a pause e fine frase |
| 3. Traduzione | [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-1.3B), sorgente `arz_Arab` (arabo egiziano) | Traduce **frasi intere** (più contesto = traduzione migliore) |
| 4. Allineamento | `split_text` | Rimette il testo tradotto sui tempi originali, proporzionalmente alla lunghezza delle righe arabe |
| 5. Incorporamento | `ffmpeg` | Aggiunge le tracce `ara`, `eng`, `ita` senza ricodificare video e audio |

## Requisiti

- **Python 3.10+**
- **ffmpeg** nel `PATH` (Windows: `winget install Gyan.FFmpeg`)
- **GPU NVIDIA consigliata** (testato su RTX 4050 6 GB). Senza GPU funziona su CPU, molto più lentamente.
- **Spazio disco: almeno 15 GB liberi.** Modelli ~8 GB (Whisper ~3 GB + NLLB ~5 GB) più torch.
- Connessione internet solo per scaricare i modelli la prima volta.

## Installazione

```bash
pip install -r requirements.txt
```

Per usare la GPU, installa torch con CUDA (il pacchetto di default su Windows è solo CPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

`requirements.txt` include anche `nvidia-cublas-cu12` e `nvidia-cudnn-cu12`: sono le librerie CUDA che faster-whisper richiede. Su Windows lo script le rende visibili automaticamente.

## Uso

```bash
python movie_subs.py film.mkv                      # tutto: .srt + film.subs.mkv
python movie_subs.py film.mp4 -o film.subs.mp4     # output mp4 (sottotitoli mov_text)
python movie_subs.py film.mkv --no-mux             # solo i file .srt
python movie_subs.py film.mkv --whisper medium --nllb facebook/nllb-200-distilled-600M   # poca VRAM
```

| Opzione | Default | Descrizione |
|---------|---------|-------------|
| `-o`, `--out` | `<film>.subs.mkv` | Video di output. L'estensione `.mp4`/`.mov` usa il codec `mov_text`, altrimenti `srt` |
| `--whisper` | `large-v3` | Modello faster-whisper (`medium`, `small` per meno VRAM) |
| `--nllb` | `facebook/nllb-200-distilled-1.3B` | Modello di traduzione (`...-distilled-600M` per meno VRAM) |
| `--no-mux` | off | Genera solo i `.srt`, senza creare il video |

### Riprendere dopo un'interruzione

La trascrizione è la parte più lunga. Se `film.ar.srt` esiste già lo script lo riusa; se esistono già `film.en.srt` o `film.it.srt` salta quella lingua. Per **rifare** una fase, cancella il relativo `.srt` e rilancia.

### Guardare il film

Apri `film.subs.mkv` con VLC (*Sottotitoli → Traccia sottotitoli*) o con qualunque player che supporti l'mkv. In alternativa lascia i `.srt` accanto al video: VLC li carica da solo se il nome inizia come quello del film.

## Qualità e limiti

- **La traduzione è automatica e grezza.** NLLB è addestrato su molto testo scritto; il parlato informale, i modi di dire e gli scherzi del cinema egiziano spesso vengono tradotti alla lettera o sbagliati. Va bene per seguire la trama, non per un sottotitolaggio professionale.
- **Gli errori si sommano**: se Whisper sbaglia una parola (frequente con il dialetto, i nomi propri e il rumore di fondo) la traduzione parte già sbagliata.
- **La suddivisione è proporzionale**: il testo tradotto è diviso sui tempi dell'arabo in base alla lunghezza, quindi una riga può terminare a metà frase.
- **Più lingue parlate** (arabo + inglese/francese) non sono gestite: Whisper è forzato su `ar`.
- Testato su un film da ~104 minuti: non ci sono stime di tempo precise, dipendono molto dalla GPU.

## Problemi comuni

| Errore | Causa | Soluzione |
|--------|-------|-----------|
| `Library cublas64_12.dll is not found` | Mancano le librerie CUDA 12 | `pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"` |
| `No space left on device` o crash (`access violation`) durante il caricamento del modello | Disco pieno; su Windows il pagefile cresce molto | Libera almeno 15 GB su C: |
| `CUDA error: out of memory` | VRAM occupata da un altro processo o modello troppo grande | Chiudi gli altri programmi che usano la GPU, oppure usa `--whisper medium --nllb ...-600M` |
| Torch non si importa (`WinError 193` o `no attribute '__version__'`) | Installazione di torch corrotta (spesso per disco pieno) | Cancella la cartella `torch` in `site-packages` e reinstalla |
| Testo arabo illeggibile in console | La console Windows non mostra UTF-8; i file `.srt` sono corretti | Aprili con un editor o con VLC |

## Test

```bash
pytest -q
```

Coprono formattazione dei timestamp, lettura/scrittura SRT, suddivisione in sottotitoli brevi, raggruppamento in frasi e distribuzione del testo tradotto. Non testano i modelli.

## Possibili miglioramenti

- Traduzione con un **LLM** (es. Claude via API) al posto di NLLB, passando le righe vicine come contesto: di solito rende meglio modi di dire e registro informale.
- Modelli Whisper ottimizzati per l'arabo egiziano, se disponibili e verificati.
- Suddivisione del testo tradotto che rispetti i confini di frase.

## Note legali

Usa lo script solo su video di cui hai il diritto di uso personale. I sottotitoli prodotti sono opere derivate del film: non distribuirli senza i diritti.
