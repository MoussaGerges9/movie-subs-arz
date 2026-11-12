# movie-subs-arz

Generate **English and Italian subtitles** for a movie in **Egyptian Arabic** and embed them in the video.
Everything runs **locally and for free**: no API keys, and the movie is never uploaded anywhere.

```bash
python movie_subs.py movie.mkv
```

Output: `movie.ar.srt`, `movie.en.srt`, `movie.it.srt` and `movie.subs.mkv` (the original video plus three subtitle tracks you can switch between in your player).

## How it works

| Step | Tool | What it does |
|------|------|--------------|
| 1. Transcription | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3`, language `ar` | Listens to the audio and produces Arabic text with a timestamp for every word |
| 2. Short subtitles | `chunk_words` | Splits speech into lines of at most ~5 s / 70 characters, at pauses and sentence ends |
| 3. Translation | [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-1.3B), source `arz_Arab` (Egyptian Arabic) | Translates **whole sentences** (more context gives a better translation) |
| 4. Re-alignment | `split_text` | Puts the translated text back onto the original timings, proportionally to the length of each Arabic line |
| 5. Embedding | `ffmpeg` | Adds the `ara`, `eng` and `ita` tracks without re-encoding video or audio |

## Requirements

- **Python 3.10+**
- **ffmpeg** on your `PATH` (Windows: `winget install Gyan.FFmpeg`)
- **An NVIDIA GPU is recommended** (tested on an RTX 4050 with 6 GB). It also runs on CPU, much more slowly.
- **At least 15 GB of free disk space.** Models take ~8 GB (Whisper ~3 GB + NLLB ~5 GB), plus torch.
- An internet connection, only to download the models the first time.

## Installation

```bash
pip install -r requirements.txt
```

To use the GPU, install torch with CUDA (the default Windows package is CPU-only):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

`requirements.txt` also includes `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`, the CUDA libraries faster-whisper needs. On Windows the script makes them visible automatically.

## Usage

```bash
python movie_subs.py movie.mkv                      # everything: .srt files + movie.subs.mkv
python movie_subs.py movie.mp4 -o movie.subs.mp4    # mp4 output (mov_text subtitles)
python movie_subs.py movie.mkv --no-mux             # only the .srt files
python movie_subs.py movie.mkv --whisper medium --nllb facebook/nllb-200-distilled-600M   # low VRAM
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--out` | `<movie>.subs.mkv` | Output video. A `.mp4`/`.mov` extension uses the `mov_text` codec, anything else uses `srt` |
| `--whisper` | `large-v3` | faster-whisper model (`medium` or `small` for less VRAM) |
| `--nllb` | `facebook/nllb-200-distilled-1.3B` | Translation model (`...-distilled-600M` for less VRAM) |
| `--no-mux` | off | Only write the `.srt` files, do not create the video |

### Resuming after an interruption

Transcription is the slowest step. If `movie.ar.srt` already exists the script reuses it, and if `movie.en.srt` or `movie.it.srt` already exist that language is skipped. To **redo** a step, delete the corresponding `.srt` and run again.

### Watching the movie

Open `movie.subs.mkv` in VLC (*Subtitle → Sub Track*) or any player that supports mkv. Alternatively, keep the `.srt` files next to the video: VLC loads them automatically when their name starts with the movie's name.

## Quality and limitations

- **The translation is automatic and rough.** NLLB is trained mostly on written text; informal speech, idioms and jokes in Egyptian cinema are often translated literally or wrongly. It is good enough to follow the plot, not for professional subtitling.
- **Errors add up.** If Whisper mishears a word (common with dialect, proper names and background noise), the translation starts from a wrong sentence.
- **Splitting is proportional.** The translated text is divided over the Arabic timings by length, so a line may end mid-sentence.
- **Mixed languages** (Arabic plus English or French) are not handled: Whisper is forced to `ar`.
- Tested on a ~104 minute movie. There are no precise timing figures, as speed depends heavily on the GPU.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Library cublas64_12.dll is not found` | CUDA 12 libraries are missing | `pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"` |
| `No space left on device`, or a crash (`access violation`) while loading a model | Disk full; on Windows the pagefile grows a lot | Free at least 15 GB on C: |
| `CUDA error: out of memory` | VRAM used by another process, or the model is too large | Close other GPU programs, or use `--whisper medium --nllb ...-600M` |
| torch fails to import (`WinError 193` or `no attribute '__version__'`) | Corrupted torch install (often caused by a full disk) | Delete the `torch` folder in `site-packages` and reinstall |
| Arabic text looks garbled in the console | The Windows console does not display UTF-8; the `.srt` files themselves are correct | Open them in an editor or in VLC |

## Tests

```bash
pytest -q
```

They cover timestamp formatting, SRT reading and writing, splitting into short subtitles, grouping into sentences and distributing the translated text. They do not test the models.

## Possible improvements

- Translate with an **LLM** instead of NLLB, passing the neighbouring lines as context. Options include a local model through LM Studio, a free-tier API, or a paid one such as Claude. This usually handles idioms and informal speech better.
- Whisper models tuned for Egyptian Arabic, if available and verified.
- Splitting the translated text so that it respects sentence boundaries.

## Legal note

Use this script only on videos you have the right to use personally. The generated subtitles are derivative works of the movie: do not distribute them without the rights.
