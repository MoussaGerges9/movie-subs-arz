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
| 1. Transcription | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3`, language `ar` | Arabic text with a timestamp for every word. A lowered VAD threshold keeps speech the default filter discards, and a prompt made of colloquial Egyptian words (plus the character names you give with `--names`) keeps Whisper in dialect instead of drifting to formal Arabic. Endless repeated lines are dropped |
| 2. Short subtitles | `chunk_words` | Splits speech into lines of at most ~5 s / 70 characters, at pauses and sentence ends |
| 3. Sentences | `group_sentences` | Regroups the short lines into whole sentences, so the translator sees complete thoughts |
| 4. Translation | **Local LLM** (default, e.g. Gemma 4 E4B behind any OpenAI-compatible server) or [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-1.3B) (`arz_Arab`) as a fallback | Translates whole sentences, with the film title and character names as context |
| 5. Re-alignment | LLM (`split_lines`), with `split_text` as fallback | Puts the translated sentence back onto the original timings. The LLM splits it by meaning; its answer is accepted only if it keeps exactly the same words and leaves no empty piece. Otherwise the text is split proportionally to the length of each Arabic line, preferring cuts at commas and full stops, with at least one word per line |
| 6. Embedding | `ffmpeg` | Adds the `ara`, `eng` and `ita` tracks without re-encoding video or audio |

## Requirements

- **Python 3.10+**
- **ffmpeg** on your `PATH` (Windows: `winget install Gyan.FFmpeg`)
- **An NVIDIA GPU is recommended** (tested on an RTX 4050 with 6 GB). It also runs on CPU, much more slowly.
- **At least 15 GB of free disk space.** Whisper ~3 GB, the LLM ~5 GB, NLLB ~5 GB (only if used), plus torch.
- **For the LLM translator** (recommended): any OpenAI-compatible server with JSON-schema output, such as [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`, [LM Studio](https://lmstudio.ai) or Ollama, serving an instruction-tuned model. Tested with Gemma 4 E4B (Q4_K_M, ~3.3 GB of VRAM).
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

### Quick start (CPU, or a GPU with plenty of VRAM)

```bash
python movie_subs.py movie.mkv --names "Nisa, Zizi"
```

With `--translator auto` (the default) the script uses the LLM if its server answers on `--llm-url`, otherwise NLLB.

### Recommended on a 6 GB GPU: two steps

Whisper `large-v3` and the LLM do not fit in 6 GB together, so transcribe first, free the GPU, then translate:

```bash
# 1. Transcription only (writes movie.ar.srt)
python movie_subs.py movie.mkv --transcribe-only --names "نيسا، زيزي / Nisa, Zizi"

# 2. Start the LLM server (llama.cpp shown; LM Studio / Ollama work too)
llama-server -m gemma-4-E4B-it-Q4_K_M.gguf -ngl 99 -c 8192 --port 1234 --jinja

# 3. Translate and embed (reuses movie.ar.srt)
python movie_subs.py movie.mkv --translator llm --names "نيسا، زيزي / Nisa, Zizi" --title "an Egyptian comedy film"
```

Other examples:

```bash
python movie_subs.py movie.mp4 -o movie.subs.mp4    # mp4 output (mov_text subtitles)
python movie_subs.py movie.mkv --no-mux             # only the .srt files
python movie_subs.py movie.mkv --translator nllb    # no server needed
python movie_subs.py movie.mkv --whisper medium --translator nllb --nllb facebook/nllb-200-distilled-600M   # low VRAM
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--out` | `<movie>.subs.mkv` | Output video. A `.mp4`/`.mov` extension uses the `mov_text` codec, anything else uses `srt` |
| `--whisper` | `large-v3` | faster-whisper model (`medium` or `small` for less VRAM) |
| `--translator` | `auto` | `llm` (local LLM server), `nllb` (no server), or `auto` (LLM if the server answers) |
| `--llm-url` | `http://localhost:1234/v1` | OpenAI-compatible server |
| `--llm-model` | `local` | Model name on the server |
| `--names` | empty | Character names, in Arabic and/or Latin letters. They help both transcription and translation |
| `--title` | asked at start | Name of the film, given to the LLM as context. In an interactive terminal the script asks for it when the option is missing; otherwise a generic description is used |
| `--lang` | asked at start | `en`, `it` or `both`. In an interactive terminal the script asks which translation you want (default both); without a terminal the default is both |
| `--nllb` | `facebook/nllb-200-distilled-1.3B` | NLLB model for the fallback translator |
| `--transcribe-only` | off | Stop after writing `movie.ar.srt` |
| `--no-mux` | off | Only write the `.srt` files, do not create the video |

### Resuming after an interruption

If `movie.ar.srt` already exists the script reuses it, and if `movie.en.srt` or `movie.it.srt` already exist that language is skipped. To **redo** a step, delete (or rename) the corresponding `.srt` and run again.

### Watching the movie

Open `movie.subs.mkv` in VLC (*Subtitle → Sub Track*) or any player that supports mkv. Alternatively, keep the `.srt` files next to the video: VLC loads them automatically when their name starts with the movie's name.

## Results

Measured on a ~104 minute Egyptian comedy, comparing the first version of this tool (NLLB 1.3B on merged sentences) with the current one (local LLM). There is no reference translation, so quality is judged by reading samples.

| | Before | Now |
|---|---|---|
| Translation of idioms and exclamations | Often literal or wrong ("What we ended up with was medicine is the story of how many days?") | Natural ("So, how many days is this whole thing going to take?") |
| Longest subtitle | 15.3 s | 5.4 s |
| Arabic words recognised | 9302 | 9463 (+1.7%) |
| Subtitle lines without text (EN / IT), out of 2199 | 6 / 13 | 1 / 1 |

## Quality and limitations

- **The translation is automatic.** The LLM handles dialect and idioms much better than NLLB, but it is still not professional subtitling.
- **Errors add up.** If Whisper mishears a word (common with dialect, proper names and background noise), the LLM often produces a fluent sentence that is wrong, and it does not flag it.
- **Re-splitting can still drift.** The translated sentence is divided over the original lines, so a piece may show up about one line early or late, mostly in fast dialogue with many one-word lines. When the LLM's split is rejected, the proportional fallback ignores meaning.
- **Mixed languages** (Arabic plus English or French) are not handled: Whisper is forced to `ar`.
- **It is slow.** On a 6 GB laptop GPU the LLM step took about two hours for a 104 minute movie (translation plus alignment, English and Italian), on top of the transcription.
- **Gemma 4** has a "thinking" mode on by default; the script turns it off per request with `chat_template_kwargs`. Other servers may ignore this field.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `nessun server LLM su ...` | `--translator llm` but the server is not running | Start the server, or use `--translator nllb` |
| `CUDA error: out of memory` | VRAM used by another process, or Whisper and the LLM are loaded together | Use the two-step flow (`--transcribe-only`), close other GPU programs, or use `--whisper medium` |
| `Library cublas64_12.dll is not found` | CUDA 12 libraries are missing | `pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"` |
| `No space left on device`, or a crash (`access violation`) while loading a model | Disk full; on Windows the pagefile grows a lot | Free at least 15 GB on C: |
| torch fails to import (`WinError 193` or `no attribute '__version__'`) | Corrupted torch install (often caused by a full disk) | Delete the `torch` folder in `site-packages` and reinstall |
| Arabic text looks garbled in the console | The Windows console does not display UTF-8; the `.srt` files themselves are correct | Open them in an editor or in VLC |

## Tests

```bash
pytest -q
```

They cover timestamp formatting, SRT reading and writing, splitting into short subtitles, grouping into sentences, distributing the translated text, dropping repeated loops, parsing the LLM's JSON output and validating its line split. They do not test the models.

## Possible improvements

- A character glossary built from the film and updated batch by batch, as in [llm-subtrans](https://github.com/machinewrapped/llm-subtrans) and [tarjem](https://github.com/samerzmd/tarjem).
- Compare other models: NLLB-200 3.3B (note: with CTranslate2 the tokenizer must be fed the language code manually, otherwise the output is garbage), Aya, Qwen, Gemma 3.
- A second ASR as a cross-check on doubtful lines.

## Legal note

Use this script only on videos you have the right to use personally. The generated subtitles are derivative works of the movie: do not distribute them without the rights.
