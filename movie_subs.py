#!/usr/bin/env python
"""Film in arabo egiziano -> sottotitoli EN + IT (.srt) e video con sottotitoli incorporati.

Pipeline, tutto locale e gratuito:
  1. faster-whisper (large-v3, lingua "ar")  -> trascrizione araba con timestamp
  2. NLLB-200 (sorgente arz_Arab = arabo egiziano) -> inglese e italiano
  3. ffmpeg -> muxa le tracce sottotitoli nel contenitore (senza ricodificare)

Uso:  python movie_subs.py film.mkv
"""
import argparse
import functools
import subprocess
import sys
from pathlib import Path

def _expose_pip_cuda_dlls() -> None:
    """Rende visibili a ctranslate2 le DLL cuBLAS/cuDNN installate via pip (nvidia-*-cu12)."""
    import os
    import site

    for root in [site.getusersitepackages(), *site.getsitepackages()]:
        for d in Path(root, "nvidia").glob("*/bin"):
            os.add_dll_directory(str(d))
            os.environ["PATH"] = f"{d}{os.pathsep}{os.environ['PATH']}"


if sys.platform == "win32":
    _expose_pip_cuda_dlls()

LANGS = {  # codice -> (ISO 639-2 per ffmpeg, codice NLLB)
    "en": ("eng", "eng_Latn"),
    "it": ("ita", "ita_Latn"),
}


def ts(sec: float) -> str:
    ms = round(sec * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(path: Path, segs, texts) -> None:
    blocks = [
        f"{n}\n{ts(s)} --> {ts(e)}\n{t.strip()}\n"
        for n, ((s, e, _), t) in enumerate(((x, t) for x, t in zip(segs, texts) if t.strip()), 1)
    ]
    path.write_text("\n".join(blocks), encoding="utf-8")


def parse_srt(path: Path):
    segs = []
    for blk in path.read_text(encoding="utf-8").strip().split("\n\n"):
        _, times, *text = blk.split("\n")
        a, b = (
            sum(float(x) * k for x, k in zip(t.replace(",", ".").split(":"), (3600, 60, 1)))
            for t in times.split(" --> ")
        )
        segs.append((a, b, " ".join(text)))
    return segs


def chunk_words(words, max_dur=5.0, max_chars=70, max_gap=0.8):
    """Raggruppa parole (start, end, testo) in sottotitoli brevi: spezza su pause, punteggiatura, durata e lunghezza."""
    out, cur = [], []

    def flush():
        if cur:
            out.append((cur[0][0], cur[-1][1], "".join(w[2] for w in cur).strip()))
            cur.clear()

    for w in words:
        if cur and (
            w[0] - cur[-1][1] > max_gap
            or w[1] - cur[0][0] > max_dur
            or len("".join(x[2] for x in cur) + w[2]) > max_chars
        ):
            flush()
        cur.append(w)
        if w[2].strip()[-1:] in (".", "?", "!", "؟", "…"):
            flush()
    flush()
    return out


def transcribe(video: Path, model_name: str, device: str):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type="int8_float16" if device == "cuda" else "int8")
    it, info = model.transcribe(
        str(video),
        language="ar",
        beam_size=5,
        vad_filter=True,  # taglia musica/silenzi: meno allucinazioni
        condition_on_previous_text=False,  # evita loop di ripetizioni sui film lunghi
        word_timestamps=True,  # servono per spezzare i discorsi lunghi in sottotitoli brevi
    )
    segs = []
    for s in it:
        words = [(w.start, w.end, w.word) for w in s.words or []]
        segs += chunk_words(words) if words else [(s.start, s.end, s.text.strip())]
        print(f"\r  trascrizione {s.end / info.duration:6.1%}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return segs


@functools.lru_cache(maxsize=1)  # ricaricare nello stesso processo rompe il client di huggingface_hub
def load_nllb(model_name: str, device: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, src_lang="arz_Arab")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    return tok, model


def translate(texts, tgt: str, model_name: str, device: str, batch: int = 16):
    import torch

    tok, model = load_nllb(model_name, device)
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i : i + batch], return_tensors="pt", padding=True, truncation=True, max_length=200).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), num_beams=4, max_length=200)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        print(f"\r  traduzione {tgt} {min(i + batch, len(texts)) / len(texts):6.1%}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return out


def group_sentences(segs, max_chars=250, max_gap=1.5):
    """Raggruppa sottotitoli consecutivi in frasi intere (fine frase, pausa lunga o troppo testo)."""
    groups, cur = [], []
    for seg in segs:
        if cur and (seg[0] - cur[-1][1] > max_gap or sum(len(x[2]) + 1 for x in cur) + len(seg[2]) > max_chars):
            groups.append(cur)
            cur = []
        cur.append(seg)
        if seg[2].strip()[-1:] in (".", "?", "!", "؟", "…"):
            groups.append(cur)
            cur = []
    return groups + ([cur] if cur else [])


def split_text(text: str, weights):
    """Divide `text` per parole in len(weights) pezzi, proporzionali ai pesi (lunghezza dell'arabo di ogni pezzo)."""
    words, total, out, prev, cum = text.split(), sum(weights), [], 0, 0
    for w in weights:
        cum += w
        end = round(len(words) * cum / total)
        out.append(" ".join(words[prev:end]))
        prev = end
    return out


def translate_sentences(segs, tgt: str, model_name: str, device: str):
    """Traduce frasi intere (più contesto = traduzione migliore), poi rimette il testo sui tempi originali."""
    groups = group_sentences(segs)
    tr = translate([" ".join(x[2] for x in g) for g in groups], tgt, model_name, device)
    return [p for g, t in zip(groups, tr) for p in split_text(t, [len(x[2]) for x in g])]


def mux(video: Path, srts: dict, out: Path) -> None:
    codec = "mov_text" if out.suffix.lower() in (".mp4", ".m4v", ".mov") else "srt"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-i", str(video)]
    for p in srts.values():
        cmd += ["-i", str(p)]
    cmd += ["-map", "0:v", "-map", "0:a?"]  # niente sub preesistenti: evita conflitti di codec
    for n in range(1, len(srts) + 1):
        cmd += ["-map", str(n)]
    cmd += ["-c", "copy", "-c:s", codec]
    for n, lang in enumerate(srts):
        cmd += [f"-metadata:s:s:{n}", f"language={lang}"]
    subprocess.run(cmd + [str(out)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="video di output (default: <film>.subs.mkv; .mp4 usa mov_text)")
    ap.add_argument("--whisper", default="large-v3", help="modello faster-whisper (es. medium con poca VRAM)")
    ap.add_argument("--nllb", default="facebook/nllb-200-distilled-1.3B", help="modello NLLB (distilled-600M con poca VRAM)")
    ap.add_argument("--no-mux", action="store_true", help="genera solo i file .srt")
    a = ap.parse_args()
    if not a.video.is_file():
        sys.exit(f"file non trovato: {a.video}")

    import ctranslate2

    device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    print(f"device: {device}", file=sys.stderr)

    stem = a.video.with_suffix("")
    ar_srt = Path(f"{stem}.ar.srt")
    if ar_srt.exists():  # riprende dopo un'interruzione: la trascrizione è la parte più lenta
        print(f"uso trascrizione esistente: {ar_srt.name}", file=sys.stderr)
        segs = parse_srt(ar_srt)
    else:
        segs = transcribe(a.video, a.whisper, device)
        write_srt(ar_srt, segs, [t for _, _, t in segs])

    srts = {"ara": ar_srt}
    for code, (iso, nllb) in LANGS.items():
        p = Path(f"{stem}.{code}.srt")
        if not p.exists():  # già tradotta in un run precedente
            write_srt(p, segs, translate_sentences(segs, nllb, a.nllb, device))
        srts[iso] = p
    if not a.no_mux:
        out = a.out or Path(f"{stem}.subs.mkv")
        mux(a.video, srts, out)
        print(f"fatto: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
