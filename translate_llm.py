#!/usr/bin/env python3
"""Traduce un .srt arabo (egiziano) in .srt EN e IT con un LLM locale (server OpenAI-compatibile,
LM Studio http://localhost:1234/v1 o llama-server). Dipendenza: solo stdlib.

Uso:  python translate_llm.py film.ar.srt --out-en film.en.srt --out-it film.it.srt
      [--mode direct|pivot] [--model NOME] [--block 8] [--ctx 3] [--title "..."] [--names "Nisa, Zizi"]

Robustezza: output vincolato da JSON schema con minItems=maxItems=N (grammar: il conteggio non puo'
sbagliare); se comunque il conteggio/parsing fallisce -> retry, poi dimezza il blocco, poi riga singola,
poi fallback = testo sorgente (mai righe perse). Le righe ripetute/rumore Whisper restano 1:1.
"""
import argparse, json, re, sys, time, urllib.request

SYS = """You are a professional film subtitler translating an Egyptian Arabic (colloquial, Cairo dialect) movie into {lang}.
Rules:
- Translate each numbered line into natural, colloquial, spoken {lang}, as a subtitle: short, idiomatic, same register and tone as the speaker. Translate idioms by meaning, not word for word.
- The transcript comes from automatic speech recognition and may contain mistakes: use the surrounding lines to infer the most plausible meaning, but never invent content that is not there.
- Keep proper names consistent and transliterate them simply (do not translate names into words).{names}
- Do not add explanations, notes, quotation marks, or speaker labels. Do not merge or split lines: exactly one output string per input line, even for fragments or single words.
- Lines marked as context are only for understanding; do not translate them.
Film: {title}"""


def chat(url, model, system, user, schema=None, temperature=0.1, max_tokens=2048, timeout=300):
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": False}}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": schema}}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse(txt, n):
    """Estrae n stringhe: prima JSON {"t":[...]}, poi righe numerate 'k| testo'. None se il conteggio non torna."""
    try:
        m = re.search(r"\{.*\}|\[.*\]", txt, re.S)
        d = json.loads(m.group(0))
        arr = d["t"] if isinstance(d, dict) else d
        if len(arr) == n and all(isinstance(x, dict) for x in arr):
            # con gli id il modello non può spostare le traduzioni di una riga (provato: senza id succedeva)
            if sorted(x["id"] for x in arr) == list(range(1, n + 1)):
                return [" ".join(x["t"].split()) for x in sorted(arr, key=lambda x: x["id"])]
        elif len(arr) == n and all(isinstance(x, str) for x in arr):
            return [" ".join(x.split()) for x in arr]  # niente newline dentro una riga
    except Exception:
        pass
    got = {}
    for ln in txt.splitlines():
        m = re.match(r"\s*(\d+)\s*[|.):\-]\s*(.*)", ln)
        if m:
            got.setdefault(int(m.group(1)), m.group(2).strip())
    if sorted(got) == list(range(1, n + 1)):
        return [" ".join(got[i].split()) for i in range(1, n + 1)]
    return None


def block(cfg, src_lines, before, after, lang, src_lang, stats):
    n = len(src_lines)
    system = SYS.format(lang=lang, title=cfg["title"], names=(" Known names: %s." % cfg["names"]) if cfg["names"] else "")
    if src_lang != "Arabic":
        system = system.replace("Egyptian Arabic (colloquial, Cairo dialect)", src_lang)
    u = ""
    if before: u += "Context before (do not translate):\n" + "\n".join("- " + x for x in before) + "\n\n"
    u += "Translate these %d lines into %s. Reply JSON {\"t\": [{\"id\": 1, \"t\": \"translation of line 1\"}, ...]}, one object per line, ids 1..%d.\n" % (n, lang, n)
    u += "\n".join("%d| %s" % (i + 1, x) for i, x in enumerate(src_lines))
    if after: u += "\n\nContext after (do not translate):\n" + "\n".join("- " + x for x in after)
    item = {"type": "object", "properties": {"id": {"type": "integer"}, "t": {"type": "string"}},
            "required": ["id", "t"], "additionalProperties": False}
    schema = {"type": "object", "properties": {"t": {"type": "array", "items": item, "minItems": n, "maxItems": n}},
              "required": ["t"], "additionalProperties": False} if cfg["schema"] else None
    for attempt in range(2):
        t0 = time.time()
        try:
            out = chat(cfg["url"], cfg["model"], system, u, schema, 0.1 if attempt == 0 else 0.4, 160 * n + 100)
        except Exception as e:
            stats["errors"] += 1; print("  errore HTTP:", e, file=sys.stderr); time.sleep(2); continue
        stats["calls"] += 1; stats["secs"] += time.time() - t0
        r = parse(out, n)
        if r: return r
        stats["mismatch"] += 1
    if n > 1:  # dimezza
        h = n // 2
        stats["splits"] += 1
        return (block(cfg, src_lines[:h], before, src_lines[h:h + 2], lang, src_lang, stats) +
                block(cfg, src_lines[h:], src_lines[max(0, h - 2):h], after, lang, src_lang, stats))
    stats["fallback"] += 1
    return [src_lines[0]]  # fallback: mai perdere la riga


def translate(cfg, lines, lang, src_lang="Arabic"):
    """lines: lista di str -> lista di str (stessa lunghezza)."""
    out, B, C = [], cfg["block"], cfg["ctx"]
    st = cfg["stats"]
    for i in range(0, len(lines), B):
        res = block(cfg, lines[i:i + B], lines[max(0, i - C):i], lines[i + B:i + B + C], lang, src_lang, st)
        assert len(res) == len(lines[i:i + B])
        out += res
        print("  %d/%d  (%.1fs/riga)" % (min(i + B, len(lines)), len(lines), st["secs"] / max(1, len(out))), file=sys.stderr)
    return out


def read_srt(p):
    txt = open(p, encoding="utf-8-sig").read().replace("\r", "")
    items = []
    for b in re.split(r"\n\s*\n", txt.strip()):
        ls = b.split("\n")
        if len(ls) >= 2 and "-->" in ls[1]:
            items.append((ls[1].strip(), " ".join(ls[2:]).strip()))
        elif ls and "-->" in ls[0]:
            items.append((ls[0].strip(), " ".join(ls[1:]).strip()))
    return items


def write_srt(p, times, texts):
    with open(p, "w", encoding="utf-8") as f:
        for i, (t, x) in enumerate(zip(times, texts), 1):
            f.write("%d\n%s\n%s\n\n" % (i, t, x))


def new_stats(): return dict(calls=0, secs=0.0, mismatch=0, splits=0, fallback=0, errors=0)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("srt"); a.add_argument("--out-en"); a.add_argument("--out-it")
    a.add_argument("--mode", choices=["direct", "pivot"], default="direct")
    a.add_argument("--url", default="http://localhost:1234/v1"); a.add_argument("--model", default="local")
    a.add_argument("--block", type=int, default=8); a.add_argument("--ctx", type=int, default=3)
    a.add_argument("--title", default="an Egyptian Arabic feature film"); a.add_argument("--names", default="")
    a.add_argument("--no-schema", action="store_true", help="senza output vincolato (usa solo parse numerato)")
    x = a.parse_args()
    cfg = dict(url=x.url, model=x.model, block=x.block, ctx=x.ctx, title=x.title, names=x.names,
               schema=not x.no_schema, stats=new_stats())
    items = read_srt(x.srt); times = [t for t, _ in items]; ar = [s for _, s in items]
    en = translate(cfg, ar, "English")
    if x.out_en: write_srt(x.out_en, times, en)
    if x.out_it:
        it = translate(cfg, en, "Italian", "English") if x.mode == "pivot" else translate(cfg, ar, "Italian")
        write_srt(x.out_it, times, it)
    print(json.dumps(cfg["stats"]), file=sys.stderr)


if __name__ == "__main__":
    main()
