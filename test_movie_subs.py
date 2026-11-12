import tempfile
from pathlib import Path

from movie_subs import chunk_words, group_sentences, parse_srt, split_text, ts, write_srt


def test_group_sentences_and_split_text():
    segs = [(0, 1, "ab"), (1, 2, "cd."), (2, 3, "ef"), (6, 7, "gh")]  # l'ultimo dopo una pausa lunga
    assert [len(g) for g in group_sentences(segs)] == [2, 1, 1]
    assert split_text("one two three four", [2, 2]) == ["one two", "three four"]
    assert split_text("one two three", [1, 1, 1]) == ["one", "two", "three"]
    assert split_text("hi", [1, 1]) == ["", "hi"]  # meno parole che pezzi: nessun crash


def test_chunk_words_splits_on_punct_gap_and_duration():
    w = lambda s, e, t: (s, e, t)
    assert chunk_words([w(0, 1, " a"), w(1, 2, " b."), w(2, 3, " c")]) == [(0, 2, "a b."), (2, 3, "c")]
    assert len(chunk_words([w(0, 1, " a"), w(3, 4, " b")])) == 2  # pausa > 0.8s
    long = [w(i, i + 1, " x") for i in range(12)]  # 12s di parlato continuo
    assert all(e - s <= 5.0 for s, e, _ in chunk_words(long))


def test_ts():
    assert ts(3661.5) == "01:01:01,500"
    assert ts(0) == "00:00:00,000"


def test_srt_roundtrip_skips_empty_and_renumbers():
    segs = [(1.0, 2.5, "a"), (3.0, 4.0, "b"), (5.0, 6.0, "c")]
    p = Path(tempfile.mkdtemp()) / "x.srt"
    write_srt(p, segs, ["مرحبا", "", "ciao"])
    assert p.read_text(encoding="utf-8").startswith("1\n00:00:01,000 --> 00:00:02,500\nمرحبا")
    assert parse_srt(p) == [(1.0, 2.5, "مرحبا"), (5.0, 6.0, "ciao")]
