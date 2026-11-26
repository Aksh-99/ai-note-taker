"""Tests for the precision tradeoff evaluation (Step 6.4c).

No real model and no torch: models are fakes with faster-whisper's
`.transcribe()` shape, and "export folders" are folders holding the three
files the model-source checks look at. The real end-to-end run on tiny
exported Whispers lives in tests/test_quantize_export.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from notetaker import quantize_eval as qe
from notetaker.audio import save_wav
from notetaker.cli import main
from notetaker.config import get_settings
from notetaker.quantize_eval import (
    ClipInfo,
    ModelRun,
    QuantizeEvalError,
    check_runs,
    drift_rate,
    evaluate_runs,
    load_runs,
    recommend,
    run_model,
    run_quantize_eval,
    save_runs,
    to_markdown,
    write_report,
)
from notetaker.term_recall import make_term

SR = 16_000
TERMS = [make_term("eigenvalue"), make_term("gradient descent")]

# Three clips: (lecture, seconds, caption). Durations differ so a fake model
# can tell clips apart by the length of the waveform it receives.
CLIPS = [
    ("lec-a", 1.0, "the eigenvalue of a matrix"),
    ("lec-a", 1.5, "we run gradient descent"),
    ("lec-b", 2.0, "the loss goes down"),
]


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch):
    for var in ("NOTETAKER_WHISPER_DEVICE", "NOTETAKER_WHISPER_COMPUTE_TYPE"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def clip_infos() -> list[ClipInfo]:
    return [
        ClipInfo(f"{lec}/{lec}_{i:05d}.wav", lec, text, seconds)
        for i, (lec, seconds, text) in enumerate(CLIPS)
    ]


def make_run(label: str, texts: list[str], **overrides) -> ModelRun:
    fields = dict(
        label=label,
        spec=label,
        quantization=None,
        device="cpu",
        compute_type="int8",
        size_mb=100.0,
        load_seconds=1.0,
        transcribe_seconds=1.0,
        audio_seconds=4.5,
        beam_size=5,
        texts=texts,
    )
    fields.update(overrides)
    return ModelRun(**fields)


PERFECT = [text for _, _, text in CLIPS]
ONE_SLIP = ["the eigen value of a matrix", PERFECT[1], PERFECT[2]]


def make_export(root: Path, name: str, quantization: str, mb: int = 1) -> Path:
    """A folder that passes `resolve_model_source` as an export-ct2 output."""
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "model.bin").write_bytes(b"\0" * (mb * 2**20))
    (folder / "config.json").write_text("{}")
    (folder / "export.json").write_text(json.dumps({"quantization": quantization}))
    return folder


def make_dataset(root: Path, split: str = "test") -> Path:
    """Write `<root>/<split>/metadata.jsonl` + WAVs like finetune-data does."""
    split_dir = root / split
    rows = []
    for i, (lec, seconds, text) in enumerate(CLIPS):
        name = f"{lec}/{lec}_{i:05d}.wav"
        t = np.arange(int(seconds * SR)) / SR
        save_wav(0.3 * np.sin(2 * np.pi * 220 * t), split_dir / name, SR)
        rows.append(
            {
                "file_name": name,
                "text": text,
                "lecture_id": lec,
                "start": 0.0,
                "end": seconds,
                "duration": seconds,
            }
        )
    (split_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return root


class FakeModel:
    """Answers by waveform length: {samples: text}. Records every call."""

    def __init__(self, by_length: dict[int, str]):
        self.by_length = by_length
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        text = self.by_length.get(len(audio), "")
        segment = SimpleNamespace(
            id=0,
            start=0.0,
            end=1.0,
            text=" " + text,
            avg_logprob=-0.1,
            no_speech_prob=0.0,
            words=None,
        )
        info = SimpleNamespace(
            language="en", language_probability=1.0, duration=len(audio) / SR
        )
        return iter([segment]), info


def answers(texts: list[str]) -> dict[int, str]:
    return {
        int(seconds * SR): t for (_, seconds, _), t in zip(CLIPS, texts, strict=True)
    }


class FakeClock:
    def __init__(self, ticks):
        self.ticks = list(ticks)

    def __call__(self):
        return self.ticks.pop(0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_evaluate_runs_scores_every_model_against_the_reference() -> None:
    runs = [
        make_run("float32", PERFECT, transcribe_seconds=2.0, size_mb=400.0),
        make_run("int8", ONE_SLIP, transcribe_seconds=1.0, size_mb=100.0),
    ]
    report = evaluate_runs(clip_infos(), runs, TERMS, resamples=200)
    ref, int8 = report.models
    assert report.reference == "float32"
    assert report.clips == 3 and report.reference_words == 13
    assert ref.summary.wer == 0 and ref.vs_reference is None and ref.drift is None
    # "eigen value" for "eigenvalue": 1 substitution + 1 insertion in 13 words.
    assert int8.summary.wer == pytest.approx(2 / 13)
    assert int8.summary.term_recall == pytest.approx(1 / 2)
    assert int8.vs_reference.delta == pytest.approx(2 / 13)
    # Drift is measured against the reference transcripts, which here equal
    # the captions, so it matches the WER.
    assert int8.drift == pytest.approx(2 / 13)
    assert int8.run.rtf == pytest.approx(1.0 / 4.5)


def test_drift_ignores_the_captions() -> None:
    """Two models making the SAME mistake have zero drift from each other."""
    runs = [make_run("float32", ONE_SLIP), make_run("int8", ONE_SLIP)]
    report = evaluate_runs(clip_infos(), runs, TERMS, resamples=50)
    assert report.models[1].summary.wer > 0
    assert report.models[1].drift == 0.0


def test_drift_rate_edge_cases() -> None:
    assert drift_rate(["a b"], ["a c"]) == pytest.approx(0.5)
    # The reference wrote nothing on clip 2: that clip is left out.
    assert drift_rate(["a b", ""], ["a b", "extra words"]) == 0.0
    assert drift_rate([""], [""]) == 0.0
    assert drift_rate([""], ["something"]) == 1.0


def test_clips_with_empty_captions_are_skipped() -> None:
    clips = clip_infos() + [ClipInfo("lec-b/x.wav", "lec-b", "[MUSIC]", 1.0)]
    runs = [make_run("a", PERFECT + ["hi"]), make_run("b", PERFECT + [""])]
    report = evaluate_runs(clips, runs, TERMS, resamples=50)
    assert report.skipped == ["lec-b/x.wav"]
    assert report.clips == 3


@pytest.mark.parametrize(
    ("runs", "message"),
    [
        ([], "No models"),
        ([make_run("a", PERFECT), make_run("a", PERFECT)], "share the label"),
        ([make_run("a", PERFECT[:2])], "2 transcripts for 3 clips"),
    ],
)
def test_check_runs_rejects_bad_input(runs, message) -> None:
    with pytest.raises(QuantizeEvalError, match=message):
        check_runs(clip_infos(), runs)


def test_check_runs_needs_clips() -> None:
    with pytest.raises(QuantizeEvalError, match="No clips"):
        check_runs([], [make_run("a", [])])


def test_nothing_to_score_raises() -> None:
    clips = [ClipInfo("x.wav", "l", "[MUSIC]", 1.0)]
    with pytest.raises(QuantizeEvalError, match="nothing to score"):
        evaluate_runs(clips, [make_run("a", ["x"])], TERMS)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def test_recommend_picks_the_fastest_model_that_is_not_worse() -> None:
    runs = [
        make_run("float32", PERFECT, transcribe_seconds=3.0),
        make_run("float16", PERFECT, transcribe_seconds=2.0),
        make_run("int8", PERFECT, transcribe_seconds=1.0),
    ]
    assert recommend(evaluate_runs(clip_infos(), runs, TERMS, resamples=50)) == "int8"


def test_recommend_skips_a_faster_but_clearly_worse_model() -> None:
    garbage = ["nothing like it", "totally wrong words", "no match here at all"]
    runs = [
        make_run("float32", PERFECT, transcribe_seconds=3.0),
        make_run("int8", garbage, transcribe_seconds=1.0),
    ]
    report = evaluate_runs(clip_infos(), runs, TERMS, resamples=200)
    assert report.models[1].vs_reference.low > 0
    assert recommend(report) == "float32"


def test_recommend_breaks_speed_ties_by_size() -> None:
    runs = [
        make_run("float32", PERFECT, size_mb=400.0),
        make_run("int8", PERFECT, size_mb=100.0),
    ]
    assert recommend(evaluate_runs(clip_infos(), runs, TERMS, resamples=50)) == "int8"


# ---------------------------------------------------------------------------
# Saving, loading, reports
# ---------------------------------------------------------------------------


def test_runs_round_trip(tmp_path: Path) -> None:
    runs = [make_run("float32", PERFECT), make_run("int8", ONE_SLIP)]
    path = save_runs(clip_infos(), runs, tmp_path / "sub" / "runs.json", split="val")
    split, clips, loaded = load_runs(path)
    assert split == "val"
    assert clips == clip_infos()
    assert [r.texts for r in loaded] == [PERFECT, ONE_SLIP]


def test_load_runs_errors(tmp_path: Path) -> None:
    with pytest.raises(QuantizeEvalError, match="No runs file"):
        load_runs(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(QuantizeEvalError, match="not a quantize-eval runs file"):
        load_runs(bad)


def test_markdown_and_json_report(tmp_path: Path) -> None:
    runs = [
        make_run("float32", PERFECT, transcribe_seconds=2.0, size_mb=400.0),
        make_run("int8", ONE_SLIP, transcribe_seconds=1.0, size_mb=100.0),
    ]
    report = evaluate_runs(clip_infos(), runs, TERMS, resamples=50)
    text = to_markdown(report)
    assert "| Model | Stored | Compute | Size (MB) | RTF |" in text
    assert "| float32 |" in text and "reference" in text
    assert "`int8`: 2.00x the speed, 4.0x smaller" in text
    assert "**Pick:**" in text
    assert "\u2014" not in text  # house style: no em dashes
    json_path, md_path = write_report(report, tmp_path / "out")
    data = json.loads(json_path.read_text())
    assert md_path.read_text() == text
    assert [m["label"] for m in data["models"]] == ["float32", "int8"]
    assert "texts" not in data["models"][0]
    assert data["models"][1]["wer_change_vs_reference"]["delta"] > 0
    assert data["recommended"] in {"float32", "int8"}


def test_markdown_with_one_model() -> None:
    report = evaluate_runs(clip_infos(), [make_run("small", PERFECT)], TERMS)
    assert "nothing to compare" in to_markdown(report)


# ---------------------------------------------------------------------------
# Running models (fake loader)
# ---------------------------------------------------------------------------


def test_run_model_times_clips_after_an_untimed_warmup(tmp_path: Path) -> None:
    folder = make_export(tmp_path, "whisper-ct2-int8", "int8")
    clips = clip_infos()
    audio = [np.zeros(int(c.duration * SR), dtype=np.float32) for c in clips]
    fake = FakeModel(answers(ONE_SLIP))
    seen = {}

    def loader(spec, device, compute_type):
        seen.update(spec=spec, device=device, compute_type=compute_type)
        return fake

    # Clock ticks: load start/end, then start/end for each of the 3 clips.
    clock = FakeClock([0.0, 2.0, 10.0, 10.5, 20.0, 20.25, 30.0, 30.25])
    run = run_model(
        str(folder),
        clips,
        audio,
        device="cpu",
        beam_size=1,
        loader=loader,
        clock=clock,
    )
    assert seen == {"spec": str(folder), "device": "cpu", "compute_type": "int8"}
    assert run.label == "whisper-ct2-int8" and run.quantization == "int8"
    assert run.load_seconds == 2.0
    assert run.transcribe_seconds == pytest.approx(1.0)
    assert run.rtf == pytest.approx(1.0 / 4.5)
    assert run.texts == ONE_SLIP
    assert run.size_mb == pytest.approx(1.0, abs=0.01)
    # 1 warm-up + 3 timed calls, all with the same decoding settings.
    assert len(fake.calls) == 4
    assert all(c["beam_size"] == 1 and not c["word_timestamps"] for c in fake.calls)


def test_run_model_auto_compute_follows_storage(tmp_path: Path) -> None:
    folder = make_export(tmp_path, "ct2-float32", "float32")
    seen = []

    def loader(spec, device, compute_type):
        seen.append(compute_type)
        return FakeModel({})

    run_model(str(folder), [], [], device="cpu", loader=loader)
    run_model(str(folder), [], [], device="cpu", compute_type="int8", loader=loader)
    assert seen == ["float32", "int8"]


def test_run_quantize_eval_end_to_end(tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "finetune")
    f32 = make_export(tmp_path, "ct2-float32", "float32", mb=4)
    int8 = make_export(tmp_path, "ct2-int8", "int8", mb=1)
    outputs = {str(f32): PERFECT, str(int8): ONE_SLIP}

    def loader(spec, device, compute_type):
        return FakeModel(answers(outputs[spec]))

    runs_path = tmp_path / "runs.json"
    report = run_quantize_eval(
        [str(f32), str(int8)],
        data,
        TERMS,
        runs_path=runs_path,
        device="cpu",
        resamples=50,
        loader=loader,
        log=lambda _: None,
    )
    assert [m.run.label for m in report.models] == ["ct2-float32", "ct2-int8"]
    assert [m.run.compute_type for m in report.models] == ["float32", "int8"]
    assert report.models[1].summary.wer == pytest.approx(2 / 13)
    assert report.models[0].run.size_mb > report.models[1].run.size_mb
    # The saved runs re-score to the same numbers without any model.
    split, clips, runs = load_runs(runs_path)
    again = evaluate_runs(clips, runs, TERMS, split=split, resamples=50)
    assert again.models[1].summary.wer == report.models[1].summary.wer


def test_run_quantize_eval_limit_and_errors(tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "finetune")
    a = make_export(tmp_path / "x", "same", "int8")
    b = make_export(tmp_path / "y", "same", "float32")
    with pytest.raises(QuantizeEvalError, match="same label"):
        run_quantize_eval([str(a), str(b)], data, TERMS, log=lambda _: None)
    with pytest.raises(QuantizeEvalError, match="at least one model"):
        run_quantize_eval([], data, TERMS, log=lambda _: None)
    report = run_quantize_eval(
        [str(a)],
        data,
        TERMS,
        runs_path=tmp_path / "r.json",
        limit=1,
        device="cpu",
        loader=lambda *_: FakeModel(answers(PERFECT)),
        log=lambda _: None,
    )
    assert report.clips == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_full_run_then_rescore(tmp_path, monkeypatch, capsys) -> None:
    data = make_dataset(tmp_path / "finetune")
    f32 = make_export(tmp_path, "ct2-float32", "float32", mb=4)
    int8 = make_export(tmp_path, "ct2-int8", "int8", mb=1)
    outputs = {str(f32): PERFECT, str(int8): ONE_SLIP}
    monkeypatch.setattr(
        qe, "_default_loader", lambda spec, d, c: FakeModel(answers(outputs[spec]))
    )
    out = tmp_path / "reports"
    code = main(
        [
            "quantize-eval",
            "-m",
            str(f32),
            str(int8),
            "--data-dir",
            str(data),
            "--device",
            "cpu",
            "--bootstrap",
            "50",
            "-o",
            str(out),
        ]
    )
    text = capsys.readouterr().out
    assert code == 0, text
    assert "ct2-int8: WER 15.4%" in text and "Pick:" in text
    assert (out / "quantization.md").is_file()
    assert (out / "quantization_runs.json").is_file()

    (out / "quantization.md").unlink()
    code = main(
        [
            "quantize-eval",
            "--from-runs",
            str(out / "quantization_runs.json"),
            "--limit",
            "2",
            "-o",
            str(out),
        ]
    )
    text = capsys.readouterr().out
    assert code == 0, text
    assert (out / "quantization.md").is_file()
    assert "2 clips" in (out / "quantization.md").read_text()


def test_cli_errors(tmp_path, capsys) -> None:
    assert main(["quantize-eval"]) == 1
    assert "give the models" in capsys.readouterr().out
    assert main(["quantize-eval", "--from-runs", "x.json", "--limit", "0"]) == 1
    assert main(["quantize-eval", "-m", "small", "--beam-size", "0"]) == 1
    assert main(["quantize-eval", "--from-runs", str(tmp_path / "none.json")]) == 1
    assert "No runs file" in capsys.readouterr().out
    # A folder that is not a model is a usage error from argparse (exit 2).
    with pytest.raises(SystemExit) as exit_info:
        main(["quantize-eval", "-m", str(tmp_path)])
    assert exit_info.value.code == 2
