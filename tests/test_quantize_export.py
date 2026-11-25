"""End-to-end tests of the CTranslate2 export (Step 6.4a) on a tiny Whisper.

Skipped unless the [finetune] extra (torch, transformers, peft) is installed.
Nothing is downloaded. Unlike the training tests, the converter and
faster-whisper need a *real* Hugging Face tokenizer (they read its
vocabulary and special tokens), so we build a tiny one here: the 256
byte-level symbols plus Whisper's special tokens, no merges. Real Whisper
uses the same layout, just with 50k learned tokens.

What we check:
    - merging the adapter changes nothing about the model's outputs
      (W + (alpha / r) B A gives the same logits as the adapter wrapper)
    - the float32 CTranslate2 model computes the same encoder output and the
      same greedy tokens as the merged PyTorch model
    - int8 shrinks the weights and stays close to float32
    - every storage type converts, and the folder loads in faster-whisper
    - folder rules (force, keep_merged) and the CLI
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
ctranslate2 = pytest.importorskip("ctranslate2")
pytest.importorskip("faster_whisper")

from notetaker import finetune  # noqa: E402  (imports must follow the skips)
from notetaker.cli import main  # noqa: E402
from notetaker.finetune import FinetuneConfig, FinetuneError  # noqa: E402
from notetaker.quantize import (  # noqa: E402
    EXPORT_SUMMARY,
    QUANTIZATIONS,
    convert_to_ct2,
    export_ct2,
    save_merged,
    verify_ct2,
)

SPECIAL = [
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|en|>",
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nocaptions|>",
    # Must be the last special token: faster-whisper treats every id after
    # it as a timestamp token, like real Whisper's vocabulary.
    "<|notimestamps|>",
]
VOCAB_SIZE = 300  # 256 bytes + 9 special tokens + 35 timestamp tokens


def tiny_tokenizer(folder: Path):
    """A real WhisperTokenizerFast with a byte-level vocabulary and no merges."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    vocab = {ch: i for i, ch in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    for token in SPECIAL:
        vocab[token] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken(t, special=True) for t in SPECIAL])
    folder.mkdir(parents=True, exist_ok=True)
    tok.save(str(folder / "tokenizer.json"))
    return transformers.WhisperTokenizerFast(
        tokenizer_file=str(folder / "tokenizer.json"),
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=SPECIAL[1:],
    )


def token_id(name: str) -> int:
    return 256 + SPECIAL.index(name)


def tiny_whisper():
    config = transformers.WhisperConfig(
        vocab_size=VOCAB_SIZE,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        num_mel_bins=80,
        max_source_positions=1500,
        # faster-whisper decodes up to 448 tokens, so give the decoder that
        # many positions (the training tests get away with 64).
        max_target_positions=448,
        pad_token_id=token_id("<|endoftext|>"),
        bos_token_id=token_id("<|endoftext|>"),
        eos_token_id=token_id("<|endoftext|>"),
        decoder_start_token_id=token_id("<|startoftranscript|>"),
        suppress_tokens=[],
        begin_suppress_tokens=[],
    )
    torch.manual_seed(0)
    return transformers.WhisperForConditionalGeneration(config).eval()


@pytest.fixture
def loader(tmp_path: Path):
    """A `load_model` hook returning (tiny Whisper, feature extractor, tokenizer)."""
    tokenizer = tiny_tokenizer(tmp_path / "tok")

    def load(config: FinetuneConfig):
        return tiny_whisper(), transformers.WhisperFeatureExtractor(), tokenizer

    return load


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """What `notetaker finetune` leaves behind: finetune.json + adapter/.

    Instead of training, we set the LoRA B matrices to random values. B
    starts at zero, which would make the adapter a no-op and the merge test
    meaningless.
    """
    config = FinetuneConfig(lora_r=4, lora_alpha=8, lora_dropout=0.0)
    model = finetune.apply_lora(tiny_whisper(), config)
    torch.manual_seed(1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.5)
    out = tmp_path / "whisper-tiny-lora"
    model.save_pretrained(str(out / "adapter"))
    (out / "finetune.json").write_text(json.dumps({"config": config.to_json()}))
    return out


def features(seed: int = 0) -> np.ndarray:
    """Log-mel features of 2 s of quiet random noise, shape (1, 80, 3000)."""
    audio = np.random.RandomState(seed).randn(32_000).astype(np.float32) * 0.1
    extractor = transformers.WhisperFeatureExtractor()
    out = extractor(audio, sampling_rate=16_000, return_tensors="np")
    return out.input_features.astype(np.float32)


PROMPT = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]


def test_merged_model_matches_the_adapter_model(
    model_dir: Path, loader, tmp_path: Path
) -> None:
    peft_model, *_ = finetune.load_finetuned(model_dir, device="cpu", load_model=loader)
    merged_dir = tmp_path / "merged"
    save_merged(model_dir, merged_dir, load_model=loader)
    for name in ("config.json", "tokenizer.json", "preprocessor_config.json"):
        assert (merged_dir / name).is_file()
    merged = transformers.WhisperForConditionalGeneration.from_pretrained(
        str(merged_dir)
    ).eval()
    # A plain Whisper again: no LoRA layers left after merging.
    assert not any("lora" in name for name, _ in merged.named_parameters())

    inputs = {
        "input_features": torch.from_numpy(features()),
        "decoder_input_ids": torch.tensor([[token_id(t) for t in PROMPT]]),
    }
    with torch.no_grad():
        expected = peft_model(**inputs).logits
        got = merged(**inputs).logits
        with peft_model.disable_adapter():
            base = peft_model(**inputs).logits
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
    # And the adapter really did something, so the check above means something.
    assert (base - expected).abs().max() > 1e-2


def test_float32_ct2_matches_pytorch(model_dir: Path, loader, tmp_path: Path) -> None:
    merged_dir = tmp_path / "merged"
    save_merged(model_dir, merged_dir, load_model=loader)
    ct2_dir = convert_to_ct2(merged_dir, tmp_path / "ct2", "float32")
    merged = transformers.WhisperForConditionalGeneration.from_pretrained(
        str(merged_dir)
    ).eval()
    feats = features()
    ct2_model = ctranslate2.models.Whisper(str(ct2_dir), device="cpu")
    view = ctranslate2.StorageView.from_array(feats)

    # Encoder: same numbers up to float rounding.
    ct2_enc = np.array(ct2_model.encode(view, to_cpu=True))
    with torch.no_grad():
        pt_enc = merged.model.encoder(torch.from_numpy(feats)).last_hidden_state
    np.testing.assert_allclose(ct2_enc, pt_enc.numpy(), atol=1e-4)

    # Decoder: greedy tokens agree. The PyTorch side is a hand-written
    # greedy loop (argmax of the last logits, append, repeat) so no Hugging
    # Face generation rules get in the way.
    prompt = [token_id(t) for t in PROMPT]
    result = ct2_model.generate(
        view,
        [prompt],
        beam_size=1,
        max_length=12,
        suppress_blank=False,
        suppress_tokens=[],
    )
    ct2_tokens = result[0].sequences_ids[0]
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(len(ct2_tokens)):
            logits = merged(
                input_features=torch.from_numpy(feats),
                decoder_input_ids=torch.tensor([ids]),
            ).logits
            ids.append(int(logits[0, -1].argmax()))
    assert len(ct2_tokens) >= 3
    assert ct2_tokens == ids[len(prompt) :]


def test_int8_is_smaller_and_close_to_float32(
    model_dir: Path, loader, tmp_path: Path
) -> None:
    merged_dir = tmp_path / "merged"
    save_merged(model_dir, merged_dir, load_model=loader)
    f32 = convert_to_ct2(merged_dir, tmp_path / "f32", "float32")
    i8 = convert_to_ct2(merged_dir, tmp_path / "i8", "int8")
    size = {d: (d / "model.bin").stat().st_size for d in (f32, i8)}
    # Real models shrink about 4x; this tiny one has relatively more
    # unquantized parts (biases, layer norms, positions), so ask for 1.2x.
    assert size[i8] * 1.2 < size[f32]

    view = ctranslate2.StorageView.from_array(features())
    outs = {}
    for name, folder, compute in (("f32", f32, "float32"), ("i8", i8, "int8")):
        model = ctranslate2.models.Whisper(str(folder), compute_type=compute)
        outs[name] = np.array(model.encode(view, to_cpu=True)).ravel()
    # Rounding each weight to one of 255 levels moves the output a little
    # but keeps it almost perfectly correlated with the float32 output.
    assert not np.allclose(outs["f32"], outs["i8"], atol=1e-6)
    assert np.corrcoef(outs["f32"], outs["i8"])[0, 1] > 0.99


@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_every_storage_type_exports_and_loads(
    quantization: str, model_dir: Path, loader, tmp_path: Path
) -> None:
    out = tmp_path / f"ct2-{quantization}"
    result = export_ct2(
        model_dir, out, quantization=quantization, load_model=loader, log=lambda _: None
    )
    for name in (
        "model.bin",
        "config.json",
        "vocabulary.json",
        "tokenizer.json",
        "preprocessor_config.json",
        EXPORT_SUMMARY,
    ):
        assert (out / name).is_file(), name
    assert result.verified
    assert result.quantization == quantization
    assert result.base_model == "openai/whisper-small"  # from finetune.json
    saved = json.loads((out / EXPORT_SUMMARY).read_text())
    assert saved["quantization"] == quantization
    assert saved["ct2_mb"] > 0 and saved["merged_mb"] > 0
    # The temporary merged folder is gone; nothing else was left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [out.name, model_dir.name, "tok"]
    )


def test_export_runs_in_faster_whisper(model_dir: Path, loader, tmp_path: Path) -> None:
    from faster_whisper import WhisperModel

    out = tmp_path / "ct2"
    export_ct2(model_dir, out, load_model=loader, verify=False, log=lambda _: None)
    assert verify_ct2(out) >= 0
    # Full faster-whisper path with word timestamps, as `transcribe` uses it.
    model = WhisperModel(str(out), device="cpu", compute_type="int8")
    audio = np.random.RandomState(0).randn(48_000).astype(np.float32) * 0.1
    segments, info = model.transcribe(
        audio, language="en", beam_size=1, word_timestamps=True
    )
    segments = list(segments)
    assert info.language == "en"
    assert all(s.end >= s.start for s in segments)


def test_force_replaces_a_previous_export_and_keep_merged(
    model_dir: Path, loader, tmp_path: Path
) -> None:
    out = tmp_path / "ct2"
    export_ct2(model_dir, out, load_model=loader, verify=False, log=lambda _: None)
    with pytest.raises(FinetuneError, match="force=True"):
        export_ct2(model_dir, out, load_model=loader, verify=False)
    keep = tmp_path / "merged"
    result = export_ct2(
        model_dir,
        out,
        quantization="float32",
        force=True,
        verify=False,
        keep_merged=keep,
        load_model=loader,
        log=lambda _: None,
    )
    assert result.quantization == "float32" and not result.verified
    assert (keep / "model.safetensors").is_file()
    assert json.loads((out / EXPORT_SUMMARY).read_text())["quantization"] == "float32"


def test_cli_exports(
    model_dir: Path,
    loader,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(finetune, "_load_pretrained", loader)
    code = main(["export-ct2", "--model-dir", str(model_dir)])
    text = capsys.readouterr().out
    assert code == 0, text
    out = tmp_path / "whisper-tiny-lora-ct2-int8"
    assert (out / "model.bin").is_file()
    assert "smaller" in text and "export.json" in text


def test_exports_are_usable_as_a_model_everywhere(
    model_dir: Path, loader, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 6.4b: real exports work as `transcribe` and `benchmark` models.

    No fakes below the CLI: `notetaker.transcribe.load_model` hands the
    folder to the real faster-whisper, and "auto" picks the compute type
    from export.json (int8 for the int8 export, float32 for float32).
    """
    from notetaker import transcribe as tr
    from notetaker.benchmark import benchmark_model
    from notetaker.config import get_settings

    for var in ("NOTETAKER_WHISPER_DEVICE", "NOTETAKER_WHISPER_COMPUTE_TYPE"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    tr.load_model.cache_clear()
    exports = {}
    for quantization in ("int8", "float32"):
        out = tmp_path / f"ct2-{quantization}"
        export_ct2(
            model_dir,
            out,
            quantization=quantization,
            load_model=loader,
            verify=False,
            log=lambda _: None,
        )
        exports[quantization] = out

    audio = np.random.RandomState(0).randn(32_000).astype(np.float32) * 0.1
    transcript = tr.transcribe(
        audio, model_name=str(exports["int8"]), language="en", beam_size=1
    )
    assert transcript.model == "ct2-int8"
    assert transcript.duration == pytest.approx(2.0, abs=0.01)

    results = {
        q: benchmark_model(audio, str(path), device="cpu", language="en", beam_size=1)
        for q, path in exports.items()
    }
    assert results["int8"].compute_type == "int8"
    assert results["float32"].compute_type == "float32"
    assert results["int8"].quantization == "int8"
    # int8 weights take about a quarter of the space of float32 weights (the
    # tokenizer and config files are the same size in both, so less than 4x
    # overall on a model this tiny, but clearly smaller).
    assert results["int8"].model_size_mb < results["float32"].model_size_mb
    tr.load_model.cache_clear()
    get_settings.cache_clear()
