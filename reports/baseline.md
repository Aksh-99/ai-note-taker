# Baseline: Whisper model size vs accuracy vs speed

Status: waiting for the first local run. The nightly build sandbox cannot download MIT OCW audio or Whisper weights, so the real numbers have to come from a laptop or a free Colab / Kaggle GPU. Running the commands below overwrites this file with the full report.

## What the report will contain

- One row per Whisper size (tiny, base, small, ...) with parameter count, pooled WER and CER, technical-term recall and precision, real-time factor (RTF), speed as "x real time", and model RAM
- The Pareto frontier: models that no other model beats on both accuracy and speed
- A recommendation: the most accurate model within a speed budget (default RTF <= 0.5, so a 90-minute lecture takes at most 45 minutes)
- "What each size step buys": absolute and relative WER drop vs the extra compute, which shows where returns start to diminish
- WER per lecture, and caveats if models were not measured the same way (different lectures, different `--max-seconds`, different devices)

## How to generate it

```bash
notetaker evalset
notetaker benchmark data/raw/ocw/mit-18.06-lec01/audio.flac -m tiny base small --max-seconds 600 --repeats 3
for m in tiny base small; do
  for d in data/raw/ocw/*/; do
    id=$(basename "$d")
    notetaker transcribe "$d/audio.flac" -m "$m" -l en -o "outputs/eval/whisper-$m/$id"
  done
  notetaker wer   --hyps "outputs/eval/whisper-$m"
  notetaker terms --hyps "outputs/eval/whisper-$m"
done
notetaker baseline            # writes reports/baseline.md and reports/baseline.json
```

On a CPU laptop, add `--max-seconds 600` to `wer` and `terms` (and use the same value for every model) to score only the first 10 minutes of each lecture.
