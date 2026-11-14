# Agent test fixtures

Used by `tests/test_agent_fixtures.py` to test the full LangGraph pipeline (`notetaker agent`) with no live LLM calls.

| File | What it is |
| --- | --- |
| `transcript.json` | A 3.5 minute lecture on eigenvalues, written for this test. It contains a problem set ("next Wednesday at 5pm"), a reading ("before Monday's class"), a midterm ("November 24th") and a false alarm ("this result is due to Cauchy"). |
| `options.json` | Pipeline settings for the run. `chunk_words` is small (200) so the 372-word lecture splits into 2 chunks and the map-reduce notes path is exercised. The lecture date is Friday 2026-11-13. |
| `cassette.json` | Every LLM request the run makes, with the reply it got. Written by `notetaker.cassette.RecordingClient`, read by `ReplayClient`. |
| `scripted_replies.json` | Hand-written stand-in replies, used to seed the cassette when no real model is reachable. One of them imitates a common small-model mistake (prose before the JSON) so the repair loop is covered. |
| `expected/` | The files the run should write ("golden files"). `pipeline.json` has its timings removed and `tasks.ics` has its DTSTAMP replaced by a fixed value, because both change on every run. |

## How replay matches requests

Each request is identified by a SHA-256 hash of its messages, JSON mode and token limit. Change any prompt and replay fails with a `CassetteMiss` that names the closest recorded request and shows where the text differs. That is on purpose: it means the cassette no longer describes what the code sends.

## Re-recording

The current cassette was seeded from `scripted_replies.json` (its `provider` field says `scripted`). Replacing it with a real recording from a free model makes the test more realistic, because real models format their JSON in messier ways:

```bash
# local Ollama, free and keyless
python scripts/record_agent_fixture.py --provider ollama --model llama3.1:8b
# or Groq's free tier (GROQ_API_KEY in .env)
python scripts/record_agent_fixture.py --provider groq

# then refresh the golden files and review what changed
UPDATE_GOLDEN=1 pytest tests/test_agent_fixtures.py
git diff tests/fixtures/agent/
```

A few assertions in the test (task titles, dates, section headings) describe the scripted replies; after a live re-recording, update them to match what the model said. `make record-fixtures` re-seeds from the scripted replies and refreshes the golden files in one go.

Never commit a cassette recorded from private lecture material: the prompts contain the transcript text.
