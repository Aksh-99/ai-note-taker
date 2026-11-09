# Dev Log

One entry per work session, newest first. Each entry covers what was built, how it works, and what's next.

---

## 2026-11-08 · Step 3.6: Quiz generation (multiple choice with distractors)

**What I did**
- Added `src/notetaker/quiz.py`: one LLM call per transcript chunk asks for up to 4 multiple-choice questions as JSON, each with the correct answer, 3 distractors and a one-sentence explanation as separate fields
- Replies are validated with pydantic (`QuestionSchema`): at least 2 distractors (extras are cut to 3), questions at most 50 words, options at most 30, all options different after normalization (so "Gradient." and "the gradients" count as the same option), and no "all/none of the above" options. Invalid replies go through the usual repair loop, and common small-model variants (bare list, `answer`/`wrong_answers`/`stem` keys) are accepted
- Grounding check reused from flashcards: at least half of the correct answer's content words must occur in the chunk, otherwise the question is dropped and listed in `dropped_ungrounded`
- Options are shuffled in code with a seeded random generator (`--seed`, per-question seed from a CRC32 checksum of the question), so the same input always gives the same quiz but the answer is spread evenly over A to D
- `--concepts concepts.json` adds "Which term matches this definition?" questions with no LLM call: the term is masked out of its own definition, and the distractors are the 3 glossary terms first mentioned closest in time to it
- Two quality signals on the `Quiz` model: `answer_positions()` (how often the key is at each letter) and `longest_option_rate()` (share of questions where the key is strictly the longest option), printed by the CLI
- Output `quiz.json` + `quiz.md` (questions first, answer key with explanations and timestamps at the end), re-validated by `load_quiz`, with its JSON Schema in `schemas/quiz.schema.json` and a drift test
- New CLI command `notetaker quiz` (`-o`, `--title`, `--concepts`, `--concepts-only`, `--max-questions`, `--seed`, `-p`, `-m`, `--chunk-words`, `--max-repairs`, `--schema`) and a README section
- 38 new tests in `tests/test_quiz.py`, all offline with `FakeLLMClient`. `make check` passes (529 tests, lint clean)

**How it works**
A multiple-choice item has a stem (the question), a key (the right answer) and distractors (plausible wrong answers). I ask the model for the key and the distractors separately instead of a list of options plus an index, for two reasons: small models often get the index wrong, and LLMs have a known position bias, putting the right answer first far more than a quarter of the time. So the code assembles and shuffles the options itself. The shuffle uses a seeded random generator so results are reproducible; the seed mixes the global seed with a CRC32 checksum of the question, because Python's built-in `hash()` of a string changes every time Python starts. A test checks that over 400 questions the key lands on each of the 4 positions roughly 100 times. For the glossary questions, good distractors come for free: other terms from the same lecture are real concepts the student heard, and terms introduced close together in time tend to be related, which makes them harder to tell apart. Finally, I measure a classic test-writing flaw: the correct option is often the longest, since writers add qualifiers to make it fully true. With 4 options a fair quiz has the key as the longest option about 25% of the time, so a much higher rate is a sign the prompt needs better distractors. Whether the questions are faithful to the lecture is checked properly by the LLM-as-judge eval in the next step.

**Next up**
Step 3.7: Notes quality eval: faithfulness check against transcript (LLM-as-judge with a free model).

---

## 2026-11-07 · Step 3.5: Flashcard generation (Q/A pairs), exportable to Anki CSV

**What I did**
- Added `src/notetaker/flashcards.py`: one LLM call per transcript chunk asks for up to 6 question/answer cards as JSON, validated with a pydantic schema (`CardSchema`: non-empty fields, questions at most 40 words, answers at most 60). Common small-model variants are accepted (a bare list, `"flashcards"` instead of `"cards"`, `Q`/`A` or `front`/`back` keys) and invalid replies go through the same repair loop as notes and concepts
- Grounding check: at least half of the answer's distinct content words (stopwords removed, plurals folded) must occur in the chunk the model read. Cards that fail are dropped and listed in `dropped_ungrounded`
- Timestamps come from the transcript: each card gets the start of the chunk paragraph that shares the most content words with it
- `--concepts concepts.json` turns every glossary entry from Step 3.4 into a "What is X?" card with no LLM call, and `--concepts-only` builds a whole deck that way for free
- De-duplication: questions are compared on their content words with a word-level similarity ratio (threshold 0.9), so "What is an eigenvalue?" and "What is the eigenvalue?" collapse into one card, while "not" is kept as a content word so opposite questions stay apart. Concept cards win ties; the deck is sorted by lecture time
- Anki export: `flashcards.csv` has Front, Back and Tags columns, written with Python's `csv` module so commas, quotes and newlines are escaped correctly. Anki file headers (`#separator`, `#deck`, `#notetype`, `#tags column`) make the import zero-setup; `--plain-csv` drops them for older Anki. The back of each card ends with its timestamp, and tags are the deck name (made space-free) plus the card origin (`llm` or `concept`)
- Output is a pydantic model (`FlashcardDeck`) written as `flashcards.json` + `flashcards.md`, re-validated by `load_flashcards`, with its JSON Schema committed in `schemas/flashcards.schema.json` and a drift test
- New CLI command `notetaker flashcards` (`-o`, `--title`, `--deck`, `--concepts`, `--concepts-only`, `--max-cards`, `--plain-csv`, `-p`, `-m`, `--chunk-words`, `--max-repairs`, `--schema`) and a README section
- 35 new tests in `tests/test_flashcards.py`, all offline with `FakeLLMClient`. `make check` passes (491 tests, lint clean)

**How it works**
The model writes the questions and answers, and code checks everything it can. Checking that an answer is faithful properly needs another model (that is Step 3.7), but a cheap proxy catches the worst failures: vocabulary overlap. I strip the answer down to its content words and compute the share of them that appear in the transcript chunk, a set-overlap score between 0 and 1. If the lecturer never used most of those words, the answer probably came from the model's own knowledge, so the card is dropped. The same content-word sets solve two more problems. Finding the timestamp is a tiny retrieval task: score each paragraph by how many words it shares with the card and take the best one. Finding duplicates is a similarity task: compare the content words of two questions with a ratio of 2 x matching words / total words, which ignores filler like "an" versus "the". For the export, Anki is the standard free spaced-repetition app, and its importer reads plain CSV with a few optional header lines, so the file opens straight into the right deck with tags attached.

**Next up**
Step 3.6: Quiz generation (multiple choice with distractors).

---

## 2026-11-06 · Step 3.4: Key concepts + definitions extraction as JSON with schema validation

**What I did**
- Added `src/notetaker/concepts.py`: one LLM call per transcript chunk asks for the technical terms explained in it, each with a definition and optional aliases, as JSON
- Replies are validated with a pydantic schema (`ChunkConceptsSchema`): non-empty term and definition, a term of at most 8 words, aliases cleaned to a list. An empty list is allowed (a chunk of logistics has no concepts). A bare list or a flat `{"term": "definition"}` object is accepted too, and invalid replies go through the same one-shot repair loop as the notes
- Grounding check: a term (or one of its aliases) must occur in the chunk the model read, otherwise it is dropped and listed in `dropped_ungrounded`
- Merging: the same concept from several chunks becomes one entry, matched on any shared name after normalization (lowercase, punctuation removed, plurals folded), so "SVD" merges with "singular value decomposition (aliases: SVD)". The first wording wins; differently worded later definitions are kept in `other_definitions`, near copies are removed with `dedupe_points` from Step 3.3
- Timestamps come from the transcript, not the model: `mentions` lists the start of every paragraph that uses the term, and `first_mention` is the earliest
- The output file is also a pydantic model (`ConceptList`): `write_concepts` writes `concepts.json` + `concepts.md` (a glossary with timestamps), `load_concepts` re-validates the file on read, and its JSON Schema is committed in `schemas/concepts.schema.json` with a test that fails if the code and the schema file drift apart
- New CLI command `notetaker concepts transcript.json` (`-o`, `--title`, `-p`, `-m`, `--chunk-words`, `--max-repairs`) and `notetaker concepts --schema`
- 29 new tests in `tests/test_concepts.py`, all with `FakeLLMClient` (no live LLM calls). `make check` passes (456 tests, lint clean)

**How it works**
The model is used only for the part that needs language understanding: deciding which phrases are concepts and paraphrasing how the lecturer defines them. Anything that can be checked is checked in code. Schema validation guards the shape of the data twice: on the way in (is the model reply usable?) and on the way out (is this concepts.json file trustworthy for the flashcard and quiz steps?). Pydantic lets me declare the rules once and get both a validator and a standard JSON Schema from the same class. To catch hallucinated terms, each term is normalized into a tuple of words and searched for as a consecutive phrase in the chunk text, with a sliding window. The plural folding is deliberately crude (drop a final "s", "ies" to "y") because a full stemmer would also merge different words like "general" and "generate". The same normalized word tuples are used as keys to merge concepts across chunks and to find every paragraph that mentions a term, which gives an index from concept to timestamps for free. Whether the definitions are faithful is a harder question and is left to the LLM-as-judge eval in Step 3.7.

**Next up**
Step 3.5: Flashcard generation (Q/A pairs), exportable to Anki CSV.

---

## 2026-11-05 · Step 3.3: Map-reduce summarization for long lectures

**What I did**
- Added `src/notetaker/mapreduce.py`, the "reduce" half of map-reduce: it merges the per-chunk sections from Step 3.2 into the lecture's main topics and writes a 2 to 4 sentence overview plus 3 to 6 key takeaways
- The model sees the sections as a numbered list (`[3] (00:16:02) Heading: summary; point; point`) and answers with JSON that groups section ids into topics, each with a new heading and summary. Ids are checked in code: an unknown id or an id used twice triggers the repair loop; a forgotten id is attached to the topic of the section just before it, so every section lands in exactly one topic
- Facts are copied, not generated: a topic's timestamp is the earliest start of its sections and its key points are their key points, with near-duplicates removed by word-level similarity (`dedupe_points`, threshold 0.9)
- Hierarchical reduce: if the list is longer than the word budget (default 2,500 words), it is packed into consecutive batches, each batch is merged, and the shorter list is merged again (at most 3 levels, and it stops early if a level merges nothing). Only the final call asks for the overview
- `Notes` gained `overview`, `takeaways`, `detail_sections` (the per-chunk sections before merging, kept for the Step 3.7 faithfulness eval), `reduce_calls`, `reduce_levels` and `reduce_error`; `NoteSection` gained `sources`. Old `notes.json` files still load. `notes.md` now shows the overview at the top and the takeaways at the end
- If the reduce step fails (bad JSON after repairs, or an API error) the per-chunk notes are kept and the CLI prints a warning instead of failing
- `notetaker notes` now merges by default; new flags `--no-reduce` and `--reduce-words`
- 39 new tests in `tests/test_mapreduce.py` with fake clients, including one that answers any reduce prompt by pairing ids so the multi-level path is tested on 24 sections. `make check` passes (427 tests, lint clean)

**How it works**
Summarizing a long lecture in one call does not work with small free models, so it is split like a classic MapReduce job: "map" summarizes each chunk independently, "reduce" combines those partial results. The key design choice is what the reduce call is allowed to write. If the model rewrote all the notes, it could drop details or invent new ones, and every merge would be a new chance to hallucinate. Instead the model does only the part that needs language understanding (deciding which sections belong together and naming the topic) while code does the bookkeeping: timestamps, key points and coverage. That makes the output checkable: every final topic lists the `sources` it came from, and a test asserts that every map section appears in exactly one topic. Duplicate key points across neighbouring chunks are removed with a similarity ratio, 2 x matching words / total words, on lowercase words. Comparing words instead of characters matters, because "the determinant is zero" and "the determinant is nonzero" are nearly identical as strings but say opposite things. When the list is too long for one prompt, the merge runs as a tree: reduce batches, then reduce the results, so n sections need about log(n) levels instead of one oversized prompt.

**Next up**
Step 3.4: Key concepts + definitions extraction as JSON with schema validation.

---

## 2026-11-04 · Step 3.2: Prompted summarizer (transcript chunks to structured notes)

**What I did**
- Added `src/notetaker/notes.py`: chunks a transcript, asks the LLM for JSON notes per chunk, validates them and writes `notes.md` + `notes.json`
- `chunk_transcript()` packs the pause-based paragraphs from Step 1.3 into chunks of at most 1,200 words (about 7 to 9 minutes of speech), never cutting inside a paragraph
- Each chunk is sent as timestamped lines (`[00:12:41] ...`) with a prompt asking for 1 to 4 sections, each with a heading, a short summary, 2 to 6 key points and the start timestamp. The system prompt tells the model to use only what the lecturer said
- Replies are validated with pydantic schemas (`SectionSchema`, `ChunkNotesSchema`): non-empty heading, at least one key point, empty bullets dropped, a bare list or a markdown code fence accepted. If validation fails, the bad reply and the error go back to the model for one repair attempt (`--max-repairs`)
- Section timestamps are parsed and checked against the chunk's time range; a missing, malformed or out-of-range time falls back to the chunk start
- `Notes` records model, provider, number of chunks and LLM calls, token totals and latency, so cost per lecture can be compared later
- New CLI command `notetaker notes transcript.json` (`-o`, `--title`, `-p`, `-m`, `--chunk-words`, `--max-repairs`) with per-chunk progress lines
- Added `pydantic` as an explicit dependency (it was already installed via pydantic-settings). 41 new tests in `tests/test_notes.py`, all with `FakeLLMClient`, so CI makes no live LLM calls. `make check` passes (388 tests, lint clean)

**How it works**
A full lecture is around 10,000 words, too long for a small free model to summarize well in one go, so the transcript is cut into chunks that each fit comfortably in the context window. Chunking is greedy packing: add paragraphs until the next one would exceed the word budget, then start a new chunk. Cuts only happen at speaker pauses, so a sentence is never split. For each chunk the model must answer in JSON, not free text, because JSON can be checked: a pydantic schema declares the fields and types we expect and raises a precise error otherwise. Small models sometimes return broken JSON, so instead of failing we run a repair loop: show the model its own reply and the exact error, and ask for a corrected version. Timestamps are treated as untrusted model output: we parse them and accept them only if they fall inside the chunk the model actually saw, otherwise we use the chunk start. That guarantees every heading links to a real moment in the recording. This step is the "map" half of map-reduce: each chunk is summarized independently; Step 3.3 adds the "reduce" that merges them.

**Next up**
Step 3.3: Map-reduce summarization for long lectures.

---

## 2026-11-03 · Step 3.1: LLM client for Ollama, Groq and Hugging Face

**What I did**
- Added `src/notetaker/llm.py`: one `LLMClient` interface with three free providers. `OllamaClient` (local, `POST /api/chat`), and `GroqClient` / `HuggingFaceClient`, which share an `OpenAICompatibleClient` because both speak the OpenAI chat-completions format
- `chat(messages)` and the shortcut `complete(prompt, system=...)` return an `LLMResponse` with the text, model, provider, prompt and completion token counts, latency and number of attempts, so later steps can log cost and speed per lecture
- `json_mode=True` asks the provider for JSON only (Ollama `format: json`, OpenAI-style `response_format`), and `LLMResponse.json()` parses it, stripping a markdown code fence if the model adds one anyway. Steps 3.4 to 3.6 rely on this
- Retries with exponential backoff (1s, 2s, 4s, capped at 30s) for rate limits (429), timeouts and 5xx errors, using the server's `Retry-After` when sent. Errors that retrying cannot fix (400, 401, 404) fail at once with a hint, e.g. "run `ollama pull <model>`" or "is Ollama running?"
- `make_client()` builds the client from settings. New settings `NOTETAKER_LLM_TIMEOUT_SECONDS` and `NOTETAKER_LLM_MAX_RETRIES`; `NOTETAKER_LLM_MODEL` may now be empty, meaning each provider's free default (all Llama 3.1 8B Instruct)
- `FakeLLMClient` for tests: scripted replies, records every prompt sent
- New CLI command `notetaker llm "prompt"` (`--provider`, `--model`, `--system`, `--json`, `--max-tokens`) to check the LLM setup
- Standard library only (urllib), no new dependencies. 39 new tests in `tests/test_llm.py` using a fake transport, so CI makes no live LLM calls; one live Ollama test is marked slow and skips if Ollama is not running. `make check` passes (347 tests, lint clean)

**How it works**
Every free provider does the same job (messages in, text out) but with a different request format, so the code is split into a shared part and a provider part. The abstract base class `LLMClient` owns everything that is the same: the retry loop, timing, and turning failures into clear errors. Each provider subclass fills in only two small functions: how to build the HTTP request and how to read the reply. The summarizer will depend only on the interface, so swapping a local Llama for a hosted one is a config change. The actual HTTP call is a separate function (the "transport") passed into the client, which is dependency injection: in tests a fake transport returns canned replies in each provider's real format, so the tests check the exact request we send and the exact reply shape we parse without touching the network. Retries use exponential backoff: wait base * 2^(k-1) seconds before retry k, so short blips cost little while a rate-limited free tier is not hammered. Only errors that might go away are retried; a wrong key will still be wrong in 4 seconds.

**Next up**
Step 3.2: Prompted summarizer: transcript chunks to structured notes (headings, key points).

---

## 2026-10-31 · Step 2.5: Baseline report (model size vs WER vs speed)

**What I did**
- Added `src/notetaker/baseline.py`, which joins the three Phase 2 measurements (speed and RAM from `reports/benchmark/benchmark.json`, WER / CER from `reports/wer/*.json`, term recall from `reports/terms/*.json`) into one row per Whisper size
- Report names are matched with `model_key()`, so `whisper-small`, `Whisper Small` and `small` are the same model while `small-int8` or `small-raw` stay separate. Parameter counts come from the Whisper paper (tiny 39M, base 74M, small 244M, medium 769M, large 1550M, plus turbo and distil)
- The report marks the Pareto frontier, recommends the most accurate model within a real-time-factor budget (default RTF <= 0.5), shows what each size step buys (WER drop in points and relative, vs the compute ratio), lists WER per lecture, and adds caveats when models were not measured the same way (different lectures, different `--max-seconds`, different devices, missing reports)
- New CLI command `notetaker baseline` (`--wer`, `--terms`, `--benchmark`, `--rtf-budget`, `-o`) writing `reports/baseline.md` and `reports/baseline.json`. Only WER reports are required; missing term or benchmark data shows as n/a
- Committed `reports/baseline.md` as a placeholder with the exact command loop, since the sandbox cannot download lectures or Whisper weights. Running the loop locally overwrites it with real numbers
- Pure Python, no new dependencies. Added 38 tests in `tests/test_baseline.py` (hand-worked numbers, NaN handling, duplicate names, comparability checks, end-to-end CLI using the real report writers). `make check` passes (308 tests, lint clean)

**How it works**
Each model gives a point (WER, RTF): accuracy against speed, lower is better on both. A model is dominated if some other model is at least as good on both axes and strictly better on one; the points nobody dominates are the Pareto frontier, and only those are reasonable choices, because any other model has a free upgrade. To pick one point I fix a speed budget (constrained optimization: minimize WER subject to RTF <= budget), which matches the real requirement: notes should be ready well before the next lecture. The step-up table makes diminishing returns visible: going up a size might cut WER by 40% relative at 2x compute, the next step by 20% at 4x compute. A baseline is only useful if it is comparable, so the report checks that every model was scored on the same lectures with the same audio cut and timed on the same device, and says so loudly when not. Any NaN from an old benchmark file is turned into "missing" on load, because NaN silently breaks comparisons (NaN < x and NaN > x are both false).

**Next up**
Step 3.1: LLM client abstraction supporting Ollama (local) and Groq / HF free tier (hosted).

---

## 2026-10-30 · Step 2.4: Technical-term recall metric

**What I did**
- Added `eval/technical_terms.json`, a curated list of 60 technical terms grouped by subject (linear algebra, algorithms, AI and probability, matching the four eval lectures, plus general machine learning for later). Each term lists the spellings that count as the same term, e.g. eigenvalue / eigenvalues / eigen value / eigen values
- Added `src/notetaker/term_recall.py`: loads and validates the list, counts every term in the reference and in the transcript, and computes term recall (spoken terms that made it into the transcript) and term precision (transcribed terms that were really spoken), per lecture, pooled, and by subject, plus a list of the most-missed terms
- New CLI command `notetaker terms`, with the same two modes as `notetaker wer`: `notetaker terms REF HYP` for one pair, `notetaker terms --hyps outputs/eval/whisper-small` for every lecture, writing `reports/terms/<system>.json`, `<system>.md` and a `summary.md` comparison table. Supports `--max-seconds`, `--terms` (your own list), `--system` and `-o`
- Pure Python, no new dependencies. Added 31 tests in `tests/test_term_recall.py` (hand-worked numbers, overlap edge cases, report round trips, CLI) and a test that the committed term list loads. `make check` passes (270 tests, lint clean)
- Still no real numbers: the sandbox cannot download lectures or Whisper weights. Run the README loop locally, then `notetaker terms --hyps ...` next to `notetaker wer`

**How it works**
WER treats "the" and "eigenvalue" as equally important, but notes are useless if the jargon is wrong, so this metric only looks at a fixed vocabulary of technical terms. Both texts go through the same normalizer as WER (Step 2.2), then for each term t we count R_t, how often it was said in the reference, and H_t, how often it appears in the transcript. Hits are clipped, hits_t = min(R_t, H_t), the same trick BLEU uses, so writing "eigenvalue" 50 times cannot earn more hits than the 5 times it was said. Recall = sum of hits / sum of R_t; precision = sum of hits / sum of H_t, which catches a model inventing jargon. Like WER, lectures are pooled by summing counts before dividing. Counting is alignment-free (it does not care where in the lecture a term appears), which makes it robust to small timing shifts at chunk boundaries; the trade-off is that a miss in one place can be cancelled by a hallucination elsewhere, and precision exposes that. Matching uses an inverted index (word -> list of positions), so each term only checks the spots where its first word occurs, then takes the longest matching spelling and skips past it, so "Gaussian elimination" is one hit, not two.

**Next up**
Step 2.5: Baseline report: model size vs WER vs speed, written up in `reports/baseline.md`.

---

## 2026-10-29 · Step 2.3: WER / CER evaluation with jiwer

**What I did**
- Added `src/notetaker/wer.py`: `score_texts()` normalizes a reference and a model transcript (Step 2.2), aligns them with jiwer, and returns word counts (hits, substitutions, deletions, insertions), WER, CER and the most common word confusions (e.g. "equations" heard as "questions")
- `SystemReport` collects one model's scores over all lectures and computes pooled WER / CER; `write_reports()` saves `reports/wer/<system>.json`, `<system>.md` and rebuilds `reports/wer/summary.md`, a comparison table with one row per model and one WER column per lecture
- New CLI command `notetaker wer`, in two modes: `notetaker wer REF HYP` scores one pair; `notetaker wer --hyps outputs/eval/whisper-small` matches every `data/raw/ocw/<id>/reference.json` to `<hyps>/<id>/transcript.json` and writes the report. `--max-seconds 600` scores only the first 10 minutes (handy on a CPU laptop), `--no-normalize` shows how much normalization matters, `--system` names the row
- Added `jiwer>=3.0.5` as a dependency (3.0.0 to 3.0.4 break with current rapidfuzz; checked 3.0.5, 3.1.0 and 4.0.0)
- Added 27 tests in `tests/test_wer.py` with hand-worked expected numbers, plus a README section on running the evaluation. `make check` passes (239 tests, lint clean)
- No real numbers yet: the build sandbox cannot reach OCW or download Whisper weights. Run the README loop locally to get the first `reports/wer/summary.md`

**How it works**
WER is edit distance on words: the minimum number of substitutions (S), deletions (D) and insertions (I) that turn the model's words into the reference's, divided by the number of reference words N. jiwer finds that minimum with the standard Levenshtein dynamic program (a table where cell (i, j) holds the cheapest way to match the first i reference words to the first j model words, filled in from the top left). Because I counts too, WER can go above 100%. CER is the same formula on characters, which is gentler on near-misses: "eigenvalues" vs "eigen values" is 2 word errors but 1 character error, so comparing WER and CER tells you whether mistakes are whole wrong words or just spelling and splitting. Across lectures I pool the counts (sum all errors, divide by all reference words) instead of averaging per-lecture WERs, so every word has equal weight and a short lecture cannot swing the result; that is the convention in the Whisper paper. The `--max-seconds` cut keeps any caption or segment that starts before the cut on both sides, so a sentence crossing the boundary is not half counted. The report stores raw counts in JSON, not just percentages, so later steps (the baseline report in 2.5) can re-pool or compare models without re-scoring.

**Next up**
Step 2.4: Technical-term recall metric (e.g. "eigenvalue", "backprop") from a curated term list.

---

## 2026-10-28 · Step 2.2: Text normalization for fair comparison

**What I did**
- Added `src/notetaker/normalize.py` with `normalize_text()` / `normalize_words()`: one function that both the reference captions and the model output go through before scoring, so writing-style differences do not count as errors
- It removes caption clutter (`[LAUGHTER]`, `(APPLAUSE)`, `>>`, speaker labels like `PROFESSOR:` or `GILBERT STRANG:`), lowercases, spells out numbers (`3` -> three, `1,024` -> one thousand twenty four, `2nd` -> second, `50%` -> fifty percent, `3.14` -> three point one four, `$5` -> five dollars), speaks a few symbols (`x²` -> x squared, `+` plus, `=` equals), drops apostrophes and other punctuation, removes filler words (um, uh, hmm) and unifies spellings (ok -> okay)
- `load_text()` reads the spoken text out of `transcript.json`, `reference.json` or a plain text file
- New CLI command `notetaker normalize FILE` (with `--keep-digits` and `--keep-fillers`) prints exactly what the WER scorer will see
- Pure Python, no new dependencies. Added 64 tests in `tests/test_normalize.py`, including "same meaning, different style" pairs that must normalize identically and a check that normalizing twice changes nothing. `make check` passes (212 tests, lint clean)

**How it works**
Word error rate counts word-level edits between two texts, so "So, there are 3 eigenvalues." versus "so there are three eigenvalues" would score 40% WER even though nothing was misheard. The fix is to map both texts to one canonical form first, which is how the Whisper paper and every published WER number does it. The order of the steps matters: speaker labels are found by their CAPITAL letters, so they are removed before lowercasing; `x²` is translated to "x squared" before Unicode NFKC folding (NFKC would otherwise turn `²` into a plain `2`); thousands separators are removed before ordinals are matched so `1,000th` works. Numbers are spelled by splitting into groups of three digits from the right (2,024 -> [2, 24]), spelling each group under 1000 and appending its scale word (thousand, million, ...). Lookaround regexes check a neighbouring character without consuming it, which is how "1,000" is treated as one number while "1,2,3" stays a list. Parentheses are only stripped when the inside is all capitals, so `(APPLAUSE)` goes but `f(x)` in a math lecture stays. A key property, tested directly, is idempotence: normalizing an already-normalized text changes nothing, so the score never depends on whether a text was cleaned before. Known limit: numbers only go digits -> words, so "nineteen ninety nine" vs "1999" still differ; this is rare and hits every model equally.

**Next up**
Step 2.3: WER / CER evaluation with jiwer, results table saved to `reports/`.

---

## 2026-10-27 · Step 2.1: Download MIT OCW lectures and official transcripts

**What I did**
- Added `eval/ocw_lectures.json`, a committed manifest of four first lectures from MIT OpenCourseWare: 18.06 Linear Algebra, 6.006 Algorithms, 6.034 Artificial Intelligence and 6.041 Probability. Everyone evaluates on the same list
- Added `src/notetaker/evalset.py`: loads and validates the manifest, finds the video (.mp4) and caption (.srt/.vtt) links on each lecture page, parses SRT and WebVTT captions into timed cues, and extracts 16 kHz mono FLAC audio with ffmpeg straight from the video URL
- New CLI command `notetaker evalset` with `--list`, `--only ID ...`, `--no-audio`, `--force` and `-o`. Output goes to `data/raw/ocw/<id>/` (gitignored: the files are large, and the CC BY-NC-SA license attaches attribution and non-commercial conditions to any copy, so each person downloads their own): `audio.flac`, `reference.json`, `reference.txt`, `meta.json`
- A manifest entry can give `audio_url` / `transcript_url` directly, which skips the page scan if OCW ever changes its HTML
- Added 31 tests in `tests/test_evalset.py` with a fake "internet" (a dict from URL to text), plus a real ffmpeg extraction from a generated stereo 44.1 kHz tone that checks the result is 16 kHz mono. `make check` passes (148 tests, lint clean)
- Honest caveat: the build sandbox cannot reach ocw.mit.edu, so the four page URLs and the page layout have not been checked against the live site yet. Run `notetaker evalset --no-audio` locally first; any lecture that fails prints which manifest field to fill in

**How it works**
OCW publishes professionally written captions for its lecture videos, and those are the "ground truth" for measuring word error rate in Step 2.3. The downloader reads each lecture page with Python's built-in HTML parser and collects every link, then picks the archive.org .mp4 and the .srt caption file (falling back to any .mp4 and to .vtt). Caption files are blocks separated by blank lines, each with a timing line `00:01:02,500 --> 00:01:05,000` followed by the text; one regular expression handles both SRT (comma before the milliseconds) and WebVTT (dot, optional hours), and formatting tags like `<i>` are stripped. For audio, ffmpeg is given the video URL directly and told to drop the video stream, mix to mono and resample to 16 kHz, so a few hundred MB of video turns into roughly 50 MB of lossless FLAC without the video ever touching the disk. Every file is first written as `name.part` and then renamed: a rename is atomic, so an interrupted run never leaves a truncated file under the real name, which is what makes "skip files that already exist" safe on re-runs. The network and ffmpeg functions are passed into `download_lecture` as arguments (dependency injection), so tests swap in fakes and run offline.

**Next up**
Step 2.2: Text normalization for fair comparison (case, punctuation, numbers).

---

## 2026-10-26 · Step 1.5: Speed and memory benchmark for Whisper sizes

**What I did**
- Added `src/notetaker/benchmark.py`: `benchmark_model()` loads one Whisper size, times the load and N repeated transcriptions, and records peak memory; `BenchmarkResult` derives the median time, real-time factor (RTF), "x real time", and model RAM; `write_report()` saves `benchmark.json` and a Markdown table `benchmark.md`
- New CLI command: `notetaker benchmark lecture.mp3 -m tiny base small --repeats 3 --max-seconds 600`, writing to `reports/benchmark/` by default
- Each model runs in a fresh child process by default (`--no-isolate` turns this off), so memory numbers do not leak from one model into the next
- If one model fails (download blocked, out of memory), the error is printed and the remaining models still run; the command only fails if none succeed
- `--max-seconds` benchmarks just the start of a long lecture for a quick estimate
- Added 20 tests in `tests/test_benchmark.py` using a fake model and a fake clock, so every timing is exactly known, plus a real child-process run with the fake loader. `make check` passes (117 tests, lint clean). The `slow` test with the real `tiny` model could not run tonight (Hugging Face downloads are blocked in the build sandbox), so run `notetaker benchmark` on a real lecture locally to get the first numbers

**How it works**
The headline metric is the real-time factor: compute time divided by audio length, so RTF 0.25 means an hour of lecture takes 15 minutes, and anything under 1 keeps up with live speech. Each model is timed over several runs and the median is kept, because the first run often pays one-off warm-up costs and the median, unlike the mean, ignores one slow outlier. Timing uses `time.perf_counter`, a high-resolution clock that never jumps when the system time changes. Memory is the process's peak resident set size (the RAM pages it actually used), read from the operating system via Python's `resource` module. A peak can only go up, so measuring tiny, base and small in one process would make every later number include the earlier models; instead each model gets a brand-new Python process ("spawn" start method), and its peak is reported along with the baseline measured just before the model was loaded, so the difference is roughly what the model itself costs. The loader and the clock are passed in as arguments (dependency injection), which is what lets the tests replace them with fakes and check the arithmetic exactly.

**Next up**
Step 2.1: Script to download a few MIT OpenCourseWare lectures + their official transcripts.

---

## 2026-10-25 · Step 1.4: VAD-based chunking for long lectures

**What I did**
- Added `src/notetaker/chunking.py`: an energy-based voice activity detector (pure NumPy, no model download) and `plan_chunks()`, which splits a waveform into consecutive pieces of at most `max_chunk_s` seconds, cutting in the middle of pauses
- `transcribe()` now splits anything longer than `NOTETAKER_CHUNK_MAX_SECONDS` (new setting, default 300 s; 0 turns it off), transcribes the chunks one at a time, and stitches them into one `Transcript`: timestamps of segments and words are shifted by each chunk's start, segment ids are renumbered 1, 2, 3, ... across chunks
- Chunks that are pure silence are skipped (Whisper tends to invent text like "Thank you." on silence), and a language auto-detected on the first chunk is locked in for the rest
- CLI: new `--max-chunk-seconds` option and a progress line per chunk (`chunk 3/18 [00:10:00 - 00:15:00]`)
- Documented the setting in `.env.example` and the README
- Added 30 tests (`tests/test_chunking.py`, `tests/test_transcribe_chunked.py`, plus CLI tests) on synthetic tones and pauses where every cut point is known in advance, including a randomized check that chunks always tile the audio exactly. `make check` passes (98 tests, lint clean). Planning the cuts for 90 minutes of audio takes under a second

**How it works**
The detector slices the audio into 30 ms frames and computes each frame's loudness in decibels, 20 log10(RMS). A frame counts as speech if it is louder than a threshold that adapts to the recording: 35 dB below its 95th-percentile loudness, but never below -50 dB, so quiet recordings still work and faint hiss never counts as speech. Runs of non-speech frames are found in one vectorized pass (pad the silence indicator with zeros, take `np.diff`, and the +1 and -1 entries mark where runs start and end). Cutting is greedy: from the current start, look in the window between half the maximum and the maximum length, pick the longest pause of at least 0.3 s there and cut at its midpoint; if the lecturer never pauses, cut at the quietest frame instead. So every chunk is between 2.5 and 5 minutes except the last, and no cut lands mid-word. Chunks are stored as integer sample ranges, so they tile the signal exactly with no rounding gaps. Each chunk is a NumPy view (a slice, not a copy), and Whisper only ever holds one chunk's decoding state at a time, which bounds its memory regardless of lecture length. One honest limit: the decoded waveform itself is still loaded whole (about 345 MB of float32 for 90 minutes); streaming the decode from ffmpeg would remove that if it ever matters.

**Next up**
Step 1.5: Benchmark script: real-time factor and memory for tiny/base/small models.

---

## 2026-10-24 · Step 1.3: `notetaker transcribe` CLI

**What I did**
- Added a `notetaker transcribe <file>` command that transcribes one recording and writes `transcript.json` and `transcript.md` into `outputs/<file name>/` (or any folder given with `-o`)
- Options: `--model` (choices come straight from the config's allowed Whisper sizes), `--language`, `--beam-size`, and `--title` for the Markdown heading
- Added `src/notetaker/export.py` with `format_timestamp` (761.4 s becomes `00:12:41`), `group_paragraphs`, `to_markdown`, and `write_outputs`
- Clear errors with exit code 1 for a missing file, a bad beam size, a missing ffmpeg, or an unreadable file, instead of a Python traceback
- Added a Usage section to the README and gitignored `outputs/` so generated transcripts are never committed
- Added 21 tests (`tests/test_export.py`, `tests/test_cli_transcribe.py`), including an end-to-end run on a generated tone through the real ffmpeg pipeline with a fake model. `make check` passes (68 tests, lint clean) in a fresh virtualenv

**How it works**
The command is a thin layer: argparse reads the arguments, `transcribe()` from Step 1.2 does the speech recognition, and the export module turns the result into files. Whisper returns sentence-sized segments, so printing one per line would give a 90-minute lecture well over a thousand choppy lines. Instead, consecutive segments are joined into paragraphs, and a new paragraph starts when the gap between one segment's end and the next one's start is at least 2 seconds (a natural pause), or when the paragraph already spans 60 seconds (so a lecturer who never pauses still gets readable blocks). Each paragraph is labelled with its start time, floored to the second, so jumping to that time in a video player lands just before the speech. The JSON file keeps everything (word timestamps, confidence scores) so later steps never need to rerun Whisper. In the tests, pytest's `monkeypatch` temporarily swaps the real `transcribe` function (or just the model loader) for a fake, which lets the whole command run in milliseconds with no model download.

**Next up**
Step 1.4: Chunk long audio (VAD-based) so 90-minute lectures don't blow up memory.

---

## 2026-10-23 · Step 1.2: Transcription with faster-whisper

**What I did**
- Added `src/notetaker/transcribe.py` with `transcribe(audio)`, which takes a file path or a 16 kHz waveform and returns a `Transcript`: a list of `Segment`s (start, end, text, confidence scores), each holding `Word`s with their own timestamps
- Added `load_model()`, which loads the configured Whisper size once and reuses it, and picks the hardware automatically: GPU with float16 if one is present, otherwise CPU with int8
- `Transcript` saves to and loads from JSON (`save_json` / `load_json`), so a 90-minute lecture only has to be transcribed once
- File inputs are always decoded at 16 kHz, even if `NOTETAKER_SAMPLE_RATE` is changed, because Whisper only understands 16 kHz
- Added `faster-whisper` as a dependency and a README note that model weights download on first use and are never committed
- Added 12 tests in `tests/test_transcribe.py` using a fake model, so CI never downloads weights. `make check` passes (47 tests, lint clean). The one `slow` test that runs the real `tiny` model could not run in tonight's build sandbox (Hugging Face downloads are blocked there), so run `pytest -m slow` once locally to confirm the end-to-end path

**How it works**
faster-whisper runs OpenAI's open Whisper weights on CTranslate2, a C++ inference engine that is several times faster than the original PyTorch code. Whisper is an encoder-decoder transformer: the encoder turns 30-second windows of the audio's log-mel spectrogram into feature vectors, and the decoder writes text token by token, also predicting timestamp tokens that mark where each segment starts and ends. Beam search (beam size 5) keeps the 5 most probable partial transcriptions at each step instead of greedily taking the single best token, which usually fixes small mistakes. Word timestamps come from cross-attention: for each output word, the model's attention over the audio frames shows which frames it was "listening to", and aligning those gives each word a start and end time. On CPU the weights are quantized to int8 (an 8-bit integer plus a scale per row), cutting memory about 4x versus float32 for a tiny accuracy cost. For testing I used dependency injection: `transcribe()` accepts any object with a `.transcribe()` method, so the tests hand it a fake model with known outputs and check the conversion exactly, with no download.

**Next up**
Step 1.3: CLI `notetaker transcribe lecture.mp3` that writes `transcript.json` and `transcript.md`.

---

## 2026-10-22 · Step 1.1: Audio loading and normalization

**What I did**
- Added `src/notetaker/audio.py` with `load_audio(path)`, which turns any file ffmpeg can read (mp3, wav, m4a, mp4 video, ...) into a 16 kHz mono float32 NumPy array in [-1, 1), the exact format Whisper expects
- Added helpers: `peak_normalize` (consistent volume for quiet recordings), `duration_seconds`, and `save_wav` (writes 16-bit mono WAV with Python's built-in `wave` module, for caching and for chunking in Step 1.4)
- Added clear error types: `FFmpegNotFoundError` with install instructions, and `AudioLoadError` for missing, corrupt, or non-audio files
- The default sample rate comes from the config module (`NOTETAKER_SAMPLE_RATE`, 16000)
- Added `numpy` as a dependency, an "Install ffmpeg" step in CI, and a Requirements section in the README
- Added 15 tests in `tests/test_audio.py`; no audio is committed, each test synthesizes a half-second tone with ffmpeg in a temp folder. `make check` passes (36 tests, lint clean) in a fresh virtualenv

**How it works**
ffmpeg does the hard part: decoding the file, averaging stereo channels down to mono, resampling to 16 kHz, and writing the result as raw 16-bit integers to its standard output. Python reads those bytes through a pipe (no temp files), reinterprets them as int16 with `np.frombuffer`, and divides by 32768 to get floats in [-1, 1). The tests check the math directly: a 440 Hz sine recorded at 44.1 kHz stereo must come back with 8000 samples (0.5 s x 16000), its FFT peak must still be at 440 Hz (bin width 16000 / 8000 = 2 Hz), and its amplitude must still be 0.5. A clip with sin on the left and -sin on the right must decode to near silence, which proves the channels are averaged rather than one being dropped. Peak normalization multiplies the whole signal by one constant, target / max|x[n]|, so it changes loudness without distorting the waveform.

**Next up**
Step 1.2: Transcribe with faster-whisper (small model), return segments with timestamps.

---

## 2026-10-21 · Step 0.5: Config module and `.env.example`

**What I did**
- Added `src/notetaker/config.py`: one typed `Settings` class (built on pydantic-settings) holding every knob the roadmap needs, grouped by phase: paths, Whisper model/device/precision/language/sample rate, LLM provider/model/temperature, Ollama URL, free-tier API keys, and the embedding model
- Added `.env.example`, a commented template documenting every setting with its default and allowed values
- Added a `notetaker config` command that prints the active settings with API keys masked, and prints a clear error (exit code 1) for invalid values
- Added `pydantic-settings` as the first runtime dependency and a Configuration section to the README
- Added 16 tests in `tests/test_config.py` covering defaults, env vars, `.env`, priority order, invalid values, blank values, secret masking, and a guard test that fails if a setting is ever missing from `.env.example`; `make check` passes (21 tests, lint clean) in a fresh virtualenv

**How it works**
Instead of hard-coding values like "use the small Whisper model" across many files, every module asks `get_settings()` for them. pydantic-settings looks for each value in three places, in priority order: a real environment variable such as `NOTETAKER_WHISPER_MODEL=base`, then a `.env` file, then the default in the code. It also checks each value against its type, so a typo like `smal` or a temperature of 5 fails immediately at startup with a readable message instead of breaking halfway through a long job. Choices like the LLM provider are restricted to a fixed list (`ollama`, `groq`, `huggingface`), which also enforces the free-models-only rule in code. API keys are stored as `SecretStr`, which prints as asterisks, so they cannot leak into logs; the real `.env` is gitignored and only the blank template is committed. `get_settings()` is cached, so the environment is read once and the whole program shares one settings object.

**Next up**
Step 1.1: Audio loading and normalization (ffmpeg, 16 kHz mono) with tests on a tiny generated clip.

---

## 2026-10-20 · Step 0.4: GitHub Actions CI

**What I did**
- Added `.github/workflows/ci.yml`, which runs `make lint` and `make test` on every push to `main`, every pull request, and on demand from the Actions tab
- Tests run on both Python 3.10 (the oldest version the package supports) and 3.12, using a build matrix
- Turned on pip caching, auto-cancel of outdated runs, a 15-minute timeout, and read-only permissions for the workflow
- Added a CI status badge to the top of `README.md`
- Checked locally: the workflow file passes `actionlint`, and `make check` (lint plus all 5 tests) is clean

**How it works**
GitHub Actions is a free service that starts a fresh Linux machine whenever an event like a push happens, then runs the steps listed in the workflow file: download the code, install Python, install the package with `make install`, then run `make lint` and `make test`. If any step fails, the commit gets a red X and the badge turns red, so a broken change is caught right away instead of weeks later. Using the same `make` commands locally and in CI means a passing `make check` on my laptop predicts a passing CI run. The matrix runs the identical job once per Python version, which catches code that accidentally depends on a newer language feature. CI deliberately skips tests marked `slow` and never calls a live LLM or downloads model weights, so it stays fast, free, and does not need any API keys.

**Next up**
Step 0.5: Config module (pydantic-settings) and `.env.example` documenting every setting.

---

## 2026-10-19 · Step 0.3: Tooling (ruff, pytest, Makefile)

**What I did**
- Added `[tool.pytest.ini_options]` and `[tool.ruff]` settings to `pyproject.toml`, so both tools are configured in the same file as the package
- Turned on a wider set of ruff rule families: style (E, W), real bugs (F), import order (I), bug patterns (B), modern syntax (UP), and simplifications (SIM)
- Registered a `slow` pytest marker for future tests that load real models or audio, and made unknown markers an error
- Added a `Makefile` with `make install`, `make lint`, `make format`, `make test`, and `make check` (lint plus tests, the same thing CI will run)
- Added `tests/test_cli.py` with smoke tests for `--version`, a bad flag, and the program name; all 5 tests pass and lint is clean

**How it works**
A linter reads code without running it and flags likely bugs and style problems; ruff does this plus formatting, so one tool replaces three older ones. pytest finds every `test_*.py` file under `tests/` and runs every `test_*` function in it; a test passes if its `assert` statements hold. The `slow` marker lets me tag expensive tests (like running Whisper on real audio) so the default `make test` and CI skip them and stay fast and free. The Makefile is just a list of named shortcuts, so `make check` always runs exactly the same commands locally and in CI, which avoids "works on my machine" surprises.

**Next up**
Step 0.4: GitHub Actions CI (lint + tests on every push).

---

## 2026-10-18 · Step 0.2: Python project setup

**What I did**
- Added `pyproject.toml` declaring the `notetaker` package, Python 3.10+, and a `dev` extra with pytest and ruff
- Created the `src/notetaker/` package with a version string and a minimal `notetaker` CLI (`notetaker --version`)
- Added `requirements.txt` (installs the package in editable mode with dev tools) and a `.gitignore` that keeps out caches, `.env` secrets, audio, and model weights
- Added two smoke tests in `tests/test_package.py`; both pass, and ruff lint and format checks are clean

**How it works**
`pyproject.toml` is the one config file modern Python tools read. It says what the package is called, what it depends on, and that installing it should create a `notetaker` command that runs `notetaker.cli:main`. The code uses the "src layout", meaning it lives in `src/notetaker` instead of the repo root. That forces tests to import the installed package rather than whatever files happen to sit next to them, so packaging mistakes show up early. Installing with `pip install -e ".[dev]"` (editable mode) links the install to the source folder, so edits take effect immediately. The version is read from installed package metadata, so it is defined once in `pyproject.toml`. The `.gitignore` blocks audio, model files, and `.env` so large files and secrets never land in git.

**Next up**
Step 0.3: Tooling (ruff + pytest config in `pyproject.toml`, a `Makefile` with `make test` / `make lint`).

---

## 2026-10-17 · Step 0.1: Repo scaffold

**What I did**
- Wrote `README.md` describing the project, its goals, and the planned stack
- Wrote `ROADMAP.md`: the full project split into ~45 small steps across 9 phases
- Started this dev log

**Why**
The project gets built in small nightly increments, so the plan has to live in the repo itself. Every session reads the roadmap, picks the next step, and logs it here.

**Next up**
Step 0.2: Python project setup (package layout, dependencies, `.gitignore`).
