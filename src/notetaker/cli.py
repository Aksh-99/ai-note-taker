"""Command-line entry point for `notetaker`.

Commands so far:
    notetaker --version   print the version
    notetaker config      print the active settings (API keys masked)
    notetaker transcribe lecture.mp3
                          transcribe a recording into transcript.json and
                          transcript.md (in outputs/<file name>/ by default)
    notetaker benchmark lecture.mp3 -m tiny base small
                          measure speed (real-time factor) and memory of
                          several Whisper sizes, report in reports/benchmark/
    notetaker evalset     download the MIT OCW evaluation lectures (audio +
                          official transcripts) into data/raw/ocw/
    notetaker normalize FILE
                          print a transcript the way the WER scorer sees it
    notetaker wer REF HYP score one transcript against a reference
    notetaker wer --hyps outputs/eval/small --system whisper-small
                          score a whole folder of lectures, report in reports/wer/
    notetaker terms REF HYP
                          technical-term recall of one transcript
    notetaker terms --hyps outputs/eval/small --system whisper-small
                          term recall over a folder, report in reports/terms/
    notetaker baseline    join benchmark, WER and term reports into
                          reports/baseline.md (model size vs WER vs speed)
    notetaker llm "Explain eigenvalues in one sentence"
                          send one prompt to the configured LLM (Ollama by
                          default) to check the setup works
    notetaker notes outputs/lecture01/transcript.json
                          write structured study notes (notes.md + notes.json)
    notetaker concepts outputs/lecture01/transcript.json
                          glossary of key terms (concepts.md + concepts.json)
                          next to the transcript, using the configured LLM
    notetaker flashcards outputs/lecture01/transcript.json
                          Q/A flashcards (flashcards.md + .json + an Anki CSV);
                          add --concepts concepts.json for glossary cards
    notetaker quiz outputs/lecture01/transcript.json
                          multiple-choice quiz with shuffled options
                          (quiz.md + quiz.json); --concepts adds glossary
                          questions
    notetaker faithfulness outputs/lecture01/notes.json
                          LLM-as-judge check of notes (or concepts, flashcards,
                          quiz) against the transcript, with verified quotes
                          (faithfulness_<kind>.md + .json); --lexical-only is a
                          free word-overlap baseline
    notetaker agent outputs/lecture01/transcript.json
                          run the whole pipeline as a LangGraph graph (notes,
                          concepts, flashcards, quiz, tasks) in one command;
                          --graph prints the graph as a Mermaid diagram;
                          --record-llm / --replay-llm save or reuse LLM replies
    notetaker tasks outputs/lecture01/transcript.json --lecture-date 2026-11-11
                          find homework, exams and readings with deadlines
                          resolved to real dates (tasks.md + tasks.json,
                          plus todo.json and a tasks.ics calendar file);
                          --no-llm uses offline rules instead of an LLM
    notetaker todo outputs/lecture01/tasks.json --reminder-minutes 120
                          re-export tasks.json as todo.json + tasks.ics with
                          other calendar options (--vtodo, --no-reminder)
    notetaker ingest outputs/ --index index/
                          embed every transcript.json under outputs/ into a
                          FAISS search index; re-runs only embed new or
                          changed lectures (--prune drops deleted ones)
    notetaker ask "What is an eigenvector?" --index index/
                          answer a question from the indexed lectures with
                          numbered citations to lecture + timestamp;
                          --no-llm returns the best passage verbatim
    notetaker finetune-data
                          cut the downloaded eval lectures into <= 30 s labeled
                          clips for Whisper fine-tuning, split train/val/test
                          by whole lecture so nothing leaks (data/finetune/)
"""

import argparse
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from notetaker import __version__
from notetaker.audio import AudioError
from notetaker.chunking import Chunk
from notetaker.config import LLMProvider, WhisperModel, get_settings

# Stage names for `notetaker agent --steps`. Kept as a literal here (and
# checked against agent.STEP_ORDER in the tests) so building the parser does
# not import langgraph, which keeps every other command fast to start.
STEP_NAMES = ("notes", "concepts", "flashcards", "quiz", "tasks")

# Ranking methods for `notetaker eval-retrieval --retriever`. Same trick:
# a literal checked against hybrid.RETRIEVERS in the tests, so the parser
# does not import FAISS.
RETRIEVERS = ("vector", "bm25", "hybrid", "hybrid-linear")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Kept separate from main() so tests can inspect the parser without
    actually running the program.
    """
    parser = argparse.ArgumentParser(
        prog="notetaker",
        description="AI lecture note taker built on free, open models.",
    )
    parser.add_argument(
        "--version", action="version", version=f"notetaker {__version__}"
    )
    # Subcommands (like `git commit` vs `git push`). `dest="command"` stores
    # which one was chosen, so main() can dispatch on it.
    subcommands = parser.add_subparsers(dest="command", metavar="<command>")
    subcommands.add_parser(
        "config", help="show active settings (from env vars, .env, defaults)"
    )

    tr = subcommands.add_parser(
        "transcribe",
        help="transcribe an audio/video file to transcript.json + transcript.md",
    )
    tr.add_argument("audio", type=Path, help="any file ffmpeg can read (mp3, mp4...)")
    tr.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="folder for the outputs (default: <NOTETAKER_OUTPUT_DIR>/<file name>)",
    )
    tr.add_argument(
        "-m",
        "--model",
        # get_args() pulls the allowed strings out of the Literal type in
        # config.py, so the CLI and the settings can never disagree.
        choices=get_args(WhisperModel),
        default=None,
        help="Whisper size (default: NOTETAKER_WHISPER_MODEL)",
    )
    tr.add_argument(
        "-l", "--language", default=None, help="language code like 'en' (default: auto)"
    )
    tr.add_argument(
        "--beam-size", type=int, default=5, help="beam search width (1 = greedy)"
    )
    tr.add_argument(
        "--title", default=None, help="heading for transcript.md (default: file name)"
    )
    tr.add_argument(
        "--max-chunk-seconds",
        type=float,
        default=None,
        help=(
            "split long recordings at pauses into pieces of at most this many "
            "seconds (default: NOTETAKER_CHUNK_MAX_SECONDS; 0 = no splitting)"
        ),
    )

    bm = subcommands.add_parser(
        "benchmark",
        help="compare speed and memory of Whisper sizes on one recording",
    )
    bm.add_argument("audio", type=Path, help="a lecture recording ffmpeg can read")
    bm.add_argument(
        "-m",
        "--models",
        nargs="+",  # one or more values: -m tiny base small
        choices=get_args(WhisperModel),
        default=["tiny", "base", "small"],
        help="Whisper sizes to compare (default: tiny base small)",
    )
    bm.add_argument(
        "--repeats", type=int, default=1, help="timed runs per model (median kept)"
    )
    bm.add_argument(
        "--beam-size", type=int, default=5, help="beam search width (1 = greedy)"
    )
    bm.add_argument(
        "-l", "--language", default=None, help="language code like 'en' (default: auto)"
    )
    bm.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="only use the first N seconds of the recording, for a quicker run",
    )
    bm.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("reports/benchmark"),
        help="folder for benchmark.json + benchmark.md (default: reports/benchmark)",
    )
    bm.add_argument(
        "--no-isolate",
        action="store_true",
        help=(
            "run all models in this process (faster start-up, but the memory "
            "numbers of later models include earlier ones)"
        ),
    )
    ev = subcommands.add_parser(
        "evalset",
        help="download the MIT OpenCourseWare evaluation lectures + transcripts",
    )
    ev.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval/ocw_lectures.json"),
        help="JSON list of lectures (default: eval/ocw_lectures.json)",
    )
    ev.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="where to save them (default: <NOTETAKER_DATA_DIR>/raw/ocw)",
    )
    ev.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="ID",
        help="download just these lecture ids (default: all)",
    )
    ev.add_argument("--list", action="store_true", help="list the lectures and exit")
    ev.add_argument(
        "--no-audio",
        action="store_true",
        help="only download the reference transcripts (a few KB each)",
    )
    ev.add_argument(
        "--force", action="store_true", help="re-download files that already exist"
    )
    nm = subcommands.add_parser(
        "normalize",
        help="print a transcript in the normalized form used for WER scoring",
    )
    nm.add_argument(
        "file",
        type=Path,
        help="transcript.json, reference.json, or any plain-text file",
    )
    nm.add_argument(
        "--keep-digits",
        action="store_true",
        help='leave numbers as digits instead of spelling them ("3" not "three")',
    )
    nm.add_argument(
        "--keep-fillers",
        action="store_true",
        help="keep filler words such as um, uh, hmm",
    )
    we = subcommands.add_parser(
        "wer",
        help="word / character error rate of transcripts against references",
        description=(
            "Either give two files (REF HYP) to score one pair, or give --hyps "
            "to score every lecture in a folder and write a report."
        ),
    )
    # nargs="*" means zero or more values, so both modes share one command.
    we.add_argument(
        "files",
        nargs="*",
        type=Path,
        metavar="REF HYP",
        help="a reference and a transcript file (pair mode)",
    )
    we.add_argument(
        "--refs",
        type=Path,
        default=None,
        help="folder with <id>/reference.json (default: <NOTETAKER_DATA_DIR>/raw/ocw)",
    )
    we.add_argument(
        "--hyps",
        type=Path,
        default=None,
        help="folder with <id>/transcript.json from one model (folder mode)",
    )
    we.add_argument(
        "--system",
        default=None,
        help="name for this model in the report (default: the --hyps folder name)",
    )
    we.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="only score the first N seconds of each lecture",
    )
    we.add_argument(
        "--no-normalize",
        action="store_true",
        help="compare raw text (to see how much normalization matters)",
    )
    we.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("reports/wer"),
        help="folder for the report in folder mode (default: reports/wer)",
    )
    te = subcommands.add_parser(
        "terms",
        help="technical-term recall of transcripts against references",
        description=(
            "Count how many of the curated technical terms spoken in the "
            "reference also appear in the transcript. Give two files (REF HYP) "
            "or --hyps to score a folder and write a report."
        ),
    )
    te.add_argument(
        "files",
        nargs="*",
        type=Path,
        metavar="REF HYP",
        help="a reference and a transcript file (pair mode)",
    )
    te.add_argument(
        "--terms",
        type=Path,
        default=Path("eval/technical_terms.json"),
        help="curated term list (default: eval/technical_terms.json)",
    )
    te.add_argument(
        "--refs",
        type=Path,
        default=None,
        help="folder with <id>/reference.json (default: <NOTETAKER_DATA_DIR>/raw/ocw)",
    )
    te.add_argument(
        "--hyps",
        type=Path,
        default=None,
        help="folder with <id>/transcript.json from one model (folder mode)",
    )
    te.add_argument(
        "--system",
        default=None,
        help="name for this model in the report (default: the --hyps folder name)",
    )
    te.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="only score the first N seconds of each lecture",
    )
    te.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("reports/terms"),
        help="folder for the report in folder mode (default: reports/terms)",
    )
    bl = subcommands.add_parser(
        "baseline",
        help="combine benchmark, WER and term reports into reports/baseline.md",
        description=(
            "Join the speed benchmark, WER reports and term-recall reports "
            "into one table per model size, mark the Pareto frontier and "
            "recommend the most accurate model within a speed budget."
        ),
    )
    bl.add_argument(
        "--wer",
        type=Path,
        default=Path("reports/wer"),
        help="folder of WER reports (default: reports/wer)",
    )
    bl.add_argument(
        "--terms",
        type=Path,
        default=Path("reports/terms"),
        help="folder of term-recall reports (default: reports/terms)",
    )
    bl.add_argument(
        "--benchmark",
        type=Path,
        default=Path("reports/benchmark/benchmark.json"),
        help="benchmark results (default: reports/benchmark/benchmark.json)",
    )
    bl.add_argument(
        "--rtf-budget",
        type=float,
        default=0.5,
        help="max real-time factor for the recommendation (default: 0.5)",
    )
    bl.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/baseline.md"),
        help="where to write the report (default: reports/baseline.md)",
    )
    lm = subcommands.add_parser(
        "llm",
        help="send one prompt to the configured LLM (setup check)",
        description=(
            "Send a single prompt to the LLM chosen in the settings and print "
            "the reply with token counts and latency. Useful to check that "
            "Ollama is running or that a free-tier key works."
        ),
    )
    lm.add_argument("prompt", help="the question or instruction to send")
    lm.add_argument(
        "-p",
        "--provider",
        choices=get_args(LLMProvider),
        default=None,
        help="override NOTETAKER_LLM_PROVIDER",
    )
    lm.add_argument("-m", "--model", default=None, help="override NOTETAKER_LLM_MODEL")
    lm.add_argument("--system", default=None, help="optional system message")
    lm.add_argument(
        "--json", action="store_true", help="ask for a JSON reply and validate it"
    )
    lm.add_argument("--max-tokens", type=int, default=None, help="cap the reply length")
    no = subcommands.add_parser(
        "notes",
        help="turn a transcript.json into structured study notes",
        description=(
            "Split a transcript into chunks of a few minutes, ask the "
            "configured LLM for notes on each (headings, summary, key points, "
            "start time), merge those into the lecture's main topics with an "
            "overview, and write notes.md + notes.json."
        ),
    )
    no.add_argument("transcript", type=Path, help="a transcript.json file")
    no.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="where to write notes (default: next to the transcript)",
    )
    no.add_argument("--title", default=None, help="notes title (default: folder name)")
    no.add_argument(
        "-p",
        "--provider",
        choices=get_args(LLMProvider),
        default=None,
        help="override NOTETAKER_LLM_PROVIDER",
    )
    no.add_argument("-m", "--model", default=None, help="override NOTETAKER_LLM_MODEL")
    no.add_argument(
        "--chunk-words",
        type=int,
        default=1200,
        help="transcript words per LLM call (default: 1200)",
    )
    no.add_argument(
        "--max-repairs",
        type=int,
        default=1,
        help="times to ask the model to fix invalid JSON (default: 1)",
    )
    no.add_argument(
        "--no-reduce",
        action="store_true",
        help="skip merging chunk notes into topics (one section list per chunk)",
    )
    no.add_argument(
        "--reduce-words",
        type=int,
        default=2500,
        help="word budget per merge call; longer lists merge in levels (default: 2500)",
    )
    co = subcommands.add_parser(
        "concepts",
        help="extract key concepts + definitions from a transcript.json",
        description=(
            "Ask the configured LLM for the technical terms defined in each "
            "chunk, keep only terms that really occur in the transcript, merge "
            "duplicates across chunks, record every timestamp where each term "
            "is mentioned, and write concepts.md + concepts.json. Use --schema "
            "to print the JSON Schema of concepts.json instead."
        ),
    )
    # nargs="?" makes the positional optional, so `notetaker concepts --schema`
    # works without a transcript. run_concepts checks it is given otherwise.
    co.add_argument("transcript", type=Path, nargs="?", help="a transcript.json file")
    co.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="where to write concepts (default: next to the transcript)",
    )
    co.add_argument(
        "--title", default=None, help="glossary title (default: folder name)"
    )
    co.add_argument(
        "-p",
        "--provider",
        choices=get_args(LLMProvider),
        default=None,
        help="override NOTETAKER_LLM_PROVIDER",
    )
    co.add_argument("-m", "--model", default=None, help="override NOTETAKER_LLM_MODEL")
    co.add_argument(
        "--chunk-words",
        type=int,
        default=1200,
        help="transcript words per LLM call (default: 1200)",
    )
    co.add_argument(
        "--max-repairs",
        type=int,
        default=1,
        help="times to ask the model to fix invalid JSON (default: 1)",
    )
    co.add_argument(
        "--schema",
        action="store_true",
        help="print the JSON Schema of concepts.json and exit",
    )
    fc = subcommands.add_parser(
        "flashcards",
        help="generate Q/A flashcards from a transcript.json, export to Anki CSV",
        description=(
            "Ask the configured LLM for flashcards on each chunk, keep only "
            "cards whose answers use the lecturer's words, link each card to "
            "a timestamp, remove duplicate questions, and write flashcards.md, "
            "flashcards.json and flashcards.csv (import the CSV in Anki with "
            "File > Import). --concepts adds one card per glossary entry from "
            "`notetaker concepts`; with --concepts-only no LLM is called."
        ),
    )
    fc.add_argument("transcript", type=Path, nargs="?", help="a transcript.json file")
    fc.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the cards (default: next to the input file)",
    )
    fc.add_argument("--title", default=None, help="deck title (default: folder name)")
    fc.add_argument("--deck", default=None, help="Anki deck name (default: the title)")
    fc.add_argument(
        "--concepts",
        type=Path,
        default=None,
        help="a concepts.json (Step 3.4) to add 'What is X?' cards from",
    )
    fc.add_argument(
        "--concepts-only",
        action="store_true",
        help="only make cards from --concepts; no transcript, no LLM calls",
    )
    fc.add_argument(
        "--max-cards",
        type=int,
        default=6,
        help="most cards per chunk (default: 6)",
    )
    fc.add_argument(
        "--plain-csv",
        action="store_true",
        help="omit the #header lines (for Anki versions older than 2.1.55)",
    )
    fc.add_argument(
        "-p",
        "--provider",
        choices=get_args(LLMProvider),
        default=None,
        help="override NOTETAKER_LLM_PROVIDER",
    )
    fc.add_argument("-m", "--model", default=None, help="override NOTETAKER_LLM_MODEL")
    fc.add_argument(
        "--chunk-words",
        type=int,
        default=1200,
        help="transcript words per LLM call (default: 1200)",
    )
    fc.add_argument(
        "--max-repairs",
        type=int,
        default=1,
        help="times to ask the model to fix invalid JSON (default: 1)",
    )
    fc.add_argument(
        "--schema",
        action="store_true",
        help="print the JSON Schema of flashcards.json and exit",
    )
    qz = subcommands.add_parser(
        "quiz",
        help="generate a multiple-choice quiz from a transcript.json",
        description=(
            "Ask the configured LLM for multiple-choice questions on each "
            "chunk (a correct answer plus 3 distractors), keep only questions "
            "whose correct answer uses the lecturer's words, shuffle the "
            "options, link each question to a timestamp, and write quiz.md "
            "(answer key at the end) and quiz.json. --concepts adds "
            "'which term matches this definition?' questions from "
            "`notetaker concepts`; with --concepts-only no LLM is called."
        ),
    )
    qz.add_argument("transcript", type=Path, nargs="?", help="a transcript.json file")
    qz.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the quiz (default: next to the input file)",
    )
    qz.add_argument("--title", default=None, help="quiz title (default: folder name)")
    qz.add_argument(
        "--concepts",
        type=Path,
        default=None,
        help="a concepts.json (Step 3.4) to add glossary questions from",
    )
    qz.add_argument(
        "--concepts-only",
        action="store_true",
        help="only make questions from --concepts; no transcript, no LLM calls",
    )
    qz.add_argument(
        "--max-questions",
        type=int,
        default=4,
        help="most questions per chunk (default: 4)",
    )
    qz.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for shuffling the options (default: 0)",
    )
    qz.add_argument(
        "-p",
        "--provider",
        choices=get_args(LLMProvider),
        default=None,
        help="override NOTETAKER_LLM_PROVIDER",
    )
    qz.add_argument("-m", "--model", default=None, help="override NOTETAKER_LLM_MODEL")
    qz.add_argument(
        "--chunk-words",
        type=int,
        default=1200,
        help="transcript words per LLM call (default: 1200)",
    )
    qz.add_argument(
        "--max-repairs",
        type=int,
        default=1,
        help="times to ask the model to fix invalid JSON (default: 1)",
    )
    qz.add_argument(
        "--schema",
        action="store_true",
        help="print the JSON Schema of quiz.json and exit",
    )

    fa = subcommands.add_parser(
        "faithfulness",
        help="check notes/concepts/flashcards/quiz against the transcript",
        description=(
            "Split a notes.json, concepts.json, flashcards.json or quiz.json "
            "into claims, retrieve the transcript passages most likely to "
            "back each one, and ask the configured LLM (as a judge, at "
            "temperature 0) whether each claim is supported, partial, "
            "unsupported or contradicted, with an exact quote that is then "
            "checked against the transcript. Writes faithfulness_<kind>.md "
            "and .json. Use a different, ideally larger, free model as judge "
            "than the one that generated the file. --lexical-only needs no LLM."
        ),
    )
    fa.add_argument(
        "target", type=Path, nargs="?", help="notes/concepts/flashcards/quiz .json"
    )
    fa.add_argument(
        "-t",
        "--transcript",
        type=Path,
        default=None,
        help="the transcript.json (default: transcript.json next to the target)",
    )
    fa.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the report (default: next to the target)",
    )
    fa.add_argument(
        "--lexical-only",
        action="store_true",
        help="word-overlap baseline only: no LLM calls, free and instant",
    )
    fa.add_argument(
        "--max-claims",
        type=int,
        default=None,
        help="check at most this many claims, spread over the lecture",
    )
    fa.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="claims per judge call (default: 6)",
    )
    fa.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="transcript passages retrieved per claim (default: 3)",
    )
    fa.add_argument(
        "-p",
        "--provider",
        choices=get_args(LLMProvider),
        default=None,
        help="judge provider (overrides NOTETAKER_LLM_PROVIDER)",
    )
    fa.add_argument(
        "-m",
        "--model",
        default=None,
        help="judge model (overrides NOTETAKER_LLM_MODEL)",
    )
    fa.add_argument(
        "--max-repairs",
        type=int,
        default=1,
        help="times to ask the judge to fix invalid JSON (default: 1)",
    )
    fa.add_argument(
        "--schema",
        action="store_true",
        help="print the JSON Schema of the report and exit",
    )
    ag = subcommands.add_parser(
        "agent",
        help="run the whole pipeline (notes -> concepts -> flashcards -> quiz -> "
        "tasks) as a LangGraph graph",
        description=(
            "Run every stage on one transcript in a single command. The stages "
            "are nodes of a LangGraph graph that share one state, so later "
            "stages reuse earlier results (flashcards and quiz get the "
            "glossary from the concepts stage). A failed stage is reported and "
            "the rest still run. Writes each stage's usual files plus "
            "pipeline.json (timings, LLM calls, errors)."
        ),
    )
    ag.add_argument("transcript", type=Path, nargs="?", help="a transcript.json file")
    ag.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="where to write outputs (default: next to the transcript)",
    )
    ag.add_argument(
        "--title", default=None, help="lecture title (default: folder name)"
    )
    ag.add_argument(
        "--steps",
        nargs="+",
        choices=STEP_NAMES,
        default=None,
        help="run only these stages (always in pipeline order; default: all)",
    )
    ag.add_argument(
        "-p",
        "--provider",
        choices=get_args(LLMProvider),
        default=None,
        help="override NOTETAKER_LLM_PROVIDER",
    )
    ag.add_argument("-m", "--model", default=None, help="override NOTETAKER_LLM_MODEL")
    ag.add_argument(
        "--chunk-words",
        type=int,
        default=1200,
        help="transcript words per LLM call (default: 1200)",
    )
    ag.add_argument(
        "--max-repairs",
        type=int,
        default=1,
        help="times to ask the model to fix invalid JSON (default: 1)",
    )
    ag.add_argument(
        "--no-reduce",
        action="store_true",
        help="notes stage: skip merging chunk notes into topics",
    )
    ag.add_argument("--seed", type=int, default=0, help="quiz shuffle seed")
    _add_task_date_options(ag)
    ag.add_argument(
        "--no-task-llm",
        action="store_true",
        help="tasks stage: use offline rules instead of the LLM",
    )
    ag.add_argument(
        "--graph",
        action="store_true",
        help="print the pipeline graph as Mermaid text and exit (no LLM calls)",
    )
    # Record/replay (Step 4.4). A "mutually exclusive group" makes argparse
    # reject giving both flags at once, with a clear error message.
    cassette = ag.add_mutually_exclusive_group()
    cassette.add_argument(
        "--record-llm",
        type=Path,
        default=None,
        metavar="CASSETTE",
        help="save every LLM request and reply to this JSON file (a cassette)",
    )
    cassette.add_argument(
        "--replay-llm",
        type=Path,
        default=None,
        metavar="CASSETTE",
        help="answer LLM calls from a recorded cassette: offline, no API key, "
        "same output every run",
    )

    tk = subcommands.add_parser(
        "tasks",
        help="extract homework, exams and readings with resolved deadlines",
        description=(
            "Scan a transcript.json for sentences that may assign work "
            "(homework, problem sets, exams, readings), let the configured "
            "LLM decide which are real tasks and copy their deadline words, "
            "then turn those words into calendar dates relative to "
            "--lecture-date. Writes tasks.md (a to-do list), tasks.json, "
            "todo.json and a tasks.ics calendar file. "
            "With --no-llm a rule-based fallback runs offline."
        ),
    )
    tk.add_argument("transcript", type=Path, help="a transcript.json file")
    tk.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the tasks (default: next to the input file)",
    )
    tk.add_argument("--title", default=None, help="title (default: folder name)")
    _add_task_date_options(tk)
    tk.add_argument(
        "--no-llm",
        action="store_true",
        help="no LLM calls: rule-based tasks from sentences that mention a time",
    )
    tk.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="candidate sentences per LLM call (default: 8)",
    )
    tk.add_argument(
        "-p",
        "--provider",
        choices=get_args(LLMProvider),
        default=None,
        help="override NOTETAKER_LLM_PROVIDER",
    )
    tk.add_argument("-m", "--model", default=None, help="override NOTETAKER_LLM_MODEL")
    tk.add_argument(
        "--max-repairs",
        type=int,
        default=1,
        help="times to ask the model to fix invalid JSON (default: 1)",
    )
    td = subcommands.add_parser(
        "todo",
        help="export tasks.json as todo.json and a tasks.ics calendar file",
        description=(
            "Re-export an existing tasks.json (from `notetaker tasks` or "
            "`notetaker agent`) without any LLM calls: todo.json is a flat "
            "to-do list, tasks.ics imports into Google Calendar, Apple "
            "Calendar, Outlook or Thunderbird. Tasks without a resolved date "
            "stay in todo.json only."
        ),
    )
    td.add_argument("tasks", type=Path, help="a tasks.json file")
    td.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the files (default: next to the input file)",
    )
    td.add_argument(
        "--vtodo",
        action="store_true",
        help="write calendar to-dos (VTODO, for task apps) instead of events",
    )
    td.add_argument(
        "--event-minutes",
        type=int,
        default=30,
        help="length of a timed deadline event in minutes (default: 30)",
    )
    reminder = td.add_mutually_exclusive_group()
    reminder.add_argument(
        "--reminder-minutes",
        type=int,
        default=24 * 60,
        help="alarm this many minutes before the deadline (default: 1440 = 1 day)",
    )
    reminder.add_argument(
        "--no-reminder", action="store_true", help="no alarm in the calendar file"
    )
    ig = subcommands.add_parser(
        "ingest",
        help="index all transcripts in a folder for search across lectures",
        description=(
            "Find every transcript under a folder (any depth), split each into "
            "time-stamped passages, embed them and save a FAISS index. Running "
            "it again only embeds lectures that are new or whose transcript "
            "changed (tracked by file hash in <index>/ingest.json)."
        ),
    )
    ig.add_argument("folder", type=Path, help="folder to search for transcripts")
    ig.add_argument(
        "-i",
        "--index",
        type=Path,
        default=Path("index"),
        help="index folder to create or update (default: index/)",
    )
    ig.add_argument(
        "--embedder",
        default=None,
        help=(
            "embedding model for a new index: a sentence-transformers name, or "
            "hashing-512 for a model-free baseline (default: "
            "NOTETAKER_EMBEDDING_MODEL). An existing index keeps its own model"
        ),
    )
    ig.add_argument(
        "--pattern",
        default="transcript.json",
        help="file name glob to look for (default: transcript.json)",
    )
    ig.add_argument(
        "--titles",
        type=Path,
        default=None,
        help='JSON file mapping lecture id to title, e.g. {"la-05": "Eigenvalues"}',
    )
    ig.add_argument(
        "--max-words",
        type=int,
        default=120,
        help="maximum words per passage (default: 120)",
    )
    ig.add_argument(
        "--max-seconds",
        type=float,
        default=90.0,
        help="maximum seconds of speech per passage (default: 90)",
    )
    ig.add_argument(
        "--rebuild",
        action="store_true",
        help="ignore the existing index and re-embed everything",
    )
    ig.add_argument(
        "--prune",
        action="store_true",
        help="remove indexed lectures whose transcript is no longer in the folder",
    )
    ig.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be indexed without embedding or saving anything",
    )
    ak = subcommands.add_parser(
        "ask",
        help="answer a question from your lectures, citing lecture + timestamp",
        description=(
            "Search the index built by `notetaker ingest` for the passages "
            "most similar to the question, then ask the configured LLM to "
            "answer using only those passages and cite them by number. Every "
            "citation is checked against the retrieved passages, and its "
            "lecture title and timestamp come from the index, not the model. "
            "With --no-llm the best passage is returned verbatim (offline)."
        ),
    )
    ak.add_argument("question", help="the question, in quotes")
    ak.add_argument(
        "-i",
        "--index",
        type=Path,
        default=Path("index"),
        help="index folder from `notetaker ingest` (default: index/)",
    )
    ak.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="passages to retrieve and show the model (default: 5)",
    )
    ak.add_argument(
        "-l",
        "--lecture",
        action="append",
        default=None,
        metavar="LECTURE_ID",
        help="only search this lecture (repeat for several)",
    )
    ak.add_argument(
        "--min-score",
        type=float,
        default=None,
        help=(
            "answer 'not covered' without calling the LLM if the best passage's "
            "similarity is below this (e.g. 0.3 for MiniLM; default: off)"
        ),
    )
    ak.add_argument(
        "--no-llm",
        action="store_true",
        help="no LLM call: return the best matching passage verbatim",
    )
    ak.add_argument(
        "-p",
        "--provider",
        choices=get_args(LLMProvider),
        default=None,
        help="override NOTETAKER_LLM_PROVIDER",
    )
    ak.add_argument("-m", "--model", default=None, help="override NOTETAKER_LLM_MODEL")
    ak.add_argument(
        "--max-repairs",
        type=int,
        default=1,
        help="times to ask the model to fix an unusable reply (default: 1)",
    )
    ak.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="also save the answer: .json for the full record, else Markdown",
    )
    ak.add_argument(
        "--json",
        action="store_true",
        help="print the full answer record as JSON instead of text",
    )
    ask_cassette = ak.add_mutually_exclusive_group()
    ask_cassette.add_argument(
        "--record-llm",
        type=Path,
        default=None,
        metavar="CASSETTE",
        help="save the LLM request and reply to a cassette file",
    )
    ask_cassette.add_argument(
        "--replay-llm",
        type=Path,
        default=None,
        metavar="CASSETTE",
        help="answer the LLM call from a recorded cassette (offline)",
    )
    er = subcommands.add_parser(
        "eval-retrieval",
        help="score search on a labeled question set (recall@k and MRR)",
        description=(
            "Run every question in a labeled set through search and check how "
            "high the passage containing the answer ranks. Answers are labeled "
            "as time spans in a lecture, so any passage overlapping the span "
            "counts. Either evaluates an existing index (--index) or builds "
            "one in memory from the transcripts the question file points at, "
            "once per --embedder and --max-words value, and compares them."
        ),
    )
    er.add_argument(
        "-q",
        "--questions",
        type=Path,
        default=Path("eval/retrieval/questions.json"),
        help="labeled question file (default: eval/retrieval/questions.json)",
    )
    # A mutually exclusive group: argparse rejects a command that gives both.
    er_source = er.add_mutually_exclusive_group()
    er_source.add_argument(
        "-i",
        "--index",
        type=Path,
        default=None,
        help="evaluate this existing index folder instead of building one",
    )
    er_source.add_argument(
        "--transcripts",
        type=Path,
        default=None,
        help="transcripts folder to index (default: 'corpus' in the question file)",
    )
    er.add_argument(
        "--embedder",
        action="append",
        default=None,
        help=(
            "embedding model to test, repeat to compare several (e.g. "
            "--embedder hashing-512 --embedder "
            "sentence-transformers/all-MiniLM-L6-v2; default: "
            "NOTETAKER_EMBEDDING_MODEL)"
        ),
    )
    er.add_argument(
        "--titles",
        type=Path,
        default=None,
        help="JSON file mapping lecture id to title (default: from the question file)",
    )
    er.add_argument(
        "--max-words",
        type=int,
        nargs="+",
        default=[120],
        help="passage size(s) in words; give several to compare (default: 120)",
    )
    er.add_argument(
        "--max-seconds",
        type=float,
        default=90.0,
        help="maximum seconds of speech per passage (default: 90)",
    )
    er.add_argument(
        "-k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="cutoffs for recall@k; the largest is the MRR depth (default: 1 3 5 10)",
    )
    er.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write a Markdown report here and the full JSON next to it",
    )
    er.add_argument(
        "--show-misses",
        action="store_true",
        help="list the questions whose answer was not in the top 5",
    )
    er.add_argument(
        "--retriever",
        nargs="+",
        choices=RETRIEVERS,
        default=["vector"],
        help=(
            "ranking method(s) to compare: vector (embeddings), bm25 (keywords), "
            "hybrid (both, fused by reciprocal rank) or hybrid-linear (both, "
            "weighted sum of scores); default: vector"
        ),
    )
    er.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hybrid-linear weight on the vector score, 0 to 1 (default: 0.5)",
    )
    er.add_argument(
        "--rrf-c",
        type=float,
        default=60.0,
        help="constant c in the RRF formula 1 / (c + rank) (default: 60)",
    )

    fd = subcommands.add_parser(
        "finetune-data",
        help="build a Whisper fine-tuning dataset from the eval lectures",
        description=(
            "Cut each downloaded eval lecture (see `notetaker evalset`) into "
            "clips of at most 30 seconds at caption boundaries, label each "
            "clip with its caption text, and split train / val / test by "
            "whole lecture (or whole course) so no lecture, speaker or "
            "sentence leaks into the test set. Writes the Hugging Face "
            "audiofolder layout: <out>/<split>/metadata.jsonl + WAV clips."
        ),
    )
    fd.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="downloaded lectures (default: <NOTETAKER_DATA_DIR>/raw/ocw)",
    )
    fd.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the dataset (default: <NOTETAKER_DATA_DIR>/finetune)",
    )
    fd.add_argument(
        "--group-by",
        choices=("lecture", "course"),
        default="lecture",
        help="keep whole lectures or whole courses on one side (default: lecture)",
    )
    fd.add_argument(
        "--val",
        nargs="+",
        default=None,
        metavar="ID",
        help="put these lecture ids in the validation split",
    )
    fd.add_argument(
        "--test",
        nargs="+",
        default=None,
        metavar="ID",
        help="put these lecture ids in the test split",
    )
    fd.add_argument(
        "--val-fraction",
        type=float,
        default=0.25,
        help="share of lectures for validation when --val is not given (0.25)",
    )
    fd.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="share of lectures for testing when --test is not given (0.25)",
    )
    fd.add_argument(
        "--seed", type=int, default=0, help="changes the random split (default: 0)"
    )
    fd.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="longest clip, at most Whisper's 30 s window (default: 30)",
    )
    fd.add_argument(
        "--min-seconds",
        type=float,
        default=1.0,
        help="drop clips shorter than this (default: 1)",
    )
    fd.add_argument(
        "--max-gap",
        type=float,
        default=1.5,
        help="start a new clip after a silence longer than this (default: 1.5 s)",
    )
    fd.add_argument(
        "--pad",
        type=float,
        default=0.2,
        help="widen each clip by up to this much silence per side (default: 0.2 s)",
    )
    fd.add_argument(
        "--dry-run",
        action="store_true",
        help="only read the captions and show the split; write nothing",
    )
    fd.add_argument("--force", action="store_true", help="replace an existing dataset")
    return parser


def _add_task_date_options(parser: argparse.ArgumentParser) -> None:
    """Options shared by `notetaker tasks` and `notetaker agent`."""
    parser.add_argument(
        "--lecture-date",
        default=None,
        help="date of the lecture, YYYY-MM-DD or 'today', so 'due Friday' "
        "becomes a real date (default: deadlines stay as phrases)",
    )
    parser.add_argument(
        "--class-days",
        nargs="+",
        default=None,
        metavar="DAY",
        help="weekdays the course meets (e.g. mon wed) so 'next class' resolves",
    )


def _parse_task_dates(args: argparse.Namespace):
    """--lecture-date / --class-days -> (date | None, weekday tuple).

    Raises ValueError with a readable message on bad input.
    """
    from datetime import date

    from notetaker.dates import parse_weekdays

    lecture_date = None
    if args.lecture_date:
        if args.lecture_date.lower() == "today":
            lecture_date = date.today()
        else:
            try:
                lecture_date = date.fromisoformat(args.lecture_date)
            except ValueError:
                raise ValueError(
                    f"--lecture-date must be YYYY-MM-DD or 'today', "
                    f"got {args.lecture_date!r}"
                ) from None
    class_days = parse_weekdays(args.class_days) if args.class_days else ()
    return lecture_date, class_days


def show_config() -> int:
    """Print every setting as `name = value`, with secrets masked."""
    try:
        settings = get_settings()
    except ValidationError as error:
        # A bad value in the environment or .env: explain it instead of
        # dumping a long traceback. Exit code 1 signals failure to the shell.
        print(f"Invalid configuration:\n{error}")
        return 1
    for name, value in settings.redacted().items():
        print(f"{name} = {value}")
    return 0


def print_chunk_progress(index: int, total: int, chunk: Chunk) -> None:
    """Show which piece of a long recording is being transcribed.

    Passed to transcribe() as a "callback": a function handed to another
    function, which calls it back at the right moments (here, once per chunk).
    Silent for short files that fit in a single chunk.
    """
    from notetaker.export import format_timestamp

    if total > 1:
        print(
            f"  chunk {index}/{total} "
            f"[{format_timestamp(chunk.start)} - {format_timestamp(chunk.end)}]"
        )


def run_transcribe(args: argparse.Namespace) -> int:
    """Handle `notetaker transcribe`: transcribe one file and write outputs."""
    # Imported here so `notetaker --version` and `notetaker config` do not pay
    # the import cost of the transcription stack.
    from notetaker.export import write_outputs
    from notetaker.transcribe import transcribe

    audio_path: Path = args.audio
    if not audio_path.is_file():
        print(f"Error: file not found: {audio_path}")
        return 1
    if args.beam_size < 1:
        print("Error: --beam-size must be at least 1")
        return 1
    if args.max_chunk_seconds is not None and args.max_chunk_seconds < 0:
        print("Error: --max-chunk-seconds must be 0 or more")
        return 1
    try:
        settings = get_settings()
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1

    # `.stem` is the file name without its extension: lecture01.mp3 -> lecture01.
    # One folder per recording keeps notes, flashcards, etc. from later steps
    # next to the transcript they came from.
    out_dir = args.output_dir or settings.output_dir / audio_path.stem
    model_name = args.model or settings.whisper_model
    print(f"Transcribing {audio_path} with whisper-{model_name} ...")
    try:
        transcript = transcribe(
            audio_path,
            language=args.language,
            beam_size=args.beam_size,
            model_name=model_name,
            max_chunk_s=args.max_chunk_seconds,
            on_chunk=print_chunk_progress,
        )
    except AudioError as error:
        # Covers both "ffmpeg is not installed" and "ffmpeg could not read
        # this file". The message already says what to do.
        print(f"Error: {error}")
        return 1

    json_path, md_path = write_outputs(
        transcript, out_dir, title=args.title or audio_path.stem
    )
    print(
        f"Done: {len(transcript.segments)} segments, "
        f"{transcript.duration:.1f} s of audio, language '{transcript.language}'."
    )
    print(f"  {json_path}")
    print(f"  {md_path}")
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    """Handle `notetaker benchmark`: time each model and write a report."""
    from notetaker.benchmark import (
        WHISPER_SAMPLE_RATE,
        benchmark_model,
        load_benchmark_audio,
        run_isolated,
        write_report,
    )

    audio_path: Path = args.audio
    if not audio_path.is_file():
        print(f"Error: file not found: {audio_path}")
        return 1
    if args.repeats < 1 or args.beam_size < 1:
        print("Error: --repeats and --beam-size must be at least 1")
        return 1
    try:
        get_settings()
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1

    options = {
        "repeats": args.repeats,
        "beam_size": args.beam_size,
        "language": args.language,
    }
    results = []
    try:
        # Decode once up front: this validates the file before any model loads,
        # and the in-process mode reuses the waveform for every model.
        audio = load_benchmark_audio(audio_path, args.max_seconds)
        seconds = len(audio) / WHISPER_SAMPLE_RATE
        print(f"Benchmarking {audio_path} ({seconds:.1f} s of audio)")
        for name in args.models:
            print(f"  whisper-{name} ...", flush=True)
            try:
                if args.no_isolate:
                    result = benchmark_model(audio, name, **options)
                else:
                    result = run_isolated(
                        audio_path, name, max_seconds=args.max_seconds, **options
                    )
            except Exception as error:  # noqa: BLE001
                # One model failing (most often: its weights could not be
                # downloaded, or it ran out of memory) should not throw away
                # the numbers already measured for the others.
                print(f"    failed: {type(error).__name__}: {error}")
                continue
            results.append(result)
            print(
                f"    load {result.load_seconds:.1f} s, "
                f"transcribe {result.median_transcribe_seconds:.1f} s, "
                f"RTF {result.rtf:.3f} ({result.speedup:.1f}x real time)"
            )
    except AudioError as error:
        print(f"Error: {error}")
        return 1

    if not results:
        print("Error: no model could be benchmarked (see messages above)")
        return 1
    json_path, md_path = write_report(results, args.output_dir, audio_path.name)
    print(f"Report written:\n  {json_path}\n  {md_path}")
    return 0


def run_evalset(args: argparse.Namespace) -> int:
    """Handle `notetaker evalset`: download the evaluation lectures."""
    from notetaker.evalset import (
        EvalSetError,
        download_lecture,
        load_manifest,
        select_lectures,
    )

    try:
        specs = select_lectures(load_manifest(args.manifest), args.only)
    except EvalSetError as error:
        print(f"Error: {error}")
        return 1
    if args.list:
        for spec in specs:
            print(f"{spec.id:<20} {spec.course}: {spec.title}")
        return 0
    try:
        out_dir = args.output_dir or get_settings().data_dir / "raw" / "ocw"
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1

    failures = 0
    for spec in specs:
        print(f"{spec.id}: {spec.title}", flush=True)
        try:
            files = download_lecture(
                spec, out_dir, force=args.force, with_audio=not args.no_audio
            )
        except Exception as error:  # noqa: BLE001
            # Network and ffmpeg errors come in many types. One broken lecture
            # should not stop the others, so report it and carry on.
            print(f"  failed: {error}")
            failures += 1
            continue
        if files.skipped:
            print(f"  already downloaded in {files.folder} (use --force to redo)")
        else:
            print(f"  saved {files.num_cues} caption cues to {files.folder}")
    done = len(specs) - failures
    print(f"{done}/{len(specs)} lectures ready in {out_dir}")
    return 0 if failures == 0 else 1


def run_normalize(args: argparse.Namespace) -> int:
    """Handle `notetaker normalize`: show exactly what the WER scorer will see."""
    from notetaker.normalize import load_text, normalize_text

    try:
        text = load_text(args.file)
    except (OSError, ValueError) as error:
        # OSError: missing/unreadable file. ValueError: bad JSON (the
        # json module's JSONDecodeError is a subclass of ValueError).
        print(f"Error: {error}")
        return 1
    print(
        normalize_text(
            text,
            numbers_to_words=not args.keep_digits,
            remove_fillers=not args.keep_fillers,
        )
    )
    return 0


def _print_score(score) -> None:
    """One-line summary of a PairScore."""
    w = score.words
    print(
        f"{score.name}: WER {100 * score.wer:.2f}% "
        f"(S={w.substitutions} D={w.deletions} I={w.insertions} "
        f"of {w.reference_length} words), CER {100 * score.cer:.2f}%"
    )


def run_wer(args: argparse.Namespace) -> int:
    """Handle `notetaker wer`: score one pair, or a whole folder of lectures."""
    from notetaker.wer import SystemReport, find_pairs, score_files, write_reports

    if args.max_seconds is not None and args.max_seconds <= 0:
        print("Error: --max-seconds must be more than 0")
        return 1
    options = {"max_seconds": args.max_seconds, "normalize": not args.no_normalize}

    # Pair mode: exactly two files, no folder.
    if args.files:
        if len(args.files) != 2 or args.hyps is not None:
            print("Error: give either two files (REF HYP) or --hyps FOLDER")
            return 1
        ref, hyp = args.files
        try:
            score = score_files(ref, hyp, name=hyp.name, **options)
        except (OSError, ValueError) as error:
            print(f"Error: {error}")
            return 1
        _print_score(score)
        return 0

    # Folder mode.
    if args.hyps is None:
        print("Error: give either two files (REF HYP) or --hyps FOLDER")
        return 1
    try:
        refs = args.refs or get_settings().data_dir / "raw" / "ocw"
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1
    pairs, missing = find_pairs(refs, args.hyps)
    for lecture_id in missing:
        print(f"{lecture_id}: no {args.hyps / lecture_id / 'transcript.json'}, skipped")
    if not pairs:
        print(f"Error: no lecture in {refs} has a transcript in {args.hyps}")
        return 1

    scores = []
    for lecture_id, ref, hyp in pairs:
        try:
            score = score_files(ref, hyp, name=lecture_id, **options)
        except (OSError, ValueError) as error:
            print(f"{lecture_id}: failed: {error}")
            continue
        _print_score(score)
        scores.append(score)
    if not scores:
        print("Error: no lecture could be scored (see messages above)")
        return 1

    system = args.system or args.hyps.name
    if args.no_normalize:
        system += "-raw"
    report = SystemReport(system=system, scores=scores)
    print(
        f"Pooled over {len(scores)} lecture(s): WER {100 * report.wer:.2f}%, "
        f"CER {100 * report.cer:.2f}%"
    )
    paths = write_reports(report, args.output_dir)
    print("Report written:\n" + "\n".join(f"  {p}" for p in paths))
    return 0


def _print_term_score(score) -> None:
    """One-line summary of a TermScore, plus its three worst misses."""
    c = score.counts
    line = (
        f"{score.name}: term recall {100 * score.recall:.1f}% "
        f"({c.total_hits} of {c.total_reference} said), "
        f"precision {100 * score.precision:.1f}%"
    )
    worst = c.misses()[:3]
    if worst:
        line += "; most missed: " + ", ".join(f"{t} ({g}/{s})" for t, s, g in worst)
    print(line)


def run_terms(args: argparse.Namespace) -> int:
    """Handle `notetaker terms`: technical-term recall, pair or folder mode."""
    from notetaker.term_recall import (
        TermReport,
        load_terms,
        score_term_files,
        write_reports,
    )
    from notetaker.wer import find_pairs

    if args.max_seconds is not None and args.max_seconds <= 0:
        print("Error: --max-seconds must be more than 0")
        return 1
    try:
        terms = load_terms(args.terms)
    except (OSError, ValueError) as error:
        print(f"Error: could not load term list: {error}")
        return 1

    if args.files:
        if len(args.files) != 2 or args.hyps is not None:
            print("Error: give either two files (REF HYP) or --hyps FOLDER")
            return 1
        ref, hyp = args.files
        try:
            score = score_term_files(
                ref, hyp, terms, name=hyp.name, max_seconds=args.max_seconds
            )
        except (OSError, ValueError) as error:
            print(f"Error: {error}")
            return 1
        _print_term_score(score)
        return 0

    if args.hyps is None:
        print("Error: give either two files (REF HYP) or --hyps FOLDER")
        return 1
    try:
        refs = args.refs or get_settings().data_dir / "raw" / "ocw"
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1
    pairs, missing = find_pairs(refs, args.hyps)
    for lecture_id in missing:
        print(f"{lecture_id}: no {args.hyps / lecture_id / 'transcript.json'}, skipped")
    if not pairs:
        print(f"Error: no lecture in {refs} has a transcript in {args.hyps}")
        return 1

    scores = []
    for lecture_id, ref, hyp in pairs:
        try:
            score = score_term_files(
                ref, hyp, terms, name=lecture_id, max_seconds=args.max_seconds
            )
        except (OSError, ValueError) as error:
            print(f"{lecture_id}: failed: {error}")
            continue
        _print_term_score(score)
        scores.append(score)
    if not scores:
        print("Error: no lecture could be scored (see messages above)")
        return 1

    report = TermReport(
        system=args.system or args.hyps.name,
        scores=scores,
        domains={t.name: t.domain for t in terms},
    )
    print(
        f"Pooled over {len(scores)} lecture(s): term recall "
        f"{100 * report.recall:.1f}%, precision {100 * report.precision:.1f}%"
    )
    paths = write_reports(report, args.output_dir)
    print("Report written:\n" + "\n".join(f"  {p}" for p in paths))
    return 0


def run_baseline(args: argparse.Namespace) -> int:
    """Handle `notetaker baseline`: the model size vs WER vs speed report."""
    from notetaker.baseline import (
        build_rows,
        load_benchmark,
        load_term_reports,
        load_wer_reports,
        recommend,
        write_baseline,
    )

    if args.rtf_budget <= 0:
        print("Error: --rtf-budget must be more than 0")
        return 1
    wer_reports, warnings = load_wer_reports(args.wer)
    if not wer_reports:
        print(
            f"Error: no WER reports in {args.wer}. Run `notetaker wer --hyps ...` "
            "for at least one model first."
        )
        return 1
    term_reports, term_warnings = load_term_reports(args.terms)
    warnings += term_warnings
    try:
        bench, bench_audio = load_benchmark(args.benchmark)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Warning: could not read {args.benchmark}: {error}")
        bench, bench_audio = [], None
    if not term_reports:
        print(f"Note: no term reports in {args.terms}; term columns will be n/a")
    if not bench:
        print(f"Note: no benchmark at {args.benchmark}; speed columns will be n/a")

    rows, join_warnings = build_rows(wer_reports, term_reports, bench)
    warnings += join_warnings
    for warning in warnings:
        print(f"Warning: {warning}")
    _, reason = recommend(rows, args.rtf_budget)
    print(f"{len(rows)} model(s). Recommendation: {reason}")
    paths = write_baseline(
        rows,
        args.output,
        rtf_budget=args.rtf_budget,
        bench_audio=bench_audio,
        warnings=warnings,
    )
    print("Report written:\n" + "\n".join(f"  {p}" for p in paths))
    return 0


def run_llm(args: argparse.Namespace, transport=None) -> int:
    """Handle `notetaker llm`: one prompt in, one reply out.

    `transport` is only for tests (a fake network call); real runs leave it
    as None and the client posts over HTTP.
    """
    from notetaker.llm import LLMError, make_client

    try:
        settings = get_settings()
        client = make_client(
            settings, provider=args.provider, model=args.model, transport=transport
        )
        reply = client.complete(
            args.prompt,
            system=args.system,
            json_mode=args.json,
            max_tokens=args.max_tokens,
        )
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1
    except LLMError as error:
        print(f"Error: {error}")
        return 1
    if args.json:
        try:
            parsed = reply.json()
        except ValueError:
            print(reply.text)
            print("Error: the reply was not valid JSON")
            return 1
        import json

        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    else:
        print(reply.text)
    tokens = (
        f"{reply.prompt_tokens} in / {reply.completion_tokens} out tokens"
        if reply.total_tokens is not None
        else "token counts n/a"
    )
    print(
        f"\n[{reply.provider} · {reply.model} · {tokens} · "
        f"{reply.latency_s:.2f}s · {reply.attempts} attempt(s)]"
    )
    return 0


def run_notes(args: argparse.Namespace, client=None) -> int:
    """Handle `notetaker notes`: transcript.json in, notes.md + notes.json out.

    `client` is only for tests (a FakeLLMClient); real runs build the client
    from the settings.
    """
    from notetaker.llm import LLMError, make_client
    from notetaker.notes import NotesError, generate_notes, write_notes
    from notetaker.transcribe import Transcript

    if args.chunk_words <= 0 or args.reduce_words <= 0 or args.max_repairs < 0:
        print(
            "Error: --chunk-words and --reduce-words must be > 0 and --max-repairs >= 0"
        )
        return 1
    try:
        transcript = Transcript.load_json(args.transcript)
    except FileNotFoundError:
        print(f"Error: no such file: {args.transcript}")
        return 1
    except (ValueError, KeyError, TypeError) as error:
        print(f"Error: {args.transcript} is not a transcript.json ({error})")
        return 1

    out_dir = args.output_dir or args.transcript.parent
    title = args.title or f"Notes: {args.transcript.parent.name or 'lecture'}"

    def progress(index: int, total: int, chunk) -> None:
        from notetaker.export import format_timestamp

        print(
            f"  chunk {index + 1}/{total} "
            f"[{format_timestamp(chunk.start)}-{format_timestamp(chunk.end)}] "
            f"{chunk.word_count} words",
            flush=True,
        )

    try:
        if client is None:
            client = make_client(
                get_settings(), provider=args.provider, model=args.model
            )
        print(f"Writing notes with {client.provider} · {client.model}")
        notes = generate_notes(
            transcript,
            client,
            title=title,
            max_words=args.chunk_words,
            max_repairs=args.max_repairs,
            on_chunk=progress,
            reduce=not args.no_reduce,
            reduce_words=args.reduce_words,
        )
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1
    except (LLMError, NotesError) as error:
        print(f"Error: {error}")
        return 1
    json_path, md_path = write_notes(notes, out_dir)
    if notes.reduce_error:
        # Not fatal: the per-chunk notes were still written.
        print(f"Warning: could not merge into topics ({notes.reduce_error})")
    elif notes.detail_sections:
        print(
            f"Merged {len(notes.detail_sections)} chunk section(s) into "
            f"{len(notes.sections)} topic(s) with {notes.reduce_calls} call(s)"
        )
    print(
        f"{len(notes.sections)} section(s) from {notes.chunks} chunk(s), "
        f"{notes.llm_calls} LLM call(s), {notes.latency_s:.1f}s"
    )
    print(f"Wrote {md_path}\nWrote {json_path}")
    return 0


def run_concepts(args: argparse.Namespace, client=None) -> int:
    """Handle `notetaker concepts`: transcript.json in, concepts.md + .json out.

    `client` is only for tests (a FakeLLMClient).
    """
    import json

    from notetaker.concepts import (
        ConceptsError,
        concepts_json_schema,
        extract_concepts,
        write_concepts,
    )
    from notetaker.llm import LLMError, make_client
    from notetaker.transcribe import Transcript

    if args.schema:
        print(json.dumps(concepts_json_schema(), indent=2))
        return 0
    if args.transcript is None:
        print("Error: give a transcript.json (or --schema)")
        return 1
    if args.chunk_words <= 0 or args.max_repairs < 0:
        print("Error: --chunk-words must be > 0 and --max-repairs >= 0")
        return 1
    try:
        transcript = Transcript.load_json(args.transcript)
    except FileNotFoundError:
        print(f"Error: no such file: {args.transcript}")
        return 1
    except (ValueError, KeyError, TypeError) as error:
        print(f"Error: {args.transcript} is not a transcript.json ({error})")
        return 1

    out_dir = args.output_dir or args.transcript.parent
    title = args.title or f"Key concepts: {args.transcript.parent.name or 'lecture'}"

    def progress(index: int, total: int, chunk) -> None:
        from notetaker.export import format_timestamp

        print(
            f"  chunk {index + 1}/{total} "
            f"[{format_timestamp(chunk.start)}-{format_timestamp(chunk.end)}] "
            f"{chunk.word_count} words",
            flush=True,
        )

    try:
        if client is None:
            client = make_client(
                get_settings(), provider=args.provider, model=args.model
            )
        print(f"Extracting concepts with {client.provider} · {client.model}")
        concepts = extract_concepts(
            transcript,
            client,
            title=title,
            max_words=args.chunk_words,
            max_repairs=args.max_repairs,
            on_chunk=progress,
        )
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1
    except (LLMError, ConceptsError) as error:
        print(f"Error: {error}")
        return 1
    json_path, md_path = write_concepts(concepts, out_dir)
    if concepts.dropped_ungrounded:
        print(
            f"Dropped {len(concepts.dropped_ungrounded)} term(s) not found in the "
            f"transcript: {', '.join(concepts.dropped_ungrounded)}"
        )
    print(
        f"{len(concepts.concepts)} concept(s) from {concepts.chunks} chunk(s), "
        f"{concepts.llm_calls} LLM call(s), {concepts.latency_s:.1f}s"
    )
    print(f"Wrote {md_path}\nWrote {json_path}")
    return 0


def run_flashcards(args: argparse.Namespace, client=None) -> int:
    """Handle `notetaker flashcards`: transcript (and/or concepts) in, cards out.

    `client` is only for tests (a FakeLLMClient).
    """
    import json

    from notetaker.concepts import load_concepts
    from notetaker.flashcards import (
        FlashcardsError,
        flashcards_json_schema,
        generate_flashcards,
        write_flashcards,
    )
    from notetaker.llm import LLMError, make_client
    from notetaker.transcribe import Transcript

    if args.schema:
        print(json.dumps(flashcards_json_schema(), indent=2))
        return 0
    if args.concepts_only and args.concepts is None:
        print("Error: --concepts-only needs --concepts concepts.json")
        return 1
    if args.transcript is None and not args.concepts_only:
        print("Error: give a transcript.json (or --concepts-only, or --schema)")
        return 1
    if args.chunk_words <= 0 or args.max_repairs < 0 or args.max_cards <= 0:
        print("Error: --chunk-words and --max-cards must be > 0, --max-repairs >= 0")
        return 1

    concepts = None
    if args.concepts is not None:
        try:
            concepts = load_concepts(args.concepts)
        except FileNotFoundError:
            print(f"Error: no such file: {args.concepts}")
            return 1
        except ValueError as error:
            print(f"Error: {error}")
            return 1

    transcript = None
    if not args.concepts_only:
        try:
            transcript = Transcript.load_json(args.transcript)
        except FileNotFoundError:
            print(f"Error: no such file: {args.transcript}")
            return 1
        except (ValueError, KeyError, TypeError) as error:
            print(f"Error: {args.transcript} is not a transcript.json ({error})")
            return 1

    source = args.concepts if args.concepts_only else args.transcript
    out_dir = args.output_dir or source.parent
    title = args.title or f"Flashcards: {source.parent.name or 'lecture'}"

    def progress(index: int, total: int, chunk) -> None:
        from notetaker.export import format_timestamp

        print(
            f"  chunk {index + 1}/{total} "
            f"[{format_timestamp(chunk.start)}-{format_timestamp(chunk.end)}] "
            f"{chunk.word_count} words",
            flush=True,
        )

    try:
        if args.concepts_only:
            client = None
        elif client is None:
            client = make_client(
                get_settings(), provider=args.provider, model=args.model
            )
        if client is not None:
            print(f"Writing flashcards with {client.provider} · {client.model}")
        deck = generate_flashcards(
            transcript,
            client,
            concepts=concepts,
            title=title,
            max_words=args.chunk_words,
            max_cards=args.max_cards,
            max_repairs=args.max_repairs,
            on_chunk=progress,
        )
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1
    except (LLMError, FlashcardsError) as error:
        print(f"Error: {error}")
        return 1
    paths = write_flashcards(
        deck, out_dir, deck_name=args.deck, anki_header=not args.plain_csv
    )
    if deck.dropped_ungrounded:
        print(
            f"Dropped {len(deck.dropped_ungrounded)} card(s) whose answers are "
            "not supported by the transcript"
        )
    print(
        f"{len(deck.cards)} card(s) ({deck.dropped_duplicates} duplicate(s) "
        f"removed), {deck.llm_calls} LLM call(s), {deck.latency_s:.1f}s"
    )
    print("\n".join(f"Wrote {p}" for p in paths))
    return 0


def run_quiz(args: argparse.Namespace, client=None) -> int:
    """Handle `notetaker quiz`: transcript (and/or concepts) in, quiz out.

    Same input handling as run_flashcards. `client` is only for tests.
    """
    import json

    from notetaker.concepts import load_concepts
    from notetaker.llm import LLMError, make_client
    from notetaker.quiz import QuizError, generate_quiz, quiz_json_schema, write_quiz
    from notetaker.transcribe import Transcript

    if args.schema:
        print(json.dumps(quiz_json_schema(), indent=2))
        return 0
    if args.concepts_only and args.concepts is None:
        print("Error: --concepts-only needs --concepts concepts.json")
        return 1
    if args.transcript is None and not args.concepts_only:
        print("Error: give a transcript.json (or --concepts-only, or --schema)")
        return 1
    if args.chunk_words <= 0 or args.max_repairs < 0 or args.max_questions <= 0:
        print(
            "Error: --chunk-words and --max-questions must be > 0, --max-repairs >= 0"
        )
        return 1

    concepts = None
    if args.concepts is not None:
        try:
            concepts = load_concepts(args.concepts)
        except FileNotFoundError:
            print(f"Error: no such file: {args.concepts}")
            return 1
        except ValueError as error:
            print(f"Error: {error}")
            return 1

    transcript = None
    if not args.concepts_only:
        try:
            transcript = Transcript.load_json(args.transcript)
        except FileNotFoundError:
            print(f"Error: no such file: {args.transcript}")
            return 1
        except (ValueError, KeyError, TypeError) as error:
            print(f"Error: {args.transcript} is not a transcript.json ({error})")
            return 1

    source = args.concepts if args.concepts_only else args.transcript
    out_dir = args.output_dir or source.parent
    title = args.title or f"Quiz: {source.parent.name or 'lecture'}"

    def progress(index: int, total: int, chunk) -> None:
        from notetaker.export import format_timestamp

        print(
            f"  chunk {index + 1}/{total} "
            f"[{format_timestamp(chunk.start)}-{format_timestamp(chunk.end)}] "
            f"{chunk.word_count} words",
            flush=True,
        )

    try:
        if args.concepts_only:
            client = None
        elif client is None:
            client = make_client(
                get_settings(), provider=args.provider, model=args.model
            )
        if client is not None:
            print(f"Writing quiz questions with {client.provider} · {client.model}")
        quiz = generate_quiz(
            transcript,
            client,
            concepts=concepts,
            title=title,
            max_words=args.chunk_words,
            max_questions=args.max_questions,
            max_repairs=args.max_repairs,
            seed=args.seed,
            on_chunk=progress,
        )
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1
    except (LLMError, QuizError) as error:
        print(f"Error: {error}")
        return 1
    paths = write_quiz(quiz, out_dir)
    if quiz.dropped_ungrounded:
        print(
            f"Dropped {len(quiz.dropped_ungrounded)} question(s) whose answers "
            "are not supported by the transcript"
        )
    if concepts is not None and len(concepts.concepts) < 4:
        print("Note: glossary questions need at least 4 concepts; none were made")
    rate = quiz.longest_option_rate()
    cue = "" if rate is None else f", key is the longest option in {rate:.0%}"
    print(
        f"{len(quiz.questions)} question(s) ({quiz.dropped_duplicates} "
        f"duplicate(s) removed{cue}), {quiz.llm_calls} LLM call(s), "
        f"{quiz.latency_s:.1f}s"
    )
    print("\n".join(f"Wrote {p}" for p in paths))
    return 0


def run_faithfulness(args: argparse.Namespace, client=None) -> int:
    """Handle `notetaker faithfulness`: generated file + transcript in, report out.

    `client` is only for tests (a FakeLLMClient instead of a real judge).
    """
    import json

    from notetaker.faithfulness import (
        evaluate_faithfulness,
        faithfulness_json_schema,
        load_target,
        write_report,
    )
    from notetaker.llm import LLMError, make_client
    from notetaker.transcribe import Transcript

    if args.schema:
        print(json.dumps(faithfulness_json_schema(), indent=2))
        return 0
    if args.target is None:
        print("Error: give a notes/concepts/flashcards/quiz .json (or --schema)")
        return 1
    if (
        args.batch_size <= 0
        or args.top_k <= 0
        or args.max_repairs < 0
        or (args.max_claims is not None and args.max_claims <= 0)
    ):
        print(
            "Error: --batch-size, --top-k and --max-claims must be > 0, "
            "--max-repairs >= 0"
        )
        return 1

    try:
        target = load_target(args.target)
    except FileNotFoundError:
        print(f"Error: no such file: {args.target}")
        return 1
    except ValueError as error:  # also covers invalid JSON (JSONDecodeError)
        print(f"Error: {error}")
        return 1

    transcript_path = args.transcript or args.target.parent / "transcript.json"
    try:
        transcript = Transcript.load_json(transcript_path)
    except FileNotFoundError:
        print(f"Error: no such file: {transcript_path} (pass it with -t)")
        return 1
    except (ValueError, KeyError, TypeError) as error:
        print(f"Error: {transcript_path} is not a transcript.json ({error})")
        return 1

    def progress(index: int, total: int) -> None:
        print(f"  judging batch {index + 1}/{total}", flush=True)

    try:
        if args.lexical_only:
            client = None
            print("Lexical baseline only (no LLM calls)")
        else:
            if client is None:
                client = make_client(
                    get_settings(), provider=args.provider, model=args.model
                )
            print(f"Judging {target.kind} with {client.provider} · {client.model}")
            if (client.provider, client.model) == (target.provider, target.model):
                print(
                    "Warning: the judge is the model that generated this file; "
                    "LLM judges favour their own writing. Pick another model "
                    "with -p/-m for a fairer score."
                )
        report = evaluate_faithfulness(
            target,
            transcript,
            client,
            target_path=str(args.target),
            max_claims=args.max_claims,
            batch_size=args.batch_size,
            top_k=args.top_k,
            max_repairs=args.max_repairs,
            on_batch=progress,
        )
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1
    except LLMError as error:
        print(f"Error: {error}")
        return 1

    paths = write_report(report, args.output_dir or args.target.parent)
    s = report.summary

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.1%}"

    counts = ", ".join(f"{n} {v}" for v, n in s.counts.items() if n)
    print(
        f"{s.claims} claim(s): {counts or 'none'}. Faithfulness {pct(s.faithfulness)} "
        f"(strict {pct(s.strict_faithfulness)}), hallucination rate "
        f"{pct(s.hallucination_rate)}"
    )
    if report.batch_errors:
        print(f"{len(report.batch_errors)} batch(es) could not be judged; see report")
    if client is not None:
        print(f"{report.llm_calls} LLM call(s), {report.latency_s:.1f}s")
    print("\n".join(f"Wrote {p}" for p in paths))
    return 0


def _print_tasks_summary(tasks) -> None:
    """Shared by `tasks` and `agent`: counts plus one line per task."""
    print(
        f"Found {len(tasks.tasks)} task(s) in "
        f"{len(tasks.candidates)} candidate sentence(s)"
    )
    for task in tasks.tasks:
        if task.due_date is not None:
            due = task.due_date.isoformat() + (
                f" {task.due_time}" if task.due_time else ""
            )
            due += " (?)" if task.date_ambiguous else ""
        else:
            due = f'"{task.due_phrase}"' if task.due_phrase else "no deadline"
        print(f"  - {task.title} ({task.kind}): {due}")
    for warning in tasks.warnings:
        print(f"Warning: {warning}")


def run_tasks(args: argparse.Namespace, client=None) -> int:
    """Handle `notetaker tasks`: transcript in, dated to-do list out.

    `client` is only for tests (a FakeLLMClient).
    """
    from notetaker.llm import LLMError, make_client
    from notetaker.tasks import extract_tasks, write_tasks
    from notetaker.todo import write_todo
    from notetaker.transcribe import Transcript

    if args.batch_size <= 0 or args.max_repairs < 0:
        print("Error: --batch-size must be > 0 and --max-repairs >= 0")
        return 1
    try:
        lecture_date, class_days = _parse_task_dates(args)
    except ValueError as error:
        print(f"Error: {error}")
        return 1
    try:
        transcript = Transcript.load_json(args.transcript)
    except FileNotFoundError:
        print(f"Error: no such file: {args.transcript}")
        return 1
    except (ValueError, KeyError, TypeError) as error:
        print(f"Error: {args.transcript} is not a transcript.json ({error})")
        return 1

    out_dir = args.output_dir or args.transcript.parent
    title = args.title or f"Tasks: {args.transcript.parent.name or 'lecture'}"
    try:
        if args.no_llm:
            client = None
            print("Extracting tasks with offline rules (no LLM)")
        else:
            if client is None:
                client = make_client(
                    get_settings(), provider=args.provider, model=args.model
                )
            print(f"Extracting tasks with {client.provider} · {client.model}")
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1
    except LLMError as error:
        print(f"Error: {error}")
        return 1

    tasks = extract_tasks(
        transcript,
        client,
        title=title,
        lecture_date=lecture_date,
        class_days=class_days,
        use_llm=not args.no_llm,
        batch_size=args.batch_size,
        max_repairs=args.max_repairs,
        on_batch=lambda i, n: print(f"  batch {i + 1}/{n}", flush=True),
    )
    paths = write_tasks(tasks, out_dir) + write_todo(tasks, out_dir)
    _print_tasks_summary(tasks)
    if tasks.llm_calls:
        print(f"{tasks.llm_calls} LLM call(s), {tasks.latency_s:.1f}s")
    print("\n".join(f"Wrote {p}" for p in paths))
    return 0


def run_todo(args: argparse.Namespace) -> int:
    """Handle `notetaker todo`: tasks.json in, todo.json + tasks.ics out."""
    from notetaker.tasks import load_tasks
    from notetaker.todo import IcsOptions, calendar_count, write_todo

    try:
        options = IcsOptions(
            component="todo" if args.vtodo else "event",
            event_minutes=args.event_minutes,
            reminder_minutes=None if args.no_reminder else args.reminder_minutes,
        )
    except ValueError as error:
        print(f"Error: {error}")
        return 1
    try:
        tasks = load_tasks(args.tasks)
    except FileNotFoundError:
        print(f"Error: no such file: {args.tasks}")
        return 1
    except ValueError as error:  # pydantic's ValidationError is a ValueError
        print(f"Error: {args.tasks} is not a tasks.json ({error})")
        return 1
    paths = write_todo(tasks, args.output_dir or args.tasks.parent, options)
    on_calendar = calendar_count(tasks.tasks)
    print(
        f"{len(tasks.tasks)} task(s): {on_calendar} on the calendar, "
        f"{len(tasks.tasks) - on_calendar} without a date (todo.json only)"
    )
    print("\n".join(f"Wrote {p}" for p in paths))
    return 0


def run_ingest(args: argparse.Namespace) -> int:
    """Handle `notetaker ingest`: a folder of transcripts in, a search index out."""
    from notetaker.embeddings import EmbeddingError
    from notetaker.ingest import (
        FAILED,
        MISSING,
        IngestError,
        LectureResult,
        SplitOptions,
        ingest_folder,
        load_titles,
    )
    from notetaker.vectorstore import VectorStoreError

    # One symbol per outcome, so a long run is easy to scan.
    marks = {"added": "+", "updated": "~", "unchanged": "=", "removed": "-"}

    def show(result: LectureResult) -> None:
        mark = marks.get(result.status, "!")
        line = f"  {mark} {result.lecture_id}: {result.status}"
        if result.status not in (FAILED, MISSING):
            line += f", {result.passages} passage(s)"
        if result.reason and result.status != "added":
            line += f" ({result.reason})"
        print(line, flush=True)

    try:
        options = SplitOptions(max_words=args.max_words, max_seconds=args.max_seconds)
        titles = load_titles(args.titles)
        print(f"Indexing {args.folder} into {args.index}/")
        report = ingest_folder(
            args.folder,
            args.index,
            embedder_name=args.embedder,
            pattern=args.pattern,
            titles=titles,
            options=options,
            rebuild=args.rebuild,
            prune=args.prune,
            dry_run=args.dry_run,
            on_result=show,
        )
    # ValueError covers bad passage sizes (e.g. --max-words 0) from the store.
    except (IngestError, VectorStoreError, EmbeddingError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    print(f"Embedding model: {report.embedder}")
    prefix = "Dry run, nothing saved: " if report.dry_run else ""
    print(prefix + report.summary())
    if report.saved:
        print(f"Wrote {args.index}/")
    return 0 if report.ok else 1


def run_ask(args: argparse.Namespace, client=None) -> int:
    """Handle `notetaker ask`: a question in, a cited answer out.

    `client` is only for tests (a FakeLLMClient).
    """
    import json
    import sys

    from notetaker.ask import AskError, ask
    from notetaker.cassette import (
        CassetteFormatError,
        CassetteMiss,
        RecordingClient,
        ReplayClient,
    )
    from notetaker.embeddings import EmbeddingError
    from notetaker.llm import LLMError, make_client
    from notetaker.vectorstore import VectorStore, VectorStoreError

    if args.top_k <= 0 or args.max_repairs < 0:
        print("Error: --top-k must be > 0 and --max-repairs >= 0")
        return 1
    try:
        store = VectorStore.load(args.index)
    except (VectorStoreError, EmbeddingError) as error:
        print(f"Error: {error}")
        if not (args.index / "manifest.json").exists():
            print("Build one first with: notetaker ingest <folder> --index", args.index)
        return 1
    if args.lecture:
        unknown = sorted(set(args.lecture) - set(store.lectures()))
        if unknown:
            print(f"Error: not in the index: {', '.join(unknown)}")
            print(f"Indexed lectures: {', '.join(store.lectures()) or '(none)'}")
            return 1

    recorder = None
    try:
        if args.no_llm:
            client = None
        elif args.replay_llm is not None:
            client = ReplayClient(args.replay_llm)
        elif client is None:
            client = make_client(
                get_settings(), provider=args.provider, model=args.model
            )
        if args.record_llm is not None and client is not None:
            recorder = client = RecordingClient(client, args.record_llm)
    except FileNotFoundError:
        print(f"Error: no such cassette: {args.replay_llm}")
        return 1
    except CassetteFormatError as error:
        print(f"Error: {error}")
        return 1
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1
    except LLMError as error:
        print(f"Error: {error}")
        return 1

    try:
        answer = ask(
            store,
            args.question,
            client,
            k=args.top_k,
            lecture_ids=args.lecture,
            min_score=args.min_score,
            use_llm=not args.no_llm,
            max_repairs=args.max_repairs,
        )
    except (AskError, CassetteMiss, LLMError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    finally:
        if recorder is not None and recorder.cassette.interactions:
            recorder.save()

    # With --json, stdout holds only the JSON record (so it can be piped into
    # another program); status lines like "Wrote ..." go to stderr instead.
    def status(message: str) -> None:
        print(message, file=sys.stderr if args.json else sys.stdout)

    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(answer.to_markdown())
        how = (
            f"{answer.provider} · {answer.model}, {answer.llm_calls} LLM call(s), "
            f"{answer.latency_s:.1f}s"
            if answer.llm_calls
            else f"{answer.mode}, no LLM call"
        )
        print(f"({len(answer.retrieved)} passage(s) retrieved; {how})")
    if recorder is not None and recorder.cassette.interactions:
        status(f"Wrote {recorder.path}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".json":
            content = json.dumps(answer.to_dict(), indent=2, ensure_ascii=False)
        else:
            content = answer.to_markdown()
        args.output.write_text(content + "\n", encoding="utf-8")
        status(f"Wrote {args.output}")
    return 0


def run_eval_retrieval(args: argparse.Namespace) -> int:
    """Handle `notetaker eval-retrieval`: labeled questions in, recall@k and MRR out."""
    from notetaker.embeddings import EmbeddingError
    from notetaker.hybrid import make_retriever
    from notetaker.ingest import IngestError
    from notetaker.retrieval_eval import (
        RetrievalEvalError,
        build_store,
        evaluate,
        load_corpus_titles,
        load_questions,
        to_json,
        to_markdown,
    )
    from notetaker.vectorstore import VectorStore, VectorStoreError

    # Options only hybrid retrievers use; make_retriever ignores them otherwise.
    hybrid_options = {}
    if any(kind.startswith("hybrid") for kind in args.retriever):
        hybrid_options = {"alpha": args.alpha, "rrf_c": args.rrf_c}

    try:
        questions = load_questions(args.questions)
        print(f"{len(questions.questions)} labeled questions from {args.questions}")
        reports = []
        if args.index is not None:
            if args.embedder:
                print(
                    "Note: --embedder is ignored with --index (it keeps its own model)"
                )
            store = VectorStore.load(args.index)
            for kind in args.retriever:
                retriever = make_retriever(store, kind, **hybrid_options)
                settings = {"index": str(args.index), "retriever": kind}
                reports.append(
                    evaluate(retriever, questions, ks=args.k, settings=settings)
                )
        else:
            folder = args.transcripts or questions.corpus
            if folder is None:
                print("Error: give --index or --transcripts (no corpus in the file)")
                return 1
            titles = load_corpus_titles(questions, args.titles)
            # One run per (embedder, passage size) pair. [None] means the
            # model from settings, the same default as `notetaker ingest`.
            # Each pair is then scored with every --retriever. BM25 ignores
            # the embedder, so it is scored once per passage size only.
            bm25_done: set[int] = set()
            for name in args.embedder or [None]:
                for max_words in args.max_words:
                    store = build_store(
                        folder,
                        embedder_name=name,
                        titles=titles,
                        max_words=max_words,
                        max_seconds=args.max_seconds,
                    )
                    for kind in args.retriever:
                        if kind == "bm25":
                            if max_words in bm25_done:
                                continue
                            bm25_done.add(max_words)
                        retriever = make_retriever(store, kind, **hybrid_options)
                        settings = {
                            "max_words": max_words,
                            "max_seconds": args.max_seconds,
                            "retriever": kind,
                        }
                        reports.append(
                            evaluate(retriever, questions, ks=args.k, settings=settings)
                        )
    except (
        RetrievalEvalError,
        IngestError,
        VectorStoreError,
        EmbeddingError,
        ValueError,
    ) as error:
        print(f"Error: {error}")
        return 1

    ks = reports[0].ks
    for rep in reports:
        recall = "  ".join(f"R@{k}={rep.recall(k):.3f}" for k in ks)
        print(
            f"  {rep.label} ({rep.n_passages} passages): {recall}  MRR={rep.mrr():.3f}"
        )
    # Chance only depends on the passage count, so print it once per count.
    for n, rep in {rep.n_passages: rep for rep in reports}.items():
        chance = "  ".join(f"R@{k}={rep.chance_recall[k]:.3f}" for k in ks)
        print(f"  chance ({n} passages): {chance}  MRR={rep.chance_mrr:.3f}")

    if args.show_misses:
        k_show = 5 if 5 in ks else max(ks)
        for rep in reports:
            misses = rep.misses(k_show)
            print(f"Not fully found in the top {k_show} by {rep.label}: {len(misses)}")
            for res in misses:
                rank = res.first_rank or "-"
                print(f"  {res.question.id} (first relevant rank {rank}):")
                print(f"    {res.question.question}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(to_markdown(reports, questions), encoding="utf-8")
        json_path = args.output.with_suffix(".json")
        json_path.write_text(to_json(reports, questions), encoding="utf-8")
        print(f"Wrote {args.output} and {json_path}")
    return 0


def run_agent(args: argparse.Namespace, client=None) -> int:
    """Handle `notetaker agent`: the whole pipeline in one command.

    `client` is only for tests (a FakeLLMClient).
    """
    from notetaker.agent import (
        PipelineOptions,
        pipeline_mermaid,
        run_pipeline,
        write_pipeline,
    )
    from notetaker.cassette import (
        CassetteFormatError,
        CassetteMiss,
        RecordingClient,
        ReplayClient,
    )
    from notetaker.llm import LLMError, make_client
    from notetaker.transcribe import Transcript

    try:
        lecture_date, class_days = _parse_task_dates(args)
        options = PipelineOptions(
            steps=tuple(args.steps or STEP_NAMES),
            chunk_words=args.chunk_words,
            max_repairs=args.max_repairs,
            reduce=not args.no_reduce,
            seed=args.seed,
            lecture_date=lecture_date,
            class_days=class_days,
            task_llm=not args.no_task_llm,
        )
    except ValueError as error:
        print(f"Error: {error}")
        return 1
    if args.graph:
        print(pipeline_mermaid(options))
        return 0
    if args.transcript is None:
        print("Error: give a transcript.json (or --graph)")
        return 1
    try:
        transcript = Transcript.load_json(args.transcript)
    except FileNotFoundError:
        print(f"Error: no such file: {args.transcript}")
        return 1
    except (ValueError, KeyError, TypeError) as error:
        print(f"Error: {args.transcript} is not a transcript.json ({error})")
        return 1

    out_dir = args.output_dir or args.transcript.parent
    title = args.title or (args.transcript.parent.name or "lecture")
    recorder = replayer = None
    try:
        if args.replay_llm is not None and options.needs_llm:
            # Replay needs no provider settings at all: the cassette holds
            # every reply, so this works on a fresh clone with no .env.
            replayer = client = ReplayClient(args.replay_llm)
        elif client is None and options.needs_llm:
            client = make_client(
                get_settings(), provider=args.provider, model=args.model
            )
        if args.record_llm is not None and client is not None:
            recorder = client = RecordingClient(client, args.record_llm)
    except FileNotFoundError:
        print(f"Error: no such cassette: {args.replay_llm}")
        return 1
    except CassetteFormatError as error:
        print(f"Error: {error}")
        return 1
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1
    except LLMError as error:
        print(f"Error: {error}")
        return 1
    if options.needs_llm:
        print(
            f"Running {' -> '.join(options.steps)} with "
            f"{client.provider} · {client.model}"
        )
    else:
        print(f"Running {' -> '.join(options.steps)} (no LLM needed)")

    def progress(step: str) -> None:
        print(f"  [{step}]", flush=True)

    try:
        result = run_pipeline(
            transcript, client, options, title=title, on_step=progress
        )
    except CassetteMiss as error:
        print(f"Error: {error}")
        return 1
    finally:
        # `finally` runs whether or not the pipeline raised, so a recording
        # that stopped halfway is still saved for inspection.
        if recorder is not None:
            print(f"Recorded {len(recorder.cassette.interactions)} LLM call(s)")
            print(f"Wrote {recorder.save()}")
    if replayer is not None and replayer.unused():
        print(
            f"Warning: {len(replayer.unused())} recorded LLM call(s) were never "
            "replayed; the cassette may be out of date"
        )
    paths = write_pipeline(result, out_dir)
    for record in result.log:
        if record["step"] in options.steps:
            calls = record.get("llm_calls")
            extra = f", {calls} LLM call(s)" if calls else ""
            print(
                f"  {record['step']}: {record['status']} "
                f"({record['seconds']:.1f}s{extra})"
            )
    if result.tasks is not None:
        _print_tasks_summary(result.tasks)
    for error in result.errors:
        print(f"Error in {error['step']}: {error['error']}")
    for path in paths:
        print(f"Wrote {path}")
    # Exit code 1 if any stage failed, so scripts and CI can notice, even
    # though the stages that worked have still been written.
    return 0 if result.ok else 1


def run_finetune_data(args: argparse.Namespace) -> int:
    """Handle `notetaker finetune-data`: build the fine-tuning dataset."""
    from notetaker.finetune_data import (
        FinetuneDataError,
        build_dataset,
        summary_markdown,
    )

    try:
        settings = get_settings()
    except ValidationError as error:
        print(f"Invalid configuration:\n{error}")
        return 1
    eval_dir = args.eval_dir or settings.data_dir / "raw" / "ocw"
    out_dir = args.output_dir or settings.data_dir / "finetune"
    action = "Planning" if args.dry_run else "Building"
    print(f"{action} fine-tuning dataset from {eval_dir}", flush=True)
    try:
        summary = build_dataset(
            eval_dir,
            out_dir,
            group_by=args.group_by,
            val=args.val,
            test=args.test,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            max_seconds=args.max_seconds,
            min_seconds=args.min_seconds,
            max_gap=args.max_gap,
            pad=args.pad,
            sample_rate=settings.sample_rate,
            dry_run=args.dry_run,
            force=args.force,
        )
    except (FinetuneDataError, ValueError, AudioError) as error:
        print(f"Error: {error}")
        return 1
    print(summary_markdown(summary))
    if args.dry_run:
        print("Dry run: nothing was written.")
    else:
        print(f"Wrote {out_dir / 'dataset.json'} and train/val/test clips")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    `argv=None` means "read the real command-line arguments". Tests pass an
    explicit list instead, which lets them call main() like a normal function.
    The return value is the process exit code (0 means success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "config":
        return show_config()
    if args.command == "transcribe":
        return run_transcribe(args)
    if args.command == "benchmark":
        return run_benchmark(args)
    if args.command == "evalset":
        return run_evalset(args)
    if args.command == "normalize":
        return run_normalize(args)
    if args.command == "wer":
        return run_wer(args)
    if args.command == "terms":
        return run_terms(args)
    if args.command == "baseline":
        return run_baseline(args)
    if args.command == "llm":
        return run_llm(args)
    if args.command == "notes":
        return run_notes(args)
    if args.command == "concepts":
        return run_concepts(args)
    if args.command == "flashcards":
        return run_flashcards(args)
    if args.command == "quiz":
        return run_quiz(args)
    if args.command == "faithfulness":
        return run_faithfulness(args)
    if args.command == "agent":
        return run_agent(args)
    if args.command == "tasks":
        return run_tasks(args)
    if args.command == "todo":
        return run_todo(args)
    if args.command == "ingest":
        return run_ingest(args)
    if args.command == "ask":
        return run_ask(args)
    if args.command == "eval-retrieval":
        return run_eval_retrieval(args)
    if args.command == "finetune-data":
        return run_finetune_data(args)
    # No subcommand given: show usage so the user sees what is available.
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
