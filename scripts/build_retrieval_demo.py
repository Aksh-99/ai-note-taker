"""Build the demo corpus and labeled questions for the retrieval eval (Step 5.4).

Why a hand-written demo corpus?
    The real evaluation lectures (MIT OCW, eval/ocw_lectures.json) are
    CC BY-NC-SA and stay out of git, and the nightly build cannot download
    them. A retrieval eval still needs *something* committed so the metrics
    code is tested end to end and anyone can reproduce a number with one
    command. So this script holds four short, original lecture transcripts
    on the same four subjects as the OCW set, plus 26 labeled questions.

Why generate the JSON instead of typing it?
    Labels are written here as segment numbers ("the answer is in sentences
    12 and 13"), which is easy to check by eye. The script turns them into
    start/end times in seconds, which is what the eval uses, because a time
    span stays correct however the passages are cut (see retrieval_eval.py).
    Editing a sentence here and re-running keeps every timestamp and label
    consistent; a test checks the committed files match this script.

Usage (from the repo root):
    python scripts/build_retrieval_demo.py
    notetaker eval-retrieval --embedder hashing-512
"""

# The lecture sentences below are long lines on purpose (one per segment).
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "eval" / "retrieval"

# Speaking rate used to invent plausible timestamps: about 150 words per
# minute, i.e. 2.5 words per second, plus a short pause between sentences.
WORDS_PER_SECOND = 2.5
PAUSE_SECONDS = 0.6

LECTURES: dict[str, dict] = {
    "la-eigen": {
        "title": "Linear Algebra, Lecture 7: Eigenvalues and Eigenvectors",
        "sentences": [
            "Okay, welcome back. Today is all about eigenvalues and eigenvectors, which are probably the most useful idea in this course.",
            "Here is the question we want to answer. When does a matrix A send a vector x to a multiple of itself?",
            "Most vectors change direction when you multiply them by A. The special ones that keep their direction are called eigenvectors.",
            "So the equation is A x equals lambda x, where lambda is just a number, and that number is the eigenvalue.",
            "If lambda is two, the vector gets stretched to twice its length. If lambda is negative, it flips around.",
            "How do we actually find them? Move everything to one side and you get A minus lambda I, times x, equals zero.",
            "We want a nonzero x in the null space of A minus lambda I, which means that matrix has to be singular.",
            "And a matrix is singular exactly when its determinant is zero. So we solve determinant of A minus lambda I equals zero.",
            "That equation is a polynomial in lambda, called the characteristic polynomial. For an n by n matrix it has degree n.",
            "Let's do a two by two example. Take the matrix with rows four one and two three.",
            "The determinant of A minus lambda I is lambda squared minus seven lambda plus ten, which factors nicely.",
            "So the eigenvalues are five and two. Plug each one back in to find the eigenvectors by elimination.",
            "Here is a quick check you should always do. The sum of the eigenvalues equals the trace, the sum of the diagonal entries.",
            "Four plus three is seven, and five plus two is seven. Good. And the product of the eigenvalues equals the determinant.",
            "Now, why do we care? Because eigenvectors make powers of a matrix easy to compute.",
            "If x is an eigenvector, then A to the hundredth times x is just lambda to the hundredth times x. No multiplying matrices a hundred times.",
            "If we have n independent eigenvectors, we can put them in the columns of a matrix S and diagonalize A.",
            "That gives A equals S times Lambda times S inverse, where Lambda is the diagonal matrix of eigenvalues.",
            "Not every matrix can be diagonalized. A rotation of the plane by ninety degrees has no real eigenvectors at all.",
            "Its eigenvalues turn out to be the complex numbers i and minus i, which makes sense since every vector gets turned.",
            "Symmetric matrices are the best case. Their eigenvalues are always real and their eigenvectors can be chosen perpendicular.",
            "This is the spectral theorem, and it is the reason principal component analysis works, which we will see later.",
            "Markov matrices are another nice family. They always have an eigenvalue equal to one, which gives the steady state.",
            "For next time, read section six point two and try the problems on diagonalization.",
        ],
    },
    "algo-peak": {
        "title": "Algorithms, Lecture 1: Peak Finding and Complexity",
        "sentences": [
            "Good morning everyone. This class is about efficient procedures for solving problems, and how we measure efficiency.",
            "We will start with a toy problem called peak finding, because it shows the whole way of thinking in one example.",
            "Picture a one dimensional array of numbers. A position is a peak if its value is at least as big as its neighbors.",
            "Does a peak always exist? Yes, with this definition using greater than or equal, every array has at least one.",
            "The obvious algorithm is to walk from left to right and stop at the first position that beats both neighbors.",
            "In the worst case the peak is at the far right end, so we look at all n elements. That is linear time, order n.",
            "Can we do better? Let's look at the middle element and compare it with the element just to its left.",
            "If the left neighbor is bigger, then there must be a peak somewhere in the left half, so we throw away the right half.",
            "Otherwise we check the right neighbor in the same way. If neither neighbor is bigger, the middle itself is a peak.",
            "Each step cuts the problem in half. That is the same divide and conquer idea as binary search.",
            "Write the running time as a recurrence. T of n equals T of n over two plus a constant amount of work.",
            "Unroll it and you get a constant times log base two of n. So this version is logarithmic time.",
            "To feel the difference, imagine ten million elements. The linear scan might look at all of them.",
            "The halving method needs about twenty three comparisons. That gap only grows as the input gets bigger.",
            "We describe this with asymptotic notation. Big O gives an upper bound and ignores constant factors.",
            "Big Theta means the bound is tight from above and below. Big Omega is the lower bound.",
            "Now the two dimensional version. You have a grid of numbers and a peak must beat its four neighbors.",
            "A greedy ascent, always stepping to a bigger neighbor, works but can wander through most of the grid.",
            "A better idea is to pick the middle column, find its maximum, and compare with the left and right neighbors.",
            "Then recurse on the half that contains the bigger neighbor. That gives n log n time for an n by n grid.",
            "Be careful though. A tempting variant that picks a peak inside the column instead of the maximum is actually wrong.",
            "Correctness matters as much as speed. A fast algorithm that returns the wrong answer is useless.",
            "Problem set one is posted tonight, and it asks you to prove the two dimensional algorithm correct.",
            "Office hours start this week. See the course website for times and rooms.",
        ],
    },
    "prob-axioms": {
        "title": "Probability, Lecture 1: Probability Models and Axioms",
        "sentences": [
            "Welcome to probability. Our goal is to build a precise language for talking about uncertainty.",
            "Every probability model starts with a sample space, the set of all possible outcomes of an experiment.",
            "The outcomes have to be mutually exclusive and collectively exhaustive. Exactly one of them happens.",
            "If you roll a four sided die twice, the sample space is the grid of sixteen ordered pairs.",
            "An event is any subset of the sample space, like the event that the first roll is a three.",
            "Probabilities obey three axioms. First, nonnegativity. The probability of any event is at least zero.",
            "Second, normalization. The probability of the entire sample space is exactly one.",
            "Third, additivity. If two events are disjoint, the probability of their union is the sum of their probabilities.",
            "Everything else follows from these. For instance the probability of an event plus its complement equals one.",
            "When all outcomes are equally likely, the probability of an event is the number of its outcomes divided by the total.",
            "So the chance that both rolls of the four sided die are equal is four out of sixteen, which is one quarter.",
            "Continuous sample spaces need more care. Throw a dart at a unit square and every single point has probability zero.",
            "There we measure probability as area. The chance of landing in a region is just the area of that region.",
            "Now conditional probability. Suppose we learn that some event B happened. How should our beliefs change?",
            "The probability of A given B is the probability of A and B together, divided by the probability of B.",
            "Think of it as shrinking the universe to B and renormalizing so that everything inside B adds up to one.",
            "Multiplying it out gives the multiplication rule. P of A and B equals P of B times P of A given B.",
            "The total probability theorem splits the sample space into pieces and adds up the contributions from each piece.",
            "Combine these and you get Bayes rule, which lets us reverse the direction of conditioning.",
            "Classic example. A test for a rare disease is ninety nine percent accurate, but only one person in a thousand has it.",
            "If you test positive, Bayes rule says the chance you actually have the disease is only about nine percent.",
            "The reason is that false positives from the huge healthy population outnumber the true positives.",
            "This kind of reasoning, updating beliefs with evidence, is the core of inference and of much of machine learning.",
            "Next lecture we cover independence. Recitation problems are on the website.",
        ],
    },
    "ml-gradient": {
        "title": "Machine Learning, Lecture 3: Linear Regression and Gradient Descent",
        "sentences": [
            "Alright, today we fit our first real model. Linear regression, trained with gradient descent.",
            "We have training examples, each a feature vector x and a target value y, and we want to predict y from x.",
            "The model is a weighted sum. The prediction is w transpose x plus a bias term b.",
            "To say how good a choice of weights is, we define a loss. We use the mean squared error over the training set.",
            "Squaring the errors punishes big mistakes heavily and makes the loss a smooth bowl shaped function of w.",
            "For linear regression there is a closed form solution, the normal equations, with X transpose X inverse.",
            "But inverting that matrix costs cubic time in the number of features, and it does not extend to neural networks.",
            "So instead we use gradient descent. Start with some weights and repeatedly take a small step downhill.",
            "The gradient points in the direction where the loss increases fastest, so we move in the opposite direction.",
            "The update rule is w becomes w minus eta times the gradient, where eta is the learning rate.",
            "Choosing the learning rate is an art. Too small and training crawls along for thousands of steps.",
            "Too large and the loss bounces around or even blows up to infinity instead of settling down.",
            "A good habit is to plot the loss after every epoch. It should go down steadily and then flatten out.",
            "Computing the gradient over the full dataset is expensive when you have millions of examples.",
            "Stochastic gradient descent estimates it from a small random mini batch, which is noisy but much cheaper.",
            "Features on very different scales make the bowl long and thin, and gradient descent zigzags across it.",
            "The fix is to standardize each feature to mean zero and variance one before training.",
            "Now the big danger. A model can fit the training data perfectly and still do badly on new data.",
            "That is overfitting. The model memorized noise in the training set instead of learning the real pattern.",
            "To detect it, hold out a validation set that the model never trains on, and watch its error.",
            "When training error keeps falling but validation error starts rising, you are overfitting.",
            "Regularization helps. Adding a penalty on the size of the weights, called ridge or L2, keeps the model simple.",
            "There is a nice probabilistic view too. Minimizing squared error is maximum likelihood under Gaussian noise.",
            "Homework two asks you to implement gradient descent from scratch in NumPy and compare it with the closed form.",
        ],
    },
}

# Each question: (id, text, tags, [(lecture_id, first_segment, last_segment)]).
# Tags describe the kind of question, so the report can break scores down:
#   lexical     shares its key words with the answer (easy for word matching)
#   paraphrase  asks the same thing in different words (needs meaning)
#   multi       the answer is spread over several places
QUESTIONS: list[tuple[str, str, list[str], list[tuple[str, int, int]]]] = [
    ("q01", "What is the characteristic polynomial?", ["lexical"],
     [("la-eigen", 7, 8)]),
    ("q02", "How can I sanity check the numbers I computed for a two by two matrix?",
     ["paraphrase"], [("la-eigen", 12, 13)]),
    ("q03", "Why is raising a matrix to a high power cheap once you know its special directions?",
     ["paraphrase"], [("la-eigen", 14, 15)]),
    ("q04", "Which matrices have real eigenvalues and perpendicular eigenvectors?",
     ["lexical"], [("la-eigen", 20, 21)]),
    ("q05", "Can a transformation fail to have any direction it leaves unchanged?",
     ["paraphrase"], [("la-eigen", 18, 19)]),
    ("q06", "How do you diagonalize a matrix using its eigenvectors?", ["lexical"],
     [("la-eigen", 16, 17)]),
    ("q07", "What is a peak in a one dimensional array?", ["lexical"],
     [("algo-peak", 2, 3)]),
    ("q08", "How can we avoid scanning every number when searching for a local maximum in a list?",
     ["paraphrase"], [("algo-peak", 6, 9)]),
    ("q09", "What does big O notation mean?", ["lexical"], [("algo-peak", 14, 15)]),
    ("q10", "How large is the speedup in practice for a really long list?",
     ["paraphrase"], [("algo-peak", 12, 13)]),
    ("q11", "What is the running time of the two dimensional peak finding algorithm?",
     ["lexical"], [("algo-peak", 18, 19)]),
    ("q12", "Which shortcut for the grid version gives incorrect results?",
     ["paraphrase"], [("algo-peak", 20, 20)]),
    ("q13", "What are the three axioms of probability?", ["lexical"],
     [("prob-axioms", 5, 7)]),
    ("q14", "How do I revise what I think after finding out that something occurred?",
     ["paraphrase"], [("prob-axioms", 13, 15)]),
    ("q15", "What is a sample space?", ["lexical"], [("prob-axioms", 1, 2)]),
    ("q16", "Why is a positive result on a medical screening often misleading?",
     ["paraphrase"], [("prob-axioms", 19, 21)]),
    ("q17", "How do we assign chances when an outcome can be any point in a region?",
     ["paraphrase"], [("prob-axioms", 11, 12)]),
    ("q18", "How do you compute probability when all outcomes are equally likely?",
     ["lexical"], [("prob-axioms", 9, 10)]),
    ("q19", "What is the update rule for gradient descent?", ["lexical"],
     [("ml-gradient", 9, 9)]),
    ("q20", "What goes wrong if the step size is set too big?", ["paraphrase"],
     [("ml-gradient", 10, 11)]),
    ("q21", "What is overfitting and how do you detect it?", ["lexical"],
     [("ml-gradient", 17, 20)]),
    ("q22", "Why should inputs be rescaled before optimizing?", ["paraphrase"],
     [("ml-gradient", 15, 16)]),
    ("q23", "How can we train cheaply when the dataset has a huge number of rows?",
     ["paraphrase"], [("ml-gradient", 13, 14)]),
    ("q24", "What is the closed form solution for linear regression?", ["lexical"],
     [("ml-gradient", 5, 6)]),
    ("q25", "How does reasoning about likelihood and evidence connect to fitting models?",
     ["multi", "paraphrase"], [("prob-axioms", 22, 22), ("ml-gradient", 22, 22)]),
    ("q26", "What reading, homework or problems were assigned for next time?",
     ["multi", "lexical"],
     [("la-eigen", 23, 23), ("algo-peak", 22, 22), ("prob-axioms", 23, 23),
      ("ml-gradient", 23, 23)]),
]  # fmt: skip


def make_segments(sentences: list[str]) -> list[dict]:
    """Turn sentences into Whisper-style segments with invented timestamps."""
    segments = []
    t = 0.0
    for i, text in enumerate(sentences):
        duration = len(text.split()) / WORDS_PER_SECOND
        start, end = round(t, 2), round(t + duration, 2)
        segments.append(
            {
                "id": i,
                "start": start,
                "end": end,
                "text": text,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.01,
                "words": [],
            }
        )
        t = end + PAUSE_SECONDS
    return segments


def build() -> dict[Path, str]:
    """Return {path: file contents} for every generated file (nothing written)."""
    files: dict[Path, str] = {}
    segments_by_lecture = {}
    for lecture_id, lecture in LECTURES.items():
        segments = make_segments(lecture["sentences"])
        segments_by_lecture[lecture_id] = segments
        transcript = {
            "segments": segments,
            "language": "en",
            "language_probability": 1.0,
            "duration": segments[-1]["end"],
            "model": "hand-written demo (scripts/build_retrieval_demo.py)",
        }
        path = OUT_DIR / "lectures" / lecture_id / "transcript.json"
        files[path] = json.dumps(transcript, indent=2) + "\n"

    titles = {lid: lec["title"] for lid, lec in LECTURES.items()}
    files[OUT_DIR / "titles.json"] = json.dumps(titles, indent=2) + "\n"

    questions = []
    for qid, text, tags, spans in QUESTIONS:
        relevant = []
        for lecture_id, first, last in spans:
            segs = segments_by_lecture[lecture_id]
            relevant.append(
                {
                    "lecture_id": lecture_id,
                    "start": segs[first]["start"],
                    "end": segs[last]["end"],
                }
            )
        questions.append(
            {"id": qid, "question": text, "tags": tags, "relevant": relevant}
        )
    question_file = {
        "description": (
            "Demo retrieval eval set: original hand-written lectures on the four "
            "OCW eval subjects, with each answer labeled as a time span in a "
            "lecture. Generated by scripts/build_retrieval_demo.py; edit that "
            "script, not this file."
        ),
        "corpus": "lectures",
        "titles": "titles.json",
        "questions": questions,
    }
    files[OUT_DIR / "questions.json"] = json.dumps(question_file, indent=2) + "\n"
    return files


def main() -> None:
    for path, content in build().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
