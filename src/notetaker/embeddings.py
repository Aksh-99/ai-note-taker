"""Turn text into vectors ("embeddings") so similar meanings sit close together.

Why embeddings?
    Search across lectures (Phase 5) has to find the passage that answers a
    question even when the question uses different words: "how do I get the
    eigenvalues?" should find "we solve det(A - lambda I) = 0". An embedding
    model maps any text to a point in R^d (d = 384 for the default model) so
    that texts with similar meaning land near each other. Search then becomes
    geometry: embed the question, return the passages whose vectors are
    closest to it.

How "closest" is measured:
    Cosine similarity, cos(u, v) = (u . v) / (|u| |v|), which is 1 for vectors
    pointing the same way and 0 for unrelated (orthogonal) ones. Every
    embedder here returns vectors already scaled to length 1 (L2-normalised),
    so cosine similarity is just the dot product u . v. That lets the vector
    store (vectorstore.py) use a plain inner-product index.

Two embedders, one interface:
    SentenceTransformerEmbedder
        The real thing: a small pretrained transformer from the free, open
        sentence-transformers library (default all-MiniLM-L6-v2, Apache 2.0,
        about 90 MB). It runs locally on the CPU; no API key. The library and
        its PyTorch dependency are large, so they are an optional extra:
            pip install -e ".[rag]"
        and the model weights download from the Hugging Face Hub on first use.

    HashingEmbedder
        A tiny bag-of-words embedder with no model at all (the "hashing
        trick", explained on the class). It only matches shared words, not
        meaning, but it is deterministic, instant and needs nothing
        downloaded. Tests and CI use it so they never fetch a model, and it
        doubles as a lexical baseline to compare the neural model against in
        the retrieval eval (Step 5.4).

    Both satisfy the `Embedder` protocol below, so the rest of the code never
    cares which one it was given.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol, runtime_checkable

import numpy as np

# The default neural model. Matches the default in config.py.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HASH_DIM = 512


class EmbeddingError(RuntimeError):
    """An embedder could not be created or produced bad output."""


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a list of texts into unit-length row vectors.

    A `Protocol` describes a shape (these attributes and methods) instead of a
    parent class, so any object with them fits. `@runtime_checkable` lets
    tests write `isinstance(x, Embedder)` to check the shape.

    Attributes:
        name: identifies the model. It is saved with an index so a later
            query cannot silently use a different model (vectors from two
            different models live in unrelated spaces and cannot be compared).
        dim: length d of every vector.
    """

    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed passages. Returns a float32 array of shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one search query. Returns a float32 array of shape (dim,)."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, leaving all-zero rows as zeros.

    An all-zero row (e.g. an empty text) has no direction, so dividing by its
    length 0 would give NaN. We divide by 1 instead, keeping it at zero, which
    gives it similarity 0 with everything.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ---------------------------------------------------------------------------
# HashingEmbedder: no model, used in tests and as a lexical baseline
# ---------------------------------------------------------------------------

# Words are runs of letters and digits; apostrophes inside a word are kept
# ("don't" stays one token).
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Very common words carry no topic information and would make every passage
# look similar. Kept short on purpose: a few dozen covers most of the mass.
# The fmt: off/on comments stop the formatter from putting one word per line.
# fmt: off
_STOPWORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "not", "of", "on", "or", "our", "she", "so", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "to", "up",
    "us", "was", "we", "were", "what", "when", "which", "who", "will", "with",
    "would", "you", "your", "okay", "ok", "um", "uh", "like", "just",
])
# fmt: on


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _stable_hash(token: str) -> int:
    """A hash of a string that is the same on every run and every machine.

    Python's built-in `hash()` is randomised per process (a security feature),
    so `hash("matrix")` changes every time Python starts. Vectors saved to
    disk today must match queries embedded tomorrow, so we use a fixed
    cryptographic hash (BLAKE2b) and read its first 8 bytes as an integer.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class HashingEmbedder:
    """Bag-of-words vectors via the "hashing trick" (feature hashing).

    A classic bag-of-words vector has one coordinate per vocabulary word,
    which needs a vocabulary built in advance. Feature hashing skips that:
    each word w is sent to coordinate h(w) mod d by a hash function, so any
    word, even one never seen before, has a slot. Two words occasionally
    share a slot (a collision); a second hash bit picks a sign of +1 or -1 so
    collisions cancel out on average instead of always adding up.

    Each word's weight is 1 + log(count), so a word said 10 times counts
    about 3.3 times as much as a word said once, not 10 times ("sublinear
    term frequency"). Adjacent word pairs (bigrams such as "linear_algebra")
    are added at half weight so a little word order survives.

    Finally the vector is L2-normalised, so the dot product of two vectors is
    their cosine similarity, which is high when they share distinctive words.
    """

    def __init__(self, dim: int = DEFAULT_HASH_DIM, bigrams: bool = True) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.bigrams = bigrams
        self.name = f"hashing-{dim}" + ("" if bigrams else "-unigram")

    def _features(self, text: str) -> dict[str, float]:
        """Map each feature (word or bigram) to its weight before hashing."""
        tokens = tokenize(text)
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        weights = {t: 1.0 + math.log(c) for t, c in counts.items()}
        if self.bigrams:
            pair_counts: dict[str, int] = {}
            for left, right in zip(tokens, tokens[1:], strict=False):
                pair = f"{left}_{right}"
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            for pair, c in pair_counts.items():
                weights[pair] = 0.5 * (1.0 + math.log(c))
        return weights

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature, weight in self._features(text).items():
            h = _stable_hash(feature)
            index = h % self.dim
            # Use a bit of the hash that the modulo above did not consume,
            # so the sign is independent of the slot.
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[index] += sign * weight
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder: the real neural model
# ---------------------------------------------------------------------------


def _import_sentence_transformers() -> Any:
    """Import the optional library, with a helpful message if it is missing.

    The import lives inside a function (a "lazy import") so that importing
    this module, and running the test suite, works without the big
    sentence-transformers + PyTorch install.
    """
    try:
        import sentence_transformers
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise EmbeddingError(
            "sentence-transformers is not installed. Install the search "
            'extra with: pip install -e ".[rag]"'
        ) from exc
    return sentence_transformers


class SentenceTransformerEmbedder:
    """A pretrained sentence-transformers model running locally.

    Args:
        model_name: any model on the Hugging Face Hub that sentence-transformers
            can load, e.g. "sentence-transformers/all-MiniLM-L6-v2" (default)
            or "BAAI/bge-small-en-v1.5".
        device: "cpu", "cuda", or None to let the library choose.
        batch_size: texts encoded per forward pass. Larger is faster but uses
            more memory.
        query_prefix / passage_prefix: some models were trained with a short
            instruction in front of the text (E5 models expect "query: " and
            "passage: "). MiniLM needs none, so both default to "".
        model: an already-built model object. Tests pass a fake here so no
            weights are downloaded; normal use leaves it as None.

    The model is loaded on first use, not in __init__, so creating the object
    is cheap and the download happens only when text is actually embedded.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        query_prefix: str = "",
        passage_prefix: str = "",
        model: Any = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model = model
        self._dim: int | None = None

    @property
    def model(self) -> Any:
        """The loaded model, created the first time it is needed."""
        if self._model is None:
            st = _import_sentence_transformers()
            self._model = st.SentenceTransformer(self.name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        """Vector length, read from the model (loads it if needed)."""
        if self._dim is None:
            # The method was renamed in newer library versions; try both.
            model = self.model
            getter = getattr(model, "get_embedding_dimension", None)
            if getter is None:
                getter = model.get_sentence_embedding_dimension
            dim = getter()
            if not dim:
                # A few models do not report it; embed one word and measure.
                dim = int(self._encode(["dimension probe"]).shape[1])
            self._dim = int(dim)
        return self._dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            # The library can normalise for us, but we re-normalise below
            # anyway so a model that ignores this flag is still handled.
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise EmbeddingError(
                f"model returned shape {vectors.shape} for {len(texts)} texts"
            )
        return l2_normalize(vectors)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


def make_embedder(name: str | None = None, **kwargs: Any) -> Embedder:
    """Build an embedder from a name, as saved in an index manifest.

    "hashing-<dim>" (optionally ending in "-unigram") gives a HashingEmbedder;
    anything else is treated as a sentence-transformers model name. None
    means the model from settings (NOTETAKER_EMBEDDING_MODEL).
    """
    if name is None:
        from notetaker.config import get_settings

        name = get_settings().embedding_model
    match = re.fullmatch(r"hashing-(\d+)(-unigram)?", name)
    if match:
        return HashingEmbedder(dim=int(match.group(1)), bigrams=not match.group(2))
    return SentenceTransformerEmbedder(model_name=name, **kwargs)
