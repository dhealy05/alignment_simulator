from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


class Embedder:
    def __init__(self, client: OpenAI, model: str = "text-embedding-3-small") -> None:
        self.client = client
        self.model = model
        self._cache: Dict[str, np.ndarray] = {}
        logger.debug("Initialized Embedder with model: %s", model)

    def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            logger.debug("Cache hit for embedding (len=%d)", len(text))
            return self._cache[text]
        logger.debug("Embedding text (len=%d): %s", len(text), text[:80] + "..." if len(text) > 80 else text)
        resp = self.client.embeddings.create(model=self.model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=float)
        self._cache[text] = vec
        logger.debug("Embedded to vector of dim %d", len(vec))
        return vec


class AsyncEmbedder:
    def __init__(self, client: AsyncOpenAI, model: str = "text-embedding-3-small") -> None:
        self.client = client
        self.model = model
        self._cache: Dict[str, np.ndarray] = {}
        self._lock = asyncio.Lock()
        logger.debug("Initialized AsyncEmbedder with model: %s", model)

    async def embed(self, text: str) -> np.ndarray:
        # Check cache first (with lock to avoid race conditions)
        async with self._lock:
            if text in self._cache:
                logger.debug("Cache hit for embedding (len=%d)", len(text))
                return self._cache[text]

        logger.debug("Embedding text (len=%d): %s", len(text), text[:80] + "..." if len(text) > 80 else text)
        resp = await self.client.embeddings.create(model=self.model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=float)

        async with self._lock:
            self._cache[text] = vec

        logger.debug("Embedded to vector of dim %d", len(vec))
        return vec

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts, using cache where possible."""
        results: List[np.ndarray | None] = [None] * len(texts)
        to_fetch: List[tuple[int, str]] = []

        async with self._lock:
            for i, text in enumerate(texts):
                if text in self._cache:
                    results[i] = self._cache[text]
                else:
                    to_fetch.append((i, text))

        if to_fetch:
            texts_to_fetch = [t for _, t in to_fetch]
            logger.debug("Batch embedding %d texts", len(texts_to_fetch))
            resp = await self.client.embeddings.create(model=self.model, input=texts_to_fetch)

            async with self._lock:
                for (idx, text), emb_data in zip(to_fetch, resp.data):
                    vec = np.array(emb_data.embedding, dtype=float)
                    self._cache[text] = vec
                    results[idx] = vec

        return results  # type: ignore


def measure_embedding_direction(
    embedder: Embedder,
    response: str,
    cell: Dict[str, Any],
    measured_def: Dict[str, Any],
) -> float:
    params = measured_def["params"]
    anchor_pos = params["anchor_positive"].format(**cell)
    anchor_neg = params["anchor_negative"].format(**cell)

    v_response = embedder.embed(response)
    v_pos = embedder.embed(anchor_pos)
    v_neg = embedder.embed(anchor_neg)

    sim_pos = cosine_sim(v_response, v_pos)
    sim_neg = cosine_sim(v_response, v_neg)
    score = sim_pos - sim_neg
    logger.debug("Embedding direction: sim_pos=%.4f, sim_neg=%.4f, score=%.4f", sim_pos, sim_neg, score)
    return score


def measure_response(
    embedder: Embedder | None,
    response: str,
    cell: Dict[str, Any],
    measured_def: Dict[str, Any],
    tool_calls: List[Dict[str, Any]] | None = None,
) -> Any:
    method = measured_def["method"]
    logger.debug("Measuring response with method: %s", method)
    if method == "embedding_direction":
        if embedder is None:
            raise ValueError("Embedder is required for embedding_direction measurements.")
        return measure_embedding_direction(embedder, response, cell, measured_def)
    if method == "json_field":
        return measure_json_field(response, measured_def.get("params", {}))
    if method == "regex":
        return measure_regex(response, measured_def.get("params", {}))
    if method == "tool_call":
        return measure_tool_call(tool_calls or [], measured_def.get("params", {}))
    if method == "derived":
        raise ValueError("Derived measurements should be computed after base measurements.")
    raise ValueError(f"Unsupported measurement method: {method!r}")


async def measure_embedding_direction_async(
    embedder: AsyncEmbedder,
    response: str,
    cell: Dict[str, Any],
    measured_def: Dict[str, Any],
) -> float:
    params = measured_def["params"]
    anchor_pos = params["anchor_positive"].format(**cell)
    anchor_neg = params["anchor_negative"].format(**cell)

    # Batch embed all three texts at once for efficiency
    vectors = await embedder.embed_batch([response, anchor_pos, anchor_neg])
    v_response, v_pos, v_neg = vectors

    sim_pos = cosine_sim(v_response, v_pos)
    sim_neg = cosine_sim(v_response, v_neg)
    score = sim_pos - sim_neg
    logger.debug("Embedding direction: sim_pos=%.4f, sim_neg=%.4f, score=%.4f", sim_pos, sim_neg, score)
    return score


async def measure_response_async(
    embedder: AsyncEmbedder | None,
    response: str,
    cell: Dict[str, Any],
    measured_def: Dict[str, Any],
    tool_calls: List[Dict[str, Any]] | None = None,
) -> Any:
    method = measured_def["method"]
    logger.debug("Measuring response (async) with method: %s", method)
    if method == "embedding_direction":
        if embedder is None:
            raise ValueError("Embedder is required for embedding_direction measurements.")
        return await measure_embedding_direction_async(embedder, response, cell, measured_def)
    if method == "json_field":
        return measure_json_field(response, measured_def.get("params", {}))
    if method == "regex":
        return measure_regex(response, measured_def.get("params", {}))
    if method == "tool_call":
        return measure_tool_call(tool_calls or [], measured_def.get("params", {}))
    if method == "derived":
        raise ValueError("Derived measurements should be computed after base measurements.")
    raise ValueError(f"Unsupported measurement method: {method!r}")


def get_output_type(measured_def: Dict[str, Any]) -> str:
    method = measured_def.get("method")
    params = measured_def.get("params", {})
    if method == "embedding_direction":
        return "numeric"
    if method == "json_field":
        return params.get("output_type", "categorical")
    if method == "regex":
        return params.get("output_type", "boolean")
    if method == "tool_call":
        return params.get("output_type", "boolean")
    if method == "derived":
        return params.get("output_type", "boolean")
    return "numeric"


def aggregate_measurements(values: Iterable[Any], output_type: str) -> Any:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    if output_type in {"numeric", "count"}:
        numeric = [float(v) for v in cleaned]
        return float(np.mean(numeric)) if numeric else None
    if output_type == "boolean":
        numeric = [1.0 if bool(v) else 0.0 for v in cleaned]
        return float(np.mean(numeric)) if numeric else None
    if output_type in {"categorical", "extract"}:
        try:
            counts = Counter(cleaned)
        except TypeError:
            counts = Counter([json.dumps(v, sort_keys=True) if isinstance(v, dict) else str(v) for v in cleaned])
        return counts.most_common(1)[0][0] if counts else None
    return None


def measure_json_field(response: str, params: Dict[str, Any]) -> Any:
    path = params.get("path", "")
    if not path:
        return None
    on_error = params.get("on_parse_error", "null")
    parsed = _parse_json_response(response)
    if parsed is None:
        if on_error == "fail":
            raise ValueError("Failed to parse JSON response.")
        return None
    try:
        value = _extract_path(parsed, path)
    except (KeyError, IndexError, TypeError):
        if on_error == "fail":
            raise
        return None
    valid_values = params.get("valid_values")
    if valid_values is not None and value not in valid_values:
        return None
    output_type = params.get("output_type")
    return _coerce_output(value, output_type)


def measure_regex(response: str, params: Dict[str, Any]) -> Any:
    pattern = params.get("pattern", "")
    if not pattern:
        return None
    flags = _parse_regex_flags(params.get("flags"))
    regex = re.compile(pattern, flags=flags)
    matches = regex.findall(response)
    output_type = params.get("output_type", "boolean")
    if output_type == "count":
        return len(matches)
    if output_type == "extract":
        if not matches:
            return None
        first = matches[0]
        if isinstance(first, tuple):
            return first[0] if len(first) == 1 else list(first)
        return first
    return bool(matches)


def measure_tool_call(tool_calls: List[Dict[str, Any]], params: Dict[str, Any]) -> Any:
    tool_name = params.get("tool_name")
    if not tool_name:
        return None
    field = params.get("field")
    output_type = params.get("output_type", "boolean")

    matching = [c for c in tool_calls if c.get("name") == tool_name]

    if output_type == "count":
        return len(matching)

    if field:
        if not matching:
            return False if output_type == "boolean" else None
        args = _parse_tool_args(matching[0].get("arguments"))
        value = args.get(field) if isinstance(args, dict) else None
        if output_type == "boolean":
            return bool(value)
        return value

    return bool(matching) if output_type == "boolean" else len(matching)


def measure_derived(cell: Dict[str, Any], measurements: Dict[str, Any], params: Dict[str, Any]) -> Any:
    formula = params.get("formula")
    if not formula:
        return None
    try:
        value = _safe_eval_derived(formula, {**cell, **measurements})
    except Exception:
        return None
    return _coerce_output(value, params.get("output_type"))


def _parse_tool_args(arguments: Any) -> Dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return {}


def _parse_json_response(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt to extract the first JSON object/array from the response
    candidates = _extract_json_candidates(text)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_json_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    stack: List[Tuple[str, int]] = []
    for i, ch in enumerate(text):
        if ch in "{[":
            stack.append((ch, i))
        elif ch in "}]":
            if not stack:
                continue
            open_ch, start = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                stack.clear()
                continue
            if not stack:
                candidates.append(text[start : i + 1])
    return candidates


def _extract_path(data: Any, path: str) -> Any:
    path = path.strip()
    if path.startswith("$."):
        path = path[2:]
    if path.startswith("$"):
        path = path[1:]
    tokens = _tokenize_path(path)
    current = data
    for token in tokens:
        if isinstance(token, int):
            current = current[token]
        else:
            current = current[token]
    return current


def _tokenize_path(path: str) -> List[Any]:
    if not path:
        return []
    tokens: List[Any] = []
    i = 0
    buf = ""
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if buf:
                tokens.append(buf)
                buf = ""
            i += 1
            continue
        if ch == "[":
            if buf:
                tokens.append(buf)
                buf = ""
            end = path.find("]", i)
            if end == -1:
                raise ValueError(f"Unclosed bracket in path: {path!r}")
            inside = path[i + 1 : end].strip().strip("'").strip('"')
            if inside.isdigit():
                tokens.append(int(inside))
            else:
                tokens.append(inside)
            i = end + 1
            continue
        buf += ch
        i += 1
    if buf:
        tokens.append(buf)
    return tokens


def _parse_regex_flags(flags: Any) -> int:
    if flags is None:
        return 0
    if isinstance(flags, int):
        return flags
    flag_map = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}
    if isinstance(flags, str):
        result = 0
        for ch in flags:
            result |= flag_map.get(ch.lower(), 0)
        return result
    if isinstance(flags, list):
        result = 0
        for item in flags:
            if isinstance(item, str):
                result |= flag_map.get(item.lower(), 0)
        return result
    return 0


def _coerce_output(value: Any, output_type: str | None) -> Any:
    if output_type is None:
        return value
    if output_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
        return None
    if output_type == "numeric":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if output_type == "categorical":
        return value
    if output_type == "count":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if output_type == "extract":
        return value
    return value


def _safe_eval_derived(expr: str, context: Dict[str, Any]) -> Any:
    import ast

    node = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.USub,
        ast.UAdd,
    )

    for n in ast.walk(node):
        if not isinstance(n, allowed_nodes):
            raise ValueError(f"Unsupported expression in derived formula: {expr!r}")
        if isinstance(n, ast.Name) and n.id not in context:
            raise KeyError(f"Unknown name in derived formula: {n.id!r}")

    env = {"__builtins__": {}}
    env.update(context)
    return eval(compile(node, "<derived>", "eval"), env, {})
