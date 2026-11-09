"""The model that turns pictures into voxels.

By default it is whatever the family resolves for vision through Hoard Link (Faustus's registry: DeepSeek V4.1 Flash on
the Sparks, served by Prometheus). Settings ``llm_url`` / ``llm_model`` point it at one OpenAI-compatible server
instead. Pictures always go as images: the model sees the reference, not a description of it.

The answer is streamed: the job shows how much has arrived, a cancel stops it at once, and an answer that starts
repeating itself (a model stuck in a loop) is cut instead of running to the token limit.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

from .errors import SeuratError
from .hoard_link import reasoning as _reasoning

log = logging.getLogger("seurat.llm")


@dataclass
class Answer:
    text: str
    model: str
    finish: str = ""            # stop | length | loop | cancelled | ""
    reasoning_chars: int = 0
    seconds: float = 0.0


def _looping(text: str, window: int = 2000) -> bool:
    """The tail repeats one chunk many times (a model stuck in a loop: the same pixel row, the same sentence)."""
    tail = text[-window:]
    if len(tail) < window:
        return False
    for size in (12, 24, 40, 80, 160):
        chunk = tail[-size:]
        if len(set(chunk)) <= 2 and size < 40:
            continue                     # "......" runs are normal inside pixel rows
        if tail.count(chunk) >= max(10, window // (size * 3)):
            return True
    return False


class ModelClient:
    def __init__(self, settings: Any, link_factory=None):
        self.settings = settings
        self._link = None
        self._link_factory = link_factory
        self._lock = threading.Lock()
        self._status: tuple[float, dict[str, Any]] = (0.0, {})

    # ------------------------------------------------------------------ resolution
    def _get_link(self):
        with self._lock:
            if self._link is None:
                if self._link_factory is not None:
                    self._link = self._link_factory()
                else:
                    from .hoard_link import Link, LinkConfig
                    self._link = Link(LinkConfig.load(app="seurat"))
            return self._link

    def endpoint(self) -> tuple[str, str, str]:
        """``(chat completions URL, model, api)`` of the model that would answer now."""
        url, model = self.settings["llm_url"], self.settings["llm_model"]
        if url:
            if not url.rstrip("/").endswith("/chat/completions"):
                url = url.rstrip("/") + ("/chat/completions" if url.rstrip("/").endswith("/v1") else "/v1/chat/completions")
            return url, model, "openai"
        try:
            res = self._get_link().sync.resolve("vision")
        except Exception as exc:  # noqa: BLE001
            raise SeuratError("no_model", f"No hay modelo disponible: {type(exc).__name__}: {exc}"[:400]) from None
        if res.state != "resolved" or not res.url:
            raise SeuratError("no_model", f"No hay modelo disponible: {res.reason}",
                              "Check that the Sparks are serving (Prometheus) and that Faustus is reachable, or set llm_url in Settings.")
        return res.url, res.model or "", str(res.api or "openai")

    def status(self, refresh: bool = False) -> dict[str, Any]:
        """Which model would answer now (cached 60 s)."""
        ts, cached = self._status
        if cached and not refresh and time.time() - ts < 60:
            return cached
        try:
            url, model, api = self.endpoint()
            out = {"available": True, "source": "settings" if self.settings["llm_url"] else "hoard_link", "url": url,
                   "model": model or "(server default)", "api": api}
        except SeuratError as exc:
            out = {"available": False, "source": "settings" if self.settings["llm_url"] else "hoard_link", "reason": exc.message}
        self._status = (time.time(), out)
        return out

    # ------------------------------------------------------------------ the call
    def chat(self, system: str, user: str, images: list[bytes], *, max_tokens: int, effort: Optional[str], timeout: float,
             on_progress: Optional[Callable[[int, int], None]] = None, cancel: Optional[threading.Event] = None,
             thinking_tokens: int = 0, json_mode: bool = True) -> Answer:
        """One streamed answer. ``max_tokens`` is the room for the answer itself; with reasoning on, ``thinking_tokens``
        (or the level's own budget) is added on top so thinking never eats the answer."""
        url, model, api = self.endpoint()
        level = _reasoning.normalize(effort)
        thinking = level not in (None, "off")
        if api != "openai":
            return self._via_link(system, user, images, max_tokens + (thinking_tokens if thinking else 0), effort, timeout)
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()}})
        body: dict[str, Any] = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                                "max_tokens": max_tokens, "temperature": 0.6 if thinking else 0.3, "stream": True}
        if model:
            body["model"] = model
        _reasoning.apply_openai(body, level)
        if thinking and thinking_tokens > 0:
            body["max_tokens"] = max_tokens + thinking_tokens
            for key in ("thinking_budget_tokens", "reasoning_budget"):
                if key in body:
                    body[key] = thinking_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}     # constrains the answer, not the thinking
        try:
            return self._stream(url, body, timeout, on_progress, cancel)
        except _RetryWithoutReasoning:
            _reasoning.strip(body)
            return self._stream(url, body, timeout, on_progress, cancel)
        except _RetryWithoutJson:
            body.pop("response_format", None)
            return self._stream(url, body, timeout, on_progress, cancel)

    def _via_link(self, system, user, images, max_tokens, effort, timeout) -> Answer:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        t0 = time.time()
        try:
            res = self._get_link().sync.chat(messages, images=images or None, max_tokens=max_tokens, capability="vision",
                                             effort=effort, timeout=timeout, temperature=0.4)
        except Exception as exc:  # noqa: BLE001
            raise SeuratError("no_model", f"The model did not answer: {type(exc).__name__}: {exc}"[:500]) from None
        return Answer(res.text or "", res.model or "", getattr(res, "finish_reason", "") or "", 0, time.time() - t0)

    def _stream(self, url: str, body: dict[str, Any], timeout: float, on_progress, cancel) -> Answer:
        t0 = time.time()
        text, reasoning, finish, model = [], [], "", body.get("model") or ""
        n_text = n_reason = 0
        last_report = 0.0
        try:
            with httpx.stream("POST", url, json=body, timeout=httpx.Timeout(timeout, connect=15.0)) as r:
                if r.status_code >= 400:
                    err = r.read().decode("utf-8", "replace")
                    if r.status_code == 400 and "response_format" in body and ("response_format" in err or "json" in err.lower()):
                        raise _RetryWithoutJson()
                    if r.status_code == 400 and _reasoning.looks_like_reasoning_error(r.status_code, err):
                        raise _RetryWithoutReasoning()
                    raise SeuratError("no_model", f"The model answered HTTP {r.status_code}: {err[:300]}")
                for line in r.iter_lines():
                    if cancel is not None and cancel.is_set():
                        finish = "cancelled"
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    model = chunk.get("model") or model
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            text.append(delta["content"])
                            n_text += len(delta["content"])
                        rc = delta.get("reasoning_content") or delta.get("reasoning")
                        if rc:
                            reasoning.append(rc)
                            n_reason += len(rc)
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                    now = time.time()
                    if on_progress and now - last_report > 1.0:
                        last_report = now
                        on_progress(n_text, n_reason)
                    if n_text > 2000 and len(text) % 25 == 0 and _looping("".join(text[-400:])):
                        finish = "loop"
                        break
                    if n_reason > 20000 and len(reasoning) % 100 == 0 and _looping("".join(reasoning[-800:]), 6000):
                        finish = "loop"          # thinking that goes round in circles: stop and use what there is
                        break
                    if time.time() - t0 > timeout:
                        finish = "timeout"
                        break
        except httpx.HTTPError as exc:
            raise SeuratError("no_model", f"{url} did not answer: {type(exc).__name__}: {exc}"[:400],
                              "Check that the Sparks are serving (Prometheus).") from None
        if on_progress:
            on_progress(n_text, n_reason)
        return Answer("".join(text), model, finish, n_reason, time.time() - t0)


class _RetryWithoutReasoning(Exception):
    pass


class _RetryWithoutJson(Exception):
    pass


FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _objects(text: str) -> list[dict[str, Any]]:
    """Every top-level JSON object that parses in ``text``, in order."""
    out = []
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = text.find("{", end)
    return out


def _salvage_ops(text: str) -> Optional[dict[str, Any]]:
    """A cut-off answer (token limit): keep every operation that arrived complete."""
    m = re.search(r'"ops"\s*:\s*\[', text)
    if not m:
        return None
    dec = json.JSONDecoder()
    i, ops = m.end(), []
    while True:
        while i < len(text) and text[i] in " \r\n\t,":
            i += 1
        if i >= len(text) or text[i] != "{":
            break
        try:
            obj, i = dec.raw_decode(text, i)
        except ValueError:
            break
        ops.append(obj)
    if not ops:
        return None
    notes = re.search(r'"notes"\s*:\s*"((?:[^"\\]|\\.)*)"', text[: m.start()])
    pal = re.search(r'"palette_add"\s*:\s*(\[[^\]]*\])', text[: m.start()])
    out: dict[str, Any] = {"ops": ops, "notes": json.loads('"' + notes.group(1) + '"') if notes else "", "salvaged": True}
    if pal:
        try:
            out["palette_add"] = json.loads(pal.group(1))
        except ValueError:
            pass
    return out


def parse_json(text: str) -> dict[str, Any]:
    """The answer object: the JSON object with ``ops`` (or ``done``) in a model answer, inside a ```json fence or bare,
    ignoring any thinking. A cut-off answer keeps its complete operations. Raises ValueError when there is nothing."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    candidates = [m.group(1) for m in FENCE_RE.finditer(text)] + [text]
    found: list[dict[str, Any]] = []
    for cand in candidates:
        found += _objects(cand)
    for obj in found:
        if "ops" in obj or "done" in obj:
            return obj
    salvaged = _salvage_ops(text)
    if salvaged:
        return salvaged
    if found:
        return found[0]
    raise ValueError("no JSON object in the answer")
