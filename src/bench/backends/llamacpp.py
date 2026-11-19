"""Compressed GGUF model, served by llama.cpp's llama-server on the GPU.

The backend starts llama-server as a child process when it is created and stops it in
close(). Everything is asked over HTTP to 127.0.0.1.

Scoring: /completion with n_predict=1, temperature 0 and n_probs=20 returns the top 20
candidates for the first generated token with their log-probabilities. The option letter
probability is the sum of exp(logprob) over candidates whose stripped text is that letter.
A letter outside the top 20 counts as 0.0. This is the same quantity the HF backend sums,
but only over the top 20, so very small probabilities can differ.

Prompt tokens: the prompt is tokenised with the model's Hugging Face tokenizer (the
`tokenizer` folder in config/models.toml) and the token ids are sent to /completion. The
compressed model therefore reads exactly the token ids the full-precision model reads.
Sending the string would make the server add its own BOS, and Gemma's template already
writes one, so the model would see two. Without a tokenizer folder the server's own
/tokenize is used; on the two bench models it gives the same ids.

Port: each backend asks the operating system for a free port, so two runs on the same
machine (for example a Qwen chain and a Gemma chain) never fight over one.

Perplexity: llama-perplexity is run on a temporary text file and its "Final estimate"
line is parsed. It uses non-overlapping windows of PERPLEXITY_CTX tokens, similar to the
HF backend, but it is a different implementation, so results are only comparable with other
llama.cpp results. It loads a second copy of the model while the server is running.
"""

from __future__ import annotations

import math
import re
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import requests

from bench.backends import cached_sha256
from bench.backends.base import Backend
from bench.records import Question

DEFAULT_BIN_DIR = Path("/root/llama.cpp/build/bin")   # the GPU machine's build (tag b8680)
CTX_SIZE = 4096               # prompt plus answer must fit; the server gives the whole window to one slot
TOP_PROBS = 20                # n_probs: how many candidates for the first token we ask for
PERPLEXITY_CTX = 1024         # window length for llama-perplexity, same order as the HF backend
SEED = 0
STARTUP_TIMEOUT_S = 900       # large models take minutes to load
REQUEST_TIMEOUT_S = 600
PERPLEXITY_TIMEOUT_S = 6 * 3600

_FINAL_PPL = re.compile(r"Final estimate: PPL = ([0-9.]+)")


class LlamaCppBackend(Backend):

    def __init__(self, gguf_path: Path, llama_bin_dir: Path | None = None, port: int | None = None,
                 ctx_size: int = CTX_SIZE, seed: int = SEED, tokenizer_path: Path | None = None):
        self.gguf_path = Path(gguf_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self.tokenizer = _load_hf_tokenizer(self.tokenizer_path) if self.tokenizer_path else None
        port = port if port is not None else _free_port()
        self.bin_dir = Path(llama_bin_dir) if llama_bin_dir else DEFAULT_BIN_DIR
        self.server_bin = self.bin_dir / "llama-server"
        self.perplexity_bin = self.bin_dir / "llama-perplexity"
        self.ctx_size = ctx_size
        self.seed = seed
        self.base_url = f"http://127.0.0.1:{port}"
        self._port = port
        self._process: subprocess.Popen | None = None
        # The server's output goes to an anonymous file so that a failed start can show its log.
        self._log = tempfile.TemporaryFile()
        try:
            self._start_server()
        except Exception:
            self.close()
            raise

    def letter_probabilities(self, prompt: str, letters: list[str], question: Question) -> dict[str, float]:
        reply = self._post("/completion", {
            "prompt": self.prompt_token_ids(prompt),
            "n_predict": 1,
            "temperature": 0,
            "n_probs": TOP_PROBS,
            # A reused KV cache can change logits in the last bits, which would make repeated
            # runs differ. Every question is computed from scratch instead.
            "cache_prompt": False,
        })
        candidates = _first_token_candidates(reply)
        wanted = set(letters)
        probs = {letter: 0.0 for letter in letters}
        for candidate in candidates:
            text = candidate["token"].strip()
            if text in wanted:
                probs[text] += math.exp(candidate["logprob"])
        return probs

    def perplexity(self, texts: list[str]) -> float:
        with tempfile.TemporaryDirectory() as tmp_dir:
            text_file = Path(tmp_dir) / "texts.txt"
            text_file.write_text("\n\n".join(texts), encoding="utf-8")
            command = [
                str(self.perplexity_bin), "-m", str(self.gguf_path), "-f", str(text_file),
                "-c", str(PERPLEXITY_CTX), "-ngl", "all", "--seed", str(self.seed),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=PERPLEXITY_TIMEOUT_S)
        output = result.stdout + result.stderr
        match = _FINAL_PPL.search(output)
        if result.returncode != 0 or match is None:
            # llama-perplexity needs at least one full window of tokens; short texts fail here.
            raise RuntimeError(f"llama-perplexity gave no final estimate (exit {result.returncode}):\n"
                               f"{_tail(output)}")
        return float(match.group(1))

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        # Plain text, so the server's own BOS handling is what we want here.
        reply = self._post("/completion", {
            "prompt": prompt,
            "n_predict": max_new_tokens,
            "temperature": 0,
            "cache_prompt": False,
        })
        return reply["content"]

    def describe(self) -> dict:
        return {
            "backend": "llamacpp",
            "path": str(self.gguf_path),
            "sha256": cached_sha256(self.gguf_path),
            "prompt_tokenizer": str(self.tokenizer_path) if self.tokenizer_path else "llama-server /tokenize",
            "llama_server_bin": str(self.server_bin),
            "llama_server_version": _binary_version(self.server_bin),
            "n_gpu_layers": "all",
            "ctx_size": self.ctx_size,
            "seed": self.seed,
            "top_probs": TOP_PROBS,
            "perplexity_ctx": PERPLEXITY_CTX,
        }

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None
        self._log.close()

    def _start_server(self) -> None:
        if not self.server_bin.exists():
            raise FileNotFoundError(f"llama-server not found at {self.server_bin}")
        command = [
            str(self.server_bin),
            "-m", str(self.gguf_path),
            "--host", "127.0.0.1",
            "--port", str(self._port),
            "-c", str(self.ctx_size),
            "-ngl", "all",
            "--parallel", "1",
            "--seed", str(self.seed),
        ]
        self._process = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT)
        self._wait_until_healthy()

    def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self._process.returncode}:\n{self._log_tail()}")
            if self._is_healthy():
                return
            time.sleep(2)
        raise TimeoutError(f"llama-server was not healthy after {STARTUP_TIMEOUT_S} s:\n{self._log_tail()}")

    def _is_healthy(self) -> bool:
        # 503 means "still loading the model"; anything but 200 is not ready yet.
        try:
            return requests.get(f"{self.base_url}/health", timeout=2).status_code == 200
        except requests.RequestException:
            return False

    def prompt_token_ids(self, prompt: str) -> list[int]:
        """Token ids of a chat-formatted prompt, the same ids the HF backend feeds its model."""
        if self.tokenizer is None:
            return self.server_tokenize(prompt)
        # add_special_tokens=False: the chat template already writes BOS where the model needs it.
        return [int(token) for token in self.tokenizer(prompt, add_special_tokens=False).input_ids]

    def server_tokenize(self, text: str) -> list[int]:
        """The server's own tokenisation (special tokens parsed, none added). Used to cross-check."""
        reply = self._post("/tokenize", {"content": text, "add_special": False, "parse_special": True})
        return [int(token) for token in reply["tokens"]]

    def _post(self, endpoint: str, body: dict) -> dict:
        response = requests.post(f"{self.base_url}{endpoint}", json=body, timeout=REQUEST_TIMEOUT_S)
        if response.status_code != 200:
            raise RuntimeError(f"llama-server {endpoint} returned {response.status_code}: {response.text[:500]}")
        return response.json()

    def _log_tail(self) -> str:
        self._log.seek(0)
        return _tail(self._log.read().decode("utf-8", errors="replace"))


def _free_port() -> int:
    """A port nothing is listening on, chosen by the operating system."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _load_hf_tokenizer(tokenizer_path: Path):
    # Imported here so that importing this module does not need transformers.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_path))


def _first_token_candidates(reply: dict) -> list[dict]:
    """The top-token list for the one generated token. Each item has 'token' and 'logprob'."""
    probabilities = reply.get("completion_probabilities") or []
    if not probabilities:
        raise RuntimeError(f"llama-server returned no token probabilities: {str(reply)[:500]}")
    return probabilities[0]["top_logprobs"]


def _binary_version(binary: Path) -> str:
    """First line of `llama-server --version` that names a version, for run.json."""
    try:
        result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    for line in lines:
        if "version" in line.lower():
            return line
    return lines[0] if lines else "unknown"


def _tail(text: str, lines: int = 30) -> str:
    return "\n".join(text.splitlines()[-lines:])
