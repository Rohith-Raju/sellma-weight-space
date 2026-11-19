"""Where everything lives. All paths are relative to the project folder."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIG = ROOT / "config"
SPLIT_CONFIG = CONFIG / "split.toml"
MODELS_CONFIG = CONFIG / "models.toml"
DECISION_CONFIG = CONFIG / "decision.toml"

EXAMS = ROOT / "exams"
RAW = EXAMS / "raw"                          # downloaded PDFs, one folder per exam date
MANIFEST = RAW / "manifest.json"             # url, sha256 and download time of every file
QUESTIONS = EXAMS / "questions.jsonl"        # every extracted question
EXTRACT_REPORT = EXAMS / "extract_report.md"
SPLIT = EXAMS / "split.json"                 # frozen split, written once

DAMAGE_DATA = ROOT / "data" / "damage"       # Swedish text for perplexity and loop checks
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
