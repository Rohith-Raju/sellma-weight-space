"""Download past Högskoleprovet verbal booklets and their answer keys.

Where the files come from
-------------------------
The listing page links one page per past exam, and each exam page links its PDFs.
The newest exam is not on the listing page yet (at the time of writing, 2026-10-18);
its PDFs are on the "latest results" page, so that page is read as well.

Which files are kept
--------------------
- Only PDFs inside an exam folder named hp-YYYY-MM-DD. Norm tables, source lists and
  the ELF examples live in other folders, so they are skipped.
- Verbal booklets (the file name contains "verb") and answer keys (the name contains
  "facit", or is "hp-25b.pdf"-style in 2025). Maths booklets are never downloaded.
- File names are not consistent between years. The exam date comes from the folder
  name, which is consistent. The pass number is a best guess from the file name; the
  extract step checks it against the booklet itself.

Downloads are untrusted. They are saved as bytes and only ever opened by PyMuPDF.
A response that does not start with the PDF signature is refused, so an HTML error
page is never stored under a .pdf name.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlparse

import requests

from bench.paths import MANIFEST, RAW, ROOT

USER_AGENT = "selma-bench/0.1"
SITE = "https://www.studera.nu"
LISTING_URL = f"{SITE}/hogskoleprov/om/forbereda/tidigare/"
LATEST_RESULTS_URL = f"{SITE}/hogskoleprov/resultat/resultat/senaste/"
DELAY_SECONDS = 1.0            # pause between requests, to be polite to the site
TIMEOUT_SECONDS = 60
PDF_SIGNATURE = b"%PDF-"

HREF_PATTERN = re.compile(r'href="([^"#?]+)"')
EXAM_FOLDER_PATTERN = re.compile(r"/hp-(\d{4}-\d{2}-\d{2})/")
PASS_IN_NAME_PATTERN = re.compile(r"provpass[-_]?(\d)|del[-_](\d)")
ANSWER_KEY_NAME_PATTERN = re.compile(r"hp-\d{2}[ab]\.pdf")

KIND_BOOKLET = "booklet"
KIND_ANSWER_KEY = "answer_key"


@dataclass(frozen=True)
class RemoteFile:
    url: str
    exam_date: str
    kind: str                  # KIND_BOOKLET or KIND_ANSWER_KEY
    exam_pass: int | None      # from the file name when it says so, else None


@dataclass
class ManifestEntry:
    url: str
    path: str                  # relative to the project folder
    sha256: str
    size: int
    fetched_at: str
    kind: str
    exam_date: str
    exam_pass: int | None


def fetch_all(only: list[str] | None = None) -> None:
    """Discover every past exam, download its verbal booklets and answer keys, update the manifest."""
    session = _make_session()
    all_remote = discover_files(session)
    selected = [f for f in all_remote if only is None or f.exam_date in only]
    if not selected:
        print(f"No files found for exam dates {only}")
        return

    manifest = load_manifest()
    _forget_files_no_longer_listed(manifest, all_remote, only)
    failures: list[str] = []
    for remote in selected:
        try:
            _fetch_one(session, remote, manifest)
            _save_manifest(manifest)   # after each file, so an interrupted run can resume
        except (requests.RequestException, ValueError) as error:
            failures.append(f"{remote.url}: {error}")
            print(f"  FAILED {remote.url}: {error}")
        time.sleep(DELAY_SECONDS)

    print(f"Downloaded or verified {len(selected) - len(failures)} of {len(selected)} files.")
    if failures:
        print(f"{len(failures)} failed; run `bench fetch` again to retry them.")


def discover_files(session: requests.Session) -> list[RemoteFile]:
    """Return the booklets and answer keys listed on the listing page and the latest-results page."""
    page_urls = discover_exam_pages(session)
    seen_urls: set[str] = set()
    seen_names: dict[tuple[str, str], str] = {}
    found: list[RemoteFile] = []
    for page_url in page_urls + [LATEST_RESULTS_URL]:
        for pdf_url in _pdf_links_on_page(session, page_url):
            if pdf_url in seen_urls:
                continue
            seen_urls.add(pdf_url)
            remote = classify_pdf(pdf_url)
            if remote is None:
                continue
            name_key = (remote.exam_date, PurePosixPath(urlparse(pdf_url).path).name.lower())
            if name_key in seen_names:
                # Two different files with the same name in one exam folder cannot both be stored.
                print(f"  WARNING: {pdf_url} has the same name as {seen_names[name_key]}; skipped")
                continue
            seen_names[name_key] = pdf_url
            found.append(remote)
    return sorted(found, key=lambda f: (f.exam_date, f.kind, f.url))


def discover_exam_pages(session: requests.Session) -> list[str]:
    """Links from the listing page that lead to an exam page (their last path part names the exam)."""
    listing = _get_text(session, LISTING_URL)
    pages: list[str] = []
    for href in HREF_PATTERN.findall(listing):
        url = urljoin(LISTING_URL, html.unescape(href))
        path = urlparse(url).path
        if not path.startswith("/hogskoleprov/"):
            continue
        last_part = PurePosixPath(path.rstrip("/")).name
        if ("provfragor" in last_part or "facit" in last_part) and url not in pages:
            pages.append(url)
    return pages


def classify_pdf(pdf_url: str) -> RemoteFile | None:
    """Say whether a PDF link is a verbal booklet or an answer key of an exam. None if it is neither."""
    path = urlparse(pdf_url).path
    folder = EXAM_FOLDER_PATTERN.search(path)
    if folder is None:
        return None
    name = PurePosixPath(path).name.lower()
    if "exempel" in name:
        # Example exams (2015 has "facit-exempelprov.pdf") are not the real answer key.
        return None
    exam_date = folder.group(1)
    if "facit" in name or ANSWER_KEY_NAME_PATTERN.fullmatch(name):
        return RemoteFile(pdf_url, exam_date, KIND_ANSWER_KEY, exam_pass=None)
    if "verb" in name and "kvant" not in name:
        return RemoteFile(pdf_url, exam_date, KIND_BOOKLET, exam_pass=_pass_from_name(name))
    return None


def _pass_from_name(name: str) -> int | None:
    match = PASS_IN_NAME_PATTERN.search(name)
    if match is None:
        return None
    return int(match.group(1) or match.group(2))


def _pdf_links_on_page(session: requests.Session, page_url: str) -> list[str]:
    try:
        page = _get_text(session, page_url)
    except requests.RequestException as error:
        print(f"  WARNING: could not read {page_url}: {error}")
        return []
    links = []
    for href in HREF_PATTERN.findall(page):
        url = urljoin(page_url, html.unescape(href))
        if urlparse(url).path.lower().endswith(".pdf"):
            links.append(url)
    return links


def _forget_files_no_longer_listed(manifest: dict[str, ManifestEntry], all_remote: list[RemoteFile],
                                   only: list[str] | None) -> None:
    """Drop manifest entries for files the site no longer lists, and delete their local copies."""
    listed = {f.url for f in all_remote}
    for url, entry in list(manifest.items()):
        in_scope = only is None or entry.exam_date in only
        if in_scope and url not in listed:
            print(f"  removing {entry.path} (no longer listed on the site)")
            (ROOT / entry.path).unlink(missing_ok=True)
            del manifest[url]
            _save_manifest(manifest)


def _fetch_one(session: requests.Session, remote: RemoteFile, manifest: dict[str, ManifestEntry]) -> None:
    """Download one file, unless the local copy already matches the manifest."""
    local_path = RAW / remote.exam_date / PurePosixPath(urlparse(remote.url).path).name
    existing = manifest.get(remote.url)
    if existing and local_path.exists() and _sha256_of(local_path) == existing.sha256:
        print(f"  have   {remote.exam_date} {remote.kind:10} {local_path.name}")
        return

    print(f"  get    {remote.exam_date} {remote.kind:10} {local_path.name}")
    content = _download_pdf(session, remote.url)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(content)
    manifest[remote.url] = ManifestEntry(
        url=remote.url,
        path=str(local_path.relative_to(ROOT)),
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        kind=remote.kind,
        exam_date=remote.exam_date,
        exam_pass=remote.exam_pass,
    )


def _download_pdf(session: requests.Session, url: str) -> bytes:
    response = session.get(url, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    if not response.content.startswith(PDF_SIGNATURE):
        raise ValueError("the response is not a PDF")
    return response.content


def _get_text(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    time.sleep(DELAY_SECONDS)
    return response.text


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict[str, ManifestEntry]:
    """The manifest as a dict keyed by the source URL. Empty if nothing was fetched yet."""
    if not MANIFEST.exists():
        return {}
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    return {entry["url"]: ManifestEntry(**entry) for entry in entries}


def _save_manifest(manifest: dict[str, ManifestEntry]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted((asdict(entry) for entry in manifest.values()),
                     key=lambda e: (e["exam_date"], e["path"]))
    MANIFEST.write_text(json.dumps({"files": entries}, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
