"""SimReady asset catalog backed by Hugging Face.

Index sources, merged:
  1. The official ``nvidia/simready-catalog`` index (needs ``hf auth login``; its schema is parsed
     defensively since it is an index of partner repos rather than asset content).
  2. A crawl of partner repos that follow the SimReady package layout:
       <package>/com.nvidia.simready.packaging.json
       <package>/simready_usd/<root>.usd

The merged index is cached as JSON; packages are downloaded on demand with ``snapshot_download``.
Searches go to SimReady Central (NVIDIA USD Search on a HF Space) first and fall back to the index.

CLI:
    python -m vrsim.catalog refresh
    python -m vrsim.catalog search "coffee mug"
    python -m vrsim.catalog fetch "balance board"
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import threading
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
from rapidfuzz import fuzz, process

from .config import Config

log = logging.getLogger(__name__)


def _hub_client_with_timeouts() -> None:
    """huggingface_hub (2.x) builds its HTTP client with ``timeout=None`` and tries huggingface.co's
    IPv6 addresses first. On a network without working IPv6 each of the eight addresses has to time
    out before IPv4 is tried; without a timeout one stalled connect hung both sidecar spawn workers
    for good, and every later catalog pick queued behind them. This client connects over IPv4 and
    fails a dead connection after 10 s (a read may pause 60 s; downloads keep streaming)."""
    try:
        import httpx2
        import huggingface_hub
        from huggingface_hub.utils import _http
    except ImportError:
        return  # older hub: requests-based, with its own timeouts
    factory = getattr(huggingface_hub, "set_client_factory", None)
    hook = getattr(_http, "hf_request_event_hook", None)
    if factory is None or hook is None:
        return

    def client():
        return httpx2.Client(event_hooks={"request": [hook]}, follow_redirects=True,
                             timeout=httpx2.Timeout(60.0, connect=10.0),
                             transport=httpx2.HTTPTransport(local_address="0.0.0.0"))  # IPv4

    factory(client)


_hub_client_with_timeouts()

PACKAGE_MANIFEST = "com.nvidia.simready.packaging.json"
USD_EXTS = (".usd", ".usda", ".usdc", ".usdz")
INDEX_VERSION = 2
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_ID_SUFFIX_RE = re.compile(r"_[A-Z0-9]{6,}$")


# Category words for browse queries ("find fruits"). Asset names describe the object itself, so a
# category expands to member words matched against name words (singular, as names use them).
CATEGORIES: dict[str, set[str]] = {
    "fruit": {"apple", "banana", "orange", "lemon", "lime", "pear", "peach", "plum", "grape", "cherry",
              "melon", "watermelon", "mango", "pineapple", "strawberry", "kiwi", "avocado", "coconut",
              "pumpkin", "fruit", "apricot", "nectarine", "papaya", "pomegranate", "fig", "tangerine"},
    "vegetable": {"carrot", "potato", "onion", "pepper", "squash", "tomato", "cucumber", "zucchini",
                  "eggplant", "cabbage", "lettuce", "broccoli", "cauliflower", "garlic", "corn",
                  "pumpkin", "radish", "celery", "leek", "ginger", "beet", "turnip", "yam"},
    "ball": {"ball", "football", "basketball", "baseball", "softball", "volleyball", "golfball", "golf",
             "tennis", "soccer", "rugby"},
    "drink": {"beverage", "bottle", "can", "soda", "beer", "wine", "juice", "milk", "water", "coffee"},
    "dishware": {"mug", "cup", "plate", "bowl", "dishware", "glass", "saucer", "jug", "pitcher", "teapot"},
    "tool": {"hammer", "wrench", "screwdriver", "pliers", "drill", "saw", "toolbox", "tool", "spanner",
             "clamp", "chisel", "mallet", "tape"},
    "toy": {"toy", "block", "doll", "puzzle", "teddy", "lego", "car", "ball"},
    "furniture": {"chair", "table", "desk", "shelf", "stool", "sofa", "couch", "cabinet", "bench", "bed"},
}
CATEGORIES["food"] = CATEGORIES["fruit"] | CATEGORIES["vegetable"] | {"food", "bread", "snack", "cereal"}
CATEGORIES["produce"] = CATEGORIES["fruit"] | CATEGORIES["vegetable"]
CATEGORIES["veggie"] = CATEGORIES["vegetable"]
CATEGORIES["dish"] = CATEGORIES["dishware"]
# Members that are also colours or other things ("crate orange"): count only on food assets.
AMBIGUOUS_MEMBERS = {"orange", "lime", "peach", "plum", "cherry", "pepper", "corn", "fig", "ginger"}


class DownloadCancelled(Exception):
    """Raised by ``SimReadyCatalog.fetch`` when its ``cancel`` event is set."""


@dataclass
class CatalogEntry:
    name: str  # human readable, e.g. "balance board"
    repo_id: str
    package_dir: str  # top-level folder inside the repo ("" when the USD path is standalone)
    root_usd: str  # repo-relative path of the entry USD
    tags: list[str] = field(default_factory=list)
    revision: str | None = None  # commit to download from (SimReady Central pins one); None = main
    dimensions: list[float] | None = None  # bounding box x, y, z [m], as SimReady Central reports it

    @property
    def key(self) -> str:
        return f"{self.repo_id}:{self.root_usd}"

    @property
    def search_text(self) -> str:
        return " ".join([self.name, *self.tags]).lower()


def humanize(package_dir: str) -> str:
    """'balance_board_BJ2USD8NG9' -> 'balance board'; 'CoffeeMug_01' -> 'coffee mug 01'."""
    name = _ID_SUFFIX_RE.sub("", package_dir.strip("_"))
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    name = re.sub(r"[_\-.]+", " ", name)
    return re.sub(r"\s+", " ", name).strip().lower()


_SOUND_CLASSES = str.maketrans("gcqbdvzx", "kkkptfss")


def sound_key(word: str) -> str:
    """Crude consonant skeleton for accent-tolerant matching: vowels and r/h/w/y dropped,
    voiced/unvoiced pairs merged, repeats collapsed. 'mark' -> 'mk', 'mug' -> 'mk'."""
    word = re.sub(r"[^a-z]", "", word.lower()).translate(_SOUND_CLASSES)
    key = re.sub(r"[aeiouyrhw]", "", word)
    return re.sub(r"(.)\1+", r"\1", key)


def _pick_root_usd(package_dir: str, files: list[str]) -> str | None:
    """Choose the package entry USD: a USD directly under ``simready_usd/`` (not payloads/layers)."""
    prefix = f"{package_dir}/simready_usd/"
    candidates = [
        f for f in files
        if f.startswith(prefix) and "/" not in f[len(prefix):]
        and f.lower().endswith(USD_EXTS) and not f[len(prefix):].startswith(".")
    ]
    if not candidates:
        return None

    def rank(path: str) -> tuple[int, int]:
        stem = Path(path).stem.lower()
        pkg = package_dir.lower()
        if stem == pkg:
            return (0, len(path))
        if stem.endswith("_main"):
            return (1, len(path))
        if pkg in stem:
            return (2, len(path))
        return (3, len(path))

    return sorted(candidates, key=rank)[0]


class SimReadyCatalog:
    def __init__(self, config: Config):
        self.config = config
        self.api = HfApi()
        self.entries: list[CatalogEntry] = []
        self._choices: dict[int, str] = {}
        self.index_path = config.cache_dir / "catalog_index.json"
        self._by_package: dict[tuple[str, str], CatalogEntry] = {}
        self._central_cache: dict[tuple[str, int], list[CatalogEntry]] = {}
        self._sizes: dict[str, tuple[int, int] | None] = {}
        self._no_dimensions: set[str] = set()

    # --- index ---------------------------------------------------------------------------------
    def load(self, refresh: bool = False) -> "SimReadyCatalog":
        if not refresh and self.index_path.exists():
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if data.get("version") == INDEX_VERSION:
                self.entries = [CatalogEntry(**e) for e in data["entries"]]
                self._reindex()
                log.info("Catalog: %d assets from cache (%s)", len(self.entries), self.index_path)
                return self
        self.build()
        return self

    def build(self) -> None:
        t0 = time.perf_counter()
        entries: dict[str, CatalogEntry] = {}
        repos = list(self.config.partner_repos)

        official, listed_repos = self._from_official_catalog()
        for e in official:
            entries.setdefault(e.key, e)
        for repo in listed_repos:
            if repo not in repos:
                repos.append(repo)

        for repo in repos:
            try:
                for e in self._crawl_partner_repo(repo):
                    entries.setdefault(e.key, e)
            except Exception as exc:  # one bad repo should not kill the index
                log.warning("Catalog: skipping %s (%s: %s)", repo, type(exc).__name__, exc)

        self.entries = sorted(entries.values(), key=lambda e: (e.name, e.repo_id))
        self._reindex()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps({"version": INDEX_VERSION, "built": time.time(), "repos": repos,
                        "entries": [asdict(e) for e in self.entries]}, indent=1),
            encoding="utf-8",
        )
        log.info("Catalog: indexed %d assets from %d repos in %.1fs",
                 len(self.entries), len(repos), time.perf_counter() - t0)

    def _reindex(self) -> None:
        self._choices = {i: e.search_text for i, e in enumerate(self.entries)}
        self._by_package = {(e.repo_id, e.package_dir): e for e in self.entries}

    def _from_official_catalog(self) -> tuple[list[CatalogEntry], list[str]]:
        """Parse the official index. Returns (entries, partner repo ids it mentions)."""
        repo = self.config.catalog_repo
        if not repo:
            return [], []
        try:
            files = self.api.list_repo_files(repo, repo_type="dataset")
        except Exception as exc:
            log.warning("Catalog: official index %s unavailable (%s). Run `hf auth login` and accept "
                        "its terms on huggingface.co to use it; falling back to partner repos.",
                        repo, type(exc).__name__)
            return [], []

        records: list[dict] = []
        for fname in files:
            low = fname.lower()
            if not low.endswith((".json", ".jsonl", ".csv", ".parquet")) or low.endswith(".gitattributes"):
                continue
            try:
                path = hf_hub_download(repo, fname, repo_type="dataset")
                records.extend(_read_records(Path(path)))
            except Exception as exc:
                log.debug("Catalog: could not read %s (%s)", fname, exc)

        entries, repos = [], set()
        for rec in records:
            flat = _flatten(rec)
            repo_ids = [v for k, v in flat.items() if isinstance(v, str) and "repo" in k.lower()
                        and _REPO_ID_RE.match(v)]
            repos.update(repo_ids)
            usd = next((v for v in flat.values() if isinstance(v, str) and v.lower().endswith(USD_EXTS)), None)
            if not (repo_ids and usd):
                continue
            usd = usd.split("/resolve/main/")[-1].lstrip("/")
            package_dir = usd.split("/")[0] if "/" in usd else ""
            name = next((flat[k] for k in flat if k.lower().split(".")[-1] in
                         ("name", "title", "asset_name", "display_name") and isinstance(flat[k], str)), None)
            tags = []
            for k, v in flat.items():
                if k.lower().split(".")[-1] in ("tags", "categories", "category", "keywords", "description"):
                    tags.extend(v if isinstance(v, list) else [str(v)])
            entries.append(CatalogEntry(
                name=(name or humanize(package_dir or Path(usd).stem)).lower(),
                repo_id=repo_ids[0], package_dir=package_dir, root_usd=usd,
                tags=[str(t).lower() for t in tags][:20],
            ))
        log.info("Catalog: official index gave %d assets, %d partner repos", len(entries), len(repos))
        return entries, sorted(repos)

    def _crawl_partner_repo(self, repo: str) -> list[CatalogEntry]:
        files = self.api.list_repo_files(repo, repo_type="dataset")
        by_package: dict[str, list[str]] = {}  # nvidia/simready-assets: ~180k files, ~2.7k packages
        for f in files:
            by_package.setdefault(f.split("/")[0], []).append(f)
        packages = sorted({f.split("/")[0] for f in files if f.endswith("/" + PACKAGE_MANIFEST)})
        org = repo.split("/")[0].lower()
        out = []
        for pkg in packages:
            root = _pick_root_usd(pkg, by_package[pkg])
            if root:
                out.append(CatalogEntry(name=humanize(pkg), repo_id=repo, package_dir=pkg,
                                        root_usd=root, tags=[org]))
        log.info("Catalog: %s -> %d packages", repo, len(out))
        return out

    # --- SimReady Central ---------------------------------------------------------------------
    def central_search(self, query: str, limit: int) -> list[CatalogEntry] | None:
        """Semantic search on SimReady Central (USD Search ``/search_hybrid``). Returns None when the
        service is off or unreachable, so callers fall back to the local index. Only hits with a text
        match count: the vector leg always returns *something* ("football" -> ping-pong balls)."""
        url = self.config.central_url.rstrip("/")
        query = query.lower().strip()
        if not query or not url or url.lower() == "none":
            return None
        cached = self._central_cache.get((query, limit))
        if cached is not None:
            return cached
        body = {"hybrid_text_query": query, "file_extension_include": "usd*", "limit": limit,
                "vector_queries": [{"field_name": "siglip2-embedding.embedding", "query_type": "text",
                                    "query": query}],
                "return_vision_generated_metadata": True, "return_usd_dimensions": True,
                "deduplicate_by_hash": True}
        request = urllib.request.Request(f"{url}/search_hybrid", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.config.central_timeout) as response:
                hits = json.load(response).get("hits", [])
        except Exception as exc:
            log.warning("SimReady Central unavailable (%s: %s); using the local index", type(exc).__name__, exc)
            return None
        found: dict[str, CatalogEntry] = {}
        for hit in hits:
            kinds = {x.get("search_type") for x in hit.get("metadata", {}).get("explanations", [])}
            entry = self._central_entry(hit.get("source", {}))
            if entry and kinds & {"hybrid", "name"}:
                found.setdefault(entry.key, entry)
        out = list(found.values())
        log.info("SimReady Central: '%s' -> %d assets (%d hits)", query, len(out), len(hits))
        if len(self._central_cache) > 64:
            self._central_cache.clear()
        self._central_cache[(query, limit)] = out
        return out

    def _central_entry(self, source: dict) -> CatalogEntry | None:
        """'hf://huggingface.co/datasets/<org>/<repo>/resolve/<sha>/<pkg>/simready_usd/<file>.usd' ->
        entry. Assets already in the local index keep their indexed name."""
        m = re.match(r"hf://huggingface\.co/datasets/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$", source.get("path", ""))
        if not m or "/simready_usd/" not in m.group(3):
            return None
        repo, revision, usd = m.groups()
        package_dir = usd.split("/simready_usd/")[0]
        dims = source.get("usd_dimensions") or {}
        mpu = dims.get("scene_mpu") or 1.0
        dimensions = ([float(dims[f"scaled_bbox_dimension_{a}"]) * mpu for a in "xyz"]
                      if all(f"scaled_bbox_dimension_{a}" in dims for a in "xyz") else None)
        known = self._by_package.get((repo, package_dir))
        if known:
            known.dimensions = known.dimensions or dimensions
            return known
        vlm = {x.get("name"): x.get("value_text") or []
               for x in source.get("plugin_thumbnail_to_vision_metadata_metadata_vlm_generated", [])}
        name = humanize(package_dir.split("/")[-1])
        if not re.search(r"\b[a-z]{3,}\b", name):  # opaque ids ("D00364"): use the VLM object type
            name = (vlm.get("object_type") or [name])[0].lower()
        caption = (vlm.get("caption") or [""])[0].lower()
        return CatalogEntry(name=name, repo_id=repo, package_dir=package_dir, root_usd=usd,
                            tags=[repo.split("/")[0].lower(), *caption.split()[:12]], revision=revision,
                            dimensions=dimensions)

    # --- search / fetch ------------------------------------------------------------------------
    def search(self, query: str, limit: int = 5) -> list[tuple[CatalogEntry, float]]:
        query = query.lower().strip()
        if not query or not self.entries:
            return []
        hits = process.extract(query, self._choices, scorer=fuzz.WRatio, limit=limit * 3)
        # Re-rank: prefer whole-word overlap so "ball" picks "ball" over "balance board".
        q_tokens = set(query.split())
        ranked = []
        for _, score, idx in hits:
            entry = self.entries[idx]
            overlap = len(q_tokens & set(entry.search_text.split())) / max(1, len(q_tokens))
            ranked.append((entry, score * 0.7 + 30.0 * overlap))
        ranked.sort(key=lambda t: -t[1])
        return ranked[:limit]

    def find_entries(self, query: str) -> list[CatalogEntry]:
        """Browse query results, every package (two scanned footballs are two entries): substring hits
        first ("ball" -> "football"), then close fuzzy matches. An empty query lists everything."""
        query = query.lower().strip()
        if not query:
            # Everyday props first; names with digits are mostly equipment part numbers ("60a tob").
            return sorted(self.entries, key=lambda e: (any(c.isdigit() for c in e.name), e.name))
        # Local name hits first (Central doesn't index every repo, e.g. ALLSIDES), then SimReady Central
        # (semantic, more datasets), then local category and fuzzy hits it missed.
        found = {e.key: e for e in self.entries if query in e.name}
        for e in self.central_search(query, self.config.central_limit) or []:
            found.setdefault(e.key, e)
        members = CATEGORIES.get(query) or CATEGORIES.get(query.rstrip("s"))
        if members:  # "fruit": names never say so, so match the member words ("food produce apple b01")
            for e in self.entries:
                words = set(e.name.split())
                hits = words & members
                if hits and (not hits <= AMBIGUOUS_MEMBERS or words & {"food", "produce", "fresh"}):
                    found.setdefault(e.key, e)
        for e in self.entries:
            if query in e.search_text:
                found.setdefault(e.key, e)
        for e, score in self.search(query, limit=20):
            if score >= max(self.config.min_match_score, 70.0):  # browsing: stricter than spawning
                found.setdefault(e.key, e)
        return list(found.values())

    def find(self, query: str, limit: int | None = None) -> list[str]:
        """Distinct asset names for a browse query (see find_entries)."""
        return list(dict.fromkeys(e.name for e in self.find_entries(query)))[:limit]

    def best_match(self, query: str) -> tuple[CatalogEntry, float] | None:
        hits = self.search(query, limit=1)
        if hits and hits[0][1] >= 90.0:  # the local index names it outright ("balance board")
            return hits[0]
        central = self.central_search(query, 5)
        if central:
            return central[0], 100.0
        if hits and hits[0][1] >= self.config.min_match_score:
            return hits[0]
        return self._sounds_like(query)

    def _sounds_like(self, query: str) -> tuple[CatalogEntry, float] | None:
        """Fallback for misheard words: match a consonant skeleton ("mark" and "mug" are both "mk")."""
        keys = [sound_key(w) for w in query.lower().split() if len(w) > 2]
        if not keys or any(len(k) < 2 for k in keys):  # short skeletons ("p", "t") match too much
            return None
        for entry in self.entries:
            words = [sound_key(w) for w in entry.name.split() if len(w) > 2]
            if all(k in words for k in keys):
                log.info("'%s' matched '%s' by sound", query, entry.name)
                return entry, self.config.min_match_score
        return None

    def _package_files(self, entry: CatalogEntry) -> list:
        """The files a download fetches (no thumbnails or metadata), from the Hub listing."""
        return [f for f in self.api.list_repo_tree(entry.repo_id, path_in_repo=entry.package_dir,
                                                   recursive=True, repo_type="dataset",
                                                   revision=entry.revision)
                if hasattr(f, "size") and "/.thumbs/" not in f.path and "/.metadata/" not in f.path]

    def package_size(self, entry: CatalogEntry) -> tuple[int, int] | None:
        """(bytes, files) a download would fetch, or None if the Hub can't be listed. Cached."""
        if entry.key not in self._sizes:
            try:
                files = self._package_files(entry) if entry.package_dir else []
                self._sizes[entry.key] = (sum(f.size for f in files), len(files)) if files else None
            except Exception as exc:
                log.debug("No size for %s: %s", entry.name, exc)
                return None
        return self._sizes[entry.key]

    def dimensions_of(self, entry: CatalogEntry) -> list[float] | None:
        """Bounding box x, y, z [m]: from SimReady Central (searching the asset's own name if no
        earlier search reported it), else measured from the root USD if it is downloaded."""
        if entry.dimensions is None and entry.key not in self._no_dimensions:
            for hit in self.central_search(entry.name, 40) or []:
                if hit.key == entry.key and hit is not entry:
                    entry.dimensions = hit.dimensions
            if entry.dimensions is None and self.is_downloaded(entry):
                entry.dimensions = _measure_usd(hf_hub_download(entry.repo_id, entry.root_usd, repo_type="dataset",
                                                                revision=entry.revision))
            if entry.dimensions is None:
                self._no_dimensions.add(entry.key)
        return entry.dimensions

    def is_downloaded(self, entry: CatalogEntry) -> bool:
        """Whether the root USD is already in the local Hugging Face cache."""
        return isinstance(try_to_load_from_cache(entry.repo_id, entry.root_usd, repo_type="dataset",
                                                 revision=entry.revision), str)

    def fetch(self, entry: CatalogEntry, progress: Callable[[int, int, int, int], None] | None = None,
              cancel: threading.Event | None = None) -> Path:
        """Download one package (USD + payloads + textures) and return the local root USD path.

        ``progress(done_bytes, total_bytes, done_files, total_files)`` is called as files finish
        (from worker threads). Scanned packages can be hundreds of MB, mostly textures.
        Setting ``cancel`` raises ``DownloadCancelled`` at once: queued files are dropped, and files
        already in flight finish in the background (``hf_hub_download`` can't be interrupted) and
        stay cached for a later attempt.
        """
        cancel = cancel or threading.Event()
        if not entry.package_dir:
            return Path(hf_hub_download(entry.repo_id, entry.root_usd, repo_type="dataset",
                                        revision=entry.revision))
        files = self._package_files(entry)
        total = sum(f.size for f in files)
        state = {"bytes": 0, "files": 0}
        lock = threading.Lock()

        def get(f) -> None:
            if cancel.is_set():
                return
            hf_hub_download(entry.repo_id, f.path, repo_type="dataset", revision=entry.revision)
            with lock:
                state["bytes"] += f.size
                state["files"] += 1
                snapshot = (state["bytes"], total, state["files"], len(files))
            if progress and not cancel.is_set():
                progress(*snapshot)

        pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="download")
        try:
            pending = {pool.submit(get, f) for f in files}
            while pending:
                done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()  # re-raise download errors
                if cancel.is_set():
                    raise DownloadCancelled(entry.name)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # in-flight files finish in the background
        path = Path(hf_hub_download(entry.repo_id, entry.root_usd, repo_type="dataset",
                                    revision=entry.revision))  # cached now
        if not path.exists():
            raise FileNotFoundError(f"{entry.root_usd} missing after download from {entry.repo_id}")
        return path


def _measure_usd(path: str) -> list[float] | None:
    """Bounding box of a USD's default prim in metres, as x, y, z with Z up."""
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(path)
        prim = stage.GetDefaultPrim()
        if not prim:
            return None
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]) \
            .ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            return None
        size = [float(v) * UsdGeom.GetStageMetersPerUnit(stage) for v in box.GetSize()]
        return [size[0], size[2], size[1]] if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y else size
    except Exception as exc:
        log.debug("Could not measure %s: %s", path, exc)
        return None


def _flatten(obj, prefix: str = "") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten(v, key))
            else:
                out[key] = v
    return out


def _read_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # optional
        except ImportError:
            log.debug("pyarrow not installed; skipping %s", path.name)
            return []
        return pq.read_table(path).to_pylist()
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".csv":
        return list(csv.DictReader(io.StringIO(text)))
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for value in data.values():  # {"assets": [...]} style
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
        return [data]
    return []


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="SimReady catalog tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refresh")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=10)
    f = sub.add_parser("fetch")
    f.add_argument("query")
    args = parser.parse_args()

    catalog = SimReadyCatalog(Config()).load(refresh=args.cmd == "refresh")
    if args.cmd == "search":
        for entry, score in catalog.search(args.query, args.n):
            print(f"{score:5.1f}  {entry.name:40s}  {entry.repo_id}/{entry.root_usd}")
    elif args.cmd == "fetch":
        match = catalog.best_match(args.query)
        if not match:
            raise SystemExit(f"No asset matches {args.query!r}")
        print(catalog.fetch(match[0]))


if __name__ == "__main__":
    main()
