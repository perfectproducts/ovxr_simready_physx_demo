"""Voice commands via Wit.ai, the speech/NLU service behind Meta's Voice SDK.

Hold A or X to talk: audio from the default input device (the Quest headset mic appears as a
Windows input device over Quest Link / Air Link) is buffered and sent to Wit.ai's ``/speech``
endpoint on release. If the Wit app has trained intents (see README), they are used; otherwise
the transcription is parsed with simple patterns, so an untrained Wit app works too.

Typed commands from the console go through the same parser, which is handy without a headset.
"""

from __future__ import annotations

import collections
import json
import logging
import queue
import re
import threading
from dataclasses import dataclass

import numpy as np
import requests
from scipy.signal import resample_poly  # import up front: lazily inside the frame loop it stalled ~9 s

PREROLL_S = 0.5  # audio kept from before the button press, so the first word isn't clipped

from .config import Config

log = logging.getLogger(__name__)

_NUMBERS = {"a": 1, "an": 1, "one": 1, "the": 1, "some": 3, "a few": 3, "two": 2, "a couple of": 2,
            "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_SPAWN_RE = re.compile(
    r"^(?:(?:hey|ok|okay|please|can you|could you|now)\s+)*"
    r"(?:add|spawn|create|drop|give me|bring(?: me)?|load|put|make|throw me|i want|i need|get me)\s+"
    r"(?:(\d+|a couple of|a few|an|a|one|two|three|four|five|six|seven|eight|nine|ten|some|the)\s+)?"
    r"(.+?)(?:\s+(?:please|here|now|in front of me|on the table))*$"
)
_QUERY_RES = (
    # "what can I add", "what do you have", "what objects are available"
    re.compile(r"^(?:what|which)\s+(?:objects?|things?|items?|assets?|stuff)?\s*(?:can i|could i|do you|is|are)\b.*$"),
    # "what balls are available", "which kinds of boxes do you have"
    re.compile(r"^(?:what|which)\s+(?:kinds? of\s+|sorts? of\s+|types? of\s+)?(?P<q>.+?)\s+"
               r"(?:are|is|do you|can i|could i|have you)\b.*$"),
    # "do you have any mugs", "are there footballs", "is there a toolbox"
    re.compile(r"^(?:do you have|have you got|is there|are there)\s+(?:any\s+|a\s+|an\s+|some\s+)?"
               r"(?P<q>.+?)(?:\s+(?:available|here|in there))?$"),
    # "list balls", "show me all the boxes", "search for tools"
    re.compile(r"^(?:list|show|search(?: for)?|find|look for)\s+(?:me\s+)?(?:all\s+)?(?:the\s+|some\s+|any\s+)?"
               r"(?P<q>.+?)(?:\s+(?:available|you have|there are))?$"),
)
_GENERIC_QUERY = {"", "objects", "object", "things", "thing", "items", "item", "assets", "asset", "stuff",
                  "everything", "all", "what", "options", "catalog", "catalogue", "browser", "menu", "list",
                  "library", "the catalog", "the browser"}
_BROWSER_CLOSE_RE = re.compile(r"\b(close|hide|exit|dismiss|remove)\b.*\b(catalog|catalogue|browser|menu|list|library)\b")
_BROWSER_PAGE_RE = re.compile(r"^(?:go\s+)?(?:to\s+)?(?:the\s+)?(next|previous|prev|last|back)(?:\s+page)?$")
_NEW_TOAST_RE = re.compile(r"\b(new|fresh|another|next)\s+(toast|slice|bread)\b|\breset\b.*\btoaster\b")
_CLEAR_RE = re.compile(r"\b(clear|remove|delete|reset)\b.*\b(all|everything|scene|table)\b|^(clear|reset)$")
_CANCEL_RE = re.compile(r"^(?:cancel|stop|abort)\b|\b(?:cancel|stop|abort)\b.*\b(?:download|downloading|loading)\b")
_UNFIX_RE = re.compile(r"\b(unfix|unlock|unpin|unfreeze|release|free)\b")
_FIX_RE = re.compile(r"^(?:please\s+)?(?:fix|fixed|lock|anchor|pin|freeze)\b")
_UNDO_RE = re.compile(r"\b(undo|remove (?:the )?last|delete (?:the )?last|take (?:it|that) away)\b")


@dataclass
class Command:
    kind: str  # spawn | clear | undo | fix | unfix | new_toast | query ("" = everything) | browser_close | browser_page (count = +-1) | cancel (downloads)
    query: str = ""
    count: int = 1
    text: str = ""


def _singular(noun: str) -> str:
    if noun.endswith("ies") and len(noun) > 4:
        return noun[:-3] + "y"
    if noun.endswith(("ses", "xes", "ches", "shes")):
        return noun[:-2]
    if noun.endswith("s") and not noun.endswith("ss") and len(noun) > 3:
        return noun[:-1]
    return noun


def parse_command(text: str, intents: list | None = None, entities: dict | None = None) -> Command | None:
    text = re.sub(r"[^\w\s']", " ", (text or "").lower()).strip()
    text = re.sub(r"\s+", " ", text)

    if _NEW_TOAST_RE.search(text):  # "new toast", "fresh toast", "reset the toaster"
        return Command("new_toast", text=text)
    if _UNFIX_RE.search(text):  # "free it", "unlock the box": before Wit's intents, which don't know these
        return Command("unfix", text=text)
    if _FIX_RE.search(text):  # "fix it", "lock it in place"
        return Command("fix", text=text)

    if intents:  # trained Wit app: intents spawn_object / clear_objects / undo
        top = max(intents, key=lambda i: i.get("confidence", 0))
        name = top.get("name", "")
        if top.get("confidence", 0) >= 0.6:
            if name in ("clear_objects", "clear"):
                return Command("clear", text=text)
            if name in ("undo", "remove_last"):
                return Command("undo", text=text)
            if name in ("list_objects", "query_objects"):
                values = {k.split(":")[0]: v[0].get("value") for k, v in (entities or {}).items() if v}
                obj = str(values.get("object") or values.get("wit$search_query") or "").lower()
                return Command("query", _singular(obj) if obj not in _GENERIC_QUERY else "", text=text)
            if name in ("spawn_object", "add_object", "spawn"):
                values = {k.split(":")[0]: v[0].get("value") for k, v in (entities or {}).items() if v}
                obj = values.get("object") or values.get("wit$search_query")
                count = values.get("wit$number") or values.get("number") or 1
                if obj:
                    return Command("spawn", _singular(str(obj).lower()), int(min(max(count, 1), 10)), text)

    if not text:
        return None
    if _CANCEL_RE.search(text):  # "cancel", "stop the download"
        return Command("cancel", text=text)
    if _BROWSER_CLOSE_RE.search(text):
        return Command("browser_close", text=text)
    m = _BROWSER_PAGE_RE.match(text)
    if m:
        return Command("browser_page", count=1 if m.group(1) == "next" else -1, text=text)
    if re.match(r"^(?:open|show)(?:\s+me)?\s+(?:the\s+)?(?:catalog|catalogue|browser|menu|library)$", text):
        return Command("query", "", text=text)
    for pattern in _QUERY_RES:  # before spawn: "show me balls" is a question, not "add balls"
        m = pattern.match(text)
        if m:
            obj = (m.groupdict().get("q") or "").strip()
            obj = re.sub(r"^(?:any|a|an|the|some|all)\s+", "", obj)
            return Command("query", "" if obj in _GENERIC_QUERY else _singular(obj), text=text)
    if _CLEAR_RE.search(text):
        return Command("clear", text=text)
    if _UNDO_RE.search(text):
        return Command("undo", text=text)
    m = _SPAWN_RE.match(text)
    if m:
        count_word, obj = m.group(1), m.group(2).strip()
        count = int(count_word) if count_word and count_word.isdigit() else _NUMBERS.get(count_word or "a", 1)
        return Command("spawn", _singular(obj), int(min(max(count, 1), 10)), text)
    return None


class WitClient:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()

    def recognize(self, pcm16: bytes) -> tuple[str, list, dict]:
        """Send 16 kHz mono int16 PCM; returns (text, intents, entities) from the final response."""
        # Push-to-talk audio starts mid-word; a little silence on both ends noticeably improves
        # recognition of the first word ("add" was heard as "it's" without it).
        pad = bytes(int(0.4 * self.config.sample_rate) * 2)
        pcm16 = pad + pcm16 + pad
        resp = self.session.post(
            f"https://api.wit.ai/speech?v={self.config.wit_api_version}",
            headers={
                "Authorization": f"Bearer {self.config.wit_token}",
                "Content-Type": f"audio/raw;encoding=signed-integer;bits=16;rate={self.config.sample_rate};endian=little",
                "Accept": "application/json",
            },
            data=pcm16, timeout=20,
        )
        resp.raise_for_status()
        # /speech streams several JSON objects (partial transcriptions, then the final result).
        final, decoder, body, pos = {}, json.JSONDecoder(), resp.text, 0
        while pos < len(body):
            while pos < len(body) and body[pos].isspace():
                pos += 1
            if pos >= len(body):
                break
            obj, pos = decoder.raw_decode(body, pos)
            if isinstance(obj, dict) and ("intents" in obj or obj.get("is_final") or not final):
                final = obj
        return final.get("text", ""), final.get("intents", []), final.get("entities", {})


_HOSTAPI_PREFERENCE = ("Windows WASAPI", "MME", "Windows DirectSound", "Windows WDM-KS")


def resolve_input_device(spec):
    """Map a --mic name fragment to one device index, preferring shared-mode Windows APIs.

    The same physical mic appears once per host API. WDM-KS opens it exclusively and, for the Quest
    Link mic, delivers silence, so shared APIs win when several match.
    """
    import sounddevice as sd

    if spec is None or isinstance(spec, int):
        return spec
    apis = sd.query_hostapis()
    matches = [(i, apis[d["hostapi"]]["name"]) for i, d in enumerate(sd.query_devices())
               if d["max_input_channels"] > 0 and spec.lower() in d["name"].lower()]
    if not matches:
        raise ValueError(f"no input device matches {spec!r} (see --list-mics)")
    rank = {name: n for n, name in enumerate(_HOSTAPI_PREFERENCE)}
    return min(matches, key=lambda m: rank.get(m[1], len(rank)))[0]


class VoiceListener:
    """Push-to-talk recorder. ``start()``/``stop()`` from the frame loop; commands land on ``commands``."""

    def __init__(self, config: Config, commands: "queue.Queue[Command]", feedback=None):
        self.config = config
        self.commands = commands
        self.feedback = feedback  # hud.Feedback, optional
        self.wit = WitClient(config)
        self._recording = False
        self._chunks: list[np.ndarray] = []
        self._preroll: collections.deque = collections.deque()
        self._preroll_len = 0
        self._lock = threading.Lock()
        self.stream = None
        self.capture_rate = config.sample_rate
        self.enabled = bool(config.wit_token)
        self.disabled_reason = ""
        if not self.enabled:
            self.disabled_reason = "Set WIT_AI_TOKEN and restart the app"
            log.warning("WIT_AI_TOKEN not set: voice disabled (typed console commands still work)")
            return
        try:
            import sounddevice as sd

            # Low-level inputs such as the Quest's "Headset Microphone (OCULUSVAD ...)" (WDM-KS) only
            # open at their native rate; capture there and resample to 16 kHz on release.
            device = resolve_input_device(config.mic_device)
            native = int(sd.query_devices(device, kind="input")["default_samplerate"])
            for rate in dict.fromkeys((config.sample_rate, native)):
                try:
                    self.stream = sd.InputStream(samplerate=rate, channels=1, dtype="int16",
                                                 device=device, callback=self._on_audio, blocksize=0)
                    break
                except sd.PortAudioError:
                    if rate == native:
                        raise
            self.capture_rate = int(self.stream.samplerate)
            self.stream.start()
            dev = sd.query_devices(self.stream.device)
            log.info("Voice: listening on '%s' at %d Hz (hold A/X to talk)", dev["name"], self.capture_rate)
        except Exception as exc:
            log.error("Voice: could not open microphone (%s). Use --list-mics / --mic.", exc)
            self.enabled = False
            self.disabled_reason = "Microphone unavailable: pick one with --mic"

    def _post(self, kind: str, title: str, detail: str = "") -> None:
        if self.feedback is not None:
            self.feedback.post(kind, title, detail)

    def _on_audio(self, data, frames, time_info, status) -> None:
        chunk = data[:, 0].copy()
        with self._lock:
            if self._recording:
                self._chunks.append(chunk)
            else:  # rolling pre-roll buffer
                self._preroll.append(chunk)
                self._preroll_len += len(chunk)
                while self._preroll and self._preroll_len - len(self._preroll[0]) >= PREROLL_S * self.capture_rate:
                    self._preroll_len -= len(self._preroll.popleft())

    def start(self) -> None:
        if not self.enabled:
            self._post("error", "Voice is off", self.disabled_reason)
            return
        with self._lock:
            self._chunks = list(self._preroll)
            self._preroll.clear()
            self._preroll_len = 0
            self._recording = True
        self._post("listening", "Listening…", "Release the button when you're done")

    def stop(self) -> None:
        if not self.enabled or not self._recording:
            return
        self._recording = False
        with self._lock:
            audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.int16)
            self._chunks = []
        seconds = len(audio) / self.capture_rate
        if seconds < 0.3:
            log.info("Voice: too short (%.2fs), ignored", seconds)
            self._post("error", "Didn't hear anything", "Hold A or X while you speak")
            return
        peak = int(np.abs(audio.astype(np.int32)).max())
        log.info("Voice: %.1fs captured, peak level %d/32767", seconds, peak)
        if peak < 50:
            self._post("error", "Microphone is silent",
                       "Enable the headset mic for Link (Windows sound settings) or pick one with --mic")
            return
        if self.capture_rate != self.config.sample_rate:
            from scipy.signal import resample_poly

            audio = np.clip(resample_poly(audio.astype(np.float32), self.config.sample_rate, self.capture_rate),
                            -32768, 32767).astype(np.int16)
        audio = audio[: int(self.config.max_utterance_s * self.config.sample_rate)]
        self._post("thinking", "Recognizing…", f"{seconds:.1f}s of audio sent to Wit.ai")
        threading.Thread(target=self._recognize, args=(audio.tobytes(),), daemon=True).start()

    def _recognize(self, pcm: bytes) -> None:
        try:
            text, intents, entities = self.wit.recognize(pcm)
        except Exception as exc:
            log.error("Voice: Wit.ai request failed: %s", exc)
            self._post("error", "Wit.ai request failed", str(exc))
            return
        command = parse_command(text, intents, entities)
        log.info("Voice: heard %r -> %s", text, command)
        if command:
            self.commands.put(command)  # the app posts what it does with it
        elif text:
            self._post("error", f"“{text}”", "Not a command. Try: add a mug · spawn three footballs · undo")
        else:
            self._post("error", "Didn't catch that", "Speak a little closer to the headset and try again")

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


def start_console(commands: "queue.Queue[Command]") -> None:
    """Read typed commands ('add a mug', 'clear all', 'undo') from stdin on a daemon thread."""

    def run() -> None:
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                return
            command = parse_command(line)
            if command:
                commands.put(command)
            elif line.strip():
                print("  try: 'add a mug', 'spawn three footballs', 'undo', 'clear all'")

    threading.Thread(target=run, daemon=True, name="console").start()
