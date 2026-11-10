"""Python sidecar for the C++ app (cpp/): voice, SimReady catalog, downloads and HUD text.

The C++ app owns everything per-frame (rendering, ovphysx, controllers). This process does the
slow, non-real-time work with the code the Python app already uses, and talks to the app over
one localhost TCP connection carrying JSON lines.

  app -> sidecar   {"t":"talk","on":true|false}          push-to-talk pressed / released
                   {"t":"spawned","name":..,"count":n,"total":k}
                   {"t":"removed","name":..,"total":k}   {"t":"cleared","n":k}
                   {"t":"fixed"|"unfixed","name":..,"parts":n}   {"t":"nothing_to_fix"}
                   {"t":"placing","name":..,"count":n}  {"t":"placing_cancelled"}   (spawn waits for a spot)
  sidecar -> app   {"t":"spawn","name":..,"count":n,"root_usd":..,"asset_ref":"@wrapper@</Prop/Asset>","correction":[16],
                    "body_paths":[..],"bottom_z":f,"size":[3],
                    "body_spheres":[[cx,cy,cz,r],..],"body_movable":[bool,..],
                    "position":[x,y,z] (optional: place the asset origin there, unrotated)}
                   {"t":"undo"}  {"t":"clear"}  {"t":"fix","on":true|false}  (a held / last touched prop)
                   {"t":"hud","kind":..,"hold":s,"points":"<base64 float32 x,y,z ...>"}
  toaster (vrsim/toaster.py): spawned with "device":"toaster", then
  app -> sidecar   {"t":"toaster_in","lever":m,"dial":deg,"cancel":m}
  sidecar -> app   {"t":"toaster","lever_target":m,"toast":[x,y,z,rot],"toast_color":[r,g,b],"glow":..,...}

    uv run python -m vrsim.sidecar           (then start cpp\\build\\vrsim_cpp.exe)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import queue
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np

from .assets import NotDynamicError, describe_asset
from .browser import PAGE_SIZE, BrowserBuilder, Page
from .catalog import DownloadCancelled, SimReadyCatalog
from .config import Config
from .hud import Feedback, TextLayout, download_progress
from .scene import write_desk_prop, write_environment, write_placement_hulls
from .toaster import ToasterDevice, ToasterScenario
from .voice import Command, VoiceListener, start_console

log = logging.getLogger("vrsim.sidecar")
PORT = 47800


class Link:
    """The single app connection; sends are thread-safe and silently dropped while disconnected."""

    def __init__(self):
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self.connected = threading.Event()

    def attach(self, sock: socket.socket | None) -> None:
        with self._lock:
            self._sock = sock
        (self.connected.set if sock else self.connected.clear)()

    def send(self, message: dict) -> None:
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        with self._lock:
            if self._sock is None:
                return
            try:
                self._sock.sendall(data)
            except OSError as exc:
                log.warning("send failed: %s", exc)


class RelayFeedback(Feedback):
    """Feedback that also rasterizes each message and ships the HUD text mesh to the app."""

    def __init__(self, link: Link):
        super().__init__()
        self.link = link
        self.layout = TextLayout()

    def post(self, kind: str, title: str, detail: str = "", hold: float | None = None) -> None:
        super().post(kind, title, detail, hold)
        msg = self.current()
        points = self.layout.points(title, detail).astype(np.float32)
        self.link.send({"t": "hud", "kind": kind, "hold": msg.hold,
                        "points": base64.b64encode(points.tobytes()).decode("ascii")})

    def resend(self) -> None:
        msg = self.current()
        if msg.title:
            self.post(msg.kind, msg.title, msg.detail, msg.hold)


class Sidecar:
    def __init__(self, config: Config, refresh: bool):
        self.config = config
        self.link = Link()
        self.feedback = RelayFeedback(self.link)
        self.commands: "queue.Queue[Command]" = queue.Queue()
        self.catalog = SimReadyCatalog(config).load(refresh=refresh)
        self.voice = VoiceListener(config, self.commands, self.feedback)
        self.browser = BrowserBuilder(config.generated_dir)
        self.env_path = write_environment(config, config.generated_dir / "environment_physics.usda", physics=True,
                                          loading_layer=self.browser.loading_layer())
        # vrsim_cpp --render minimal loads this variant: Minimal rendering ignores the dome light.
        minimal = write_environment(config, config.generated_dir / "environment_physics_minimal.usda", physics=True,
                                    loading_layer=self.browser.loading_layer(), fake_sky=True)
        log.info("Environment for the C++ app: %s (Minimal rendering: %s)", self.env_path, minimal.name)
        write_placement_hulls(config, config.generated_dir / "placement_hulls.bin")  # the app aims placements at them
        # Scene objects made grabbable (the NVIDIA mug): spawned where they stood on every connect.
        self.desk_props = []
        self._quiet_spawns = 0
        if config.environment.usd is not None:
            for name in config.environment.desk_props:
                built = write_desk_prop(config, name, config.generated_dir)
                if built:
                    path, position = built
                    self.desk_props.append((name, describe_asset(path, config.generated_dir), position))
        # The dbeyond smart toaster: spawned fixed on the desk on every connect, its FMUs run here.
        self.toaster: ToasterScenario | None = None
        self.toaster_info = None
        self.toaster_device: ToasterDevice | None = None
        if config.environment.usd is not None and config.environment.toaster.enabled:
            try:
                t = config.environment.toaster
                self.toaster = ToasterScenario.load(t.position, t.yaw)
                self.toaster_info = describe_asset(self.toaster.toaster_usd, config.generated_dir)
                log.info("Toaster station: %s (%d FMUs)", self.toaster.toaster_usd.name, len(self.toaster.instances))
            except Exception:
                log.exception("Toaster station not available")
                self.toaster = None
        self._infos = {}
        self._downloads: dict[str, threading.Event] = {}  # entry.key -> cancel flag, while downloading
        self._downloads_lock = threading.Lock()
        # Downloads must never block browsing: spawns get their own workers, and page builds a
        # single worker where the newest request wins.
        self._spawn_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="spawn")
        self._page_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser")
        self._info_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="info")
        self._hover_request = 0
        self._page_lock = threading.Lock()
        self._page_request = 0
        self._page_index = 0
        self.browser_entries = []
        self.browser_title = ""
        self.page: Page | None = None
        threading.Thread(target=self._command_loop, daemon=True, name="commands").start()

    # --- app connection ------------------------------------------------------------------------
    def serve(self) -> None:
        server = socket.create_server(("127.0.0.1", PORT))
        log.info("Waiting for the C++ app on 127.0.0.1:%d ...", PORT)
        while True:
            sock, addr = server.accept()
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            log.info("App connected from %s:%d", *addr)
            self.link.attach(sock)
            self._quiet_spawns = len(self.desk_props)  # keep the connect hint on the panel
            for name, info, position in self.desk_props:
                self.link.send({"t": "spawn", "name": name, "count": 1, **self._spawn_payload(info),
                                "position": list(position)})
            self._start_toaster()
            self.feedback.post("info", "Hold A or X and speak",
                               "“add a mug” · “what balls are available?” · “undo” · “clear all”", hold=12.0)
            try:
                self._read_loop(sock)
            finally:
                self._stop_toaster()
                self.link.attach(None)
                sock.close()
                log.info("App disconnected")

    def _read_loop(self, sock: socket.socket) -> None:
        buf = b""
        while True:
            try:
                chunk = sock.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    self._on_app_message(json.loads(line))

    def _on_app_message(self, msg: dict) -> None:
        kind = msg.get("t")
        if kind == "talk":
            (self.voice.start if msg.get("on") else self.voice.stop)()
        elif kind == "spawned" and self._quiet_spawns > 0:
            self._quiet_spawns -= 1
        elif kind == "spawned":
            n, total = msg.get("count", 1), msg.get("total", 0)
            title = f"Added {n} × {msg['name']}" if n > 1 else f"Added {msg['name']}"
            self.feedback.post("ok", title, "fixed in place · hold it and use the thumbstick to turn it · "
                                            "trigger while holding frees it")
        elif kind == "removed":
            self.feedback.post("ok", f"Removed {msg['name']}", f"{msg.get('total', 0)} objects in the scene")
        elif kind == "cleared":
            n = msg.get("n", 0)
            self.feedback.post("ok" if n else "info", f"Cleared {n} object{'s' if n != 1 else ''}" if n
                               else "Nothing to clear")
        elif kind == "placing":
            n = msg.get("count", 1)
            self.feedback.post("info", f"Where should {'they' if n > 1 else 'it'} stand?",
                               f"point the red marker at a spot for {msg['name']} · thumbstick turns it · "
                               "trigger places it · B cancels", hold=30.0)
        elif kind == "placing_cancelled":
            self.feedback.post("info", "Placement cancelled")
        elif kind == "fixed":
            parts = msg.get("parts", 0)
            self.feedback.post("ok", f"Fixed {msg['name']} in place",
                               (f"grab its {parts} jointed part{'s' if parts != 1 else ''} to move them · "
                                if parts else "") + "trigger while holding it to free it")
        elif kind == "unfixed":
            self.feedback.post("ok", f"Freed {msg['name']}", "grip to grab, release to throw")
        elif kind == "nothing_to_fix":
            self.feedback.post("info", "Nothing to fix", "Say “add a mug” to spawn something")
        elif kind == "nothing_to_undo":
            self.feedback.post("info", "Nothing to undo", "Say “add a mug” to spawn something")
        elif kind == "browser_pick":
            self.commands.put(Command("browser_pick", count=int(msg.get("index", -1))))
        elif kind == "browser_hover":
            self._hover_request += 1
            index = int(msg.get("index", -1))
            if self.page and 0 <= index < len(self.page.entries):
                self._info_pool.submit(self._safely, self._show_entry_info, self._hover_request,
                                       self.page.entries[index])
        elif kind == "browser_page":
            self.commands.put(Command("browser_page", count=int(msg.get("delta", 1))))
        elif kind == "browser_closed":  # closed from the panel's ✕ button
            self.page = None
        elif kind == "hud_closed":  # the info panel's ×; while downloading it means "cancel"
            self.commands.put(Command("cancel", text="✕"))
        elif kind == "browser_toggle":
            self.commands.put(Command("browser_close") if self.page else Command("query", ""))
        elif kind == "toaster_in" and self.toaster_device:
            self.toaster_device.set_inputs(msg.get("lever"), msg.get("dial"), msg.get("cancel"))
        elif kind == "new_toast":
            self._start_toaster(respawn=False)
            self.feedback.post("ok", "Fresh toast", "a new slice and a cold toaster", hold=4.0)
        elif kind == "log":
            log.info("app: %s", msg.get("text"))

    # --- voice / console commands --------------------------------------------------------------
    def _command_loop(self) -> None:
        while True:
            command = self.commands.get()
            log.info("Command: %s", command)
            try:
                if command.kind == "spawn":
                    self._spawn_pool.submit(self._safely, self._spawn, command)
                elif command.kind == "query":
                    self._answer_query(command.query)
                elif command.kind == "cancel":
                    self._cancel_downloads(command.text != "✕")
                elif command.kind == "new_toast":
                    self._on_app_message({"t": "new_toast"})
                elif command.kind in ("undo", "clear"):
                    self.link.send({"t": command.kind})
                elif command.kind in ("fix", "unfix"):
                    self.link.send({"t": "fix", "on": command.kind == "fix"})
                elif command.kind == "browser_page" and self.browser_entries:
                    self._request_page(self._page_index + command.count)
                elif command.kind == "browser_close":
                    with self._page_lock:
                        self._page_request += 1  # drop any page still being built
                    self.page = None
                    self.link.send({"t": "browser_close"})
                elif command.kind == "browser_pick" and self.page and 0 <= command.count < len(self.page.entries):
                    entry = self.page.entries[command.count]
                    self._spawn_pool.submit(self._safely, self._spawn_entry, entry, 1, f"Add {entry.name}")
            except Exception as exc:
                log.exception("Command failed")
                self.feedback.post("error", "Something went wrong", str(exc))

    def _start_toaster(self, respawn: bool = True) -> None:
        """(Re)start the toaster's FMUs from their initial state; on connect also spawn it."""
        if self.toaster is None:
            return
        self._stop_toaster()
        if respawn:
            self._quiet_spawns += 1
            self.link.send({"t": "spawn", "name": "Smart toaster", "count": 1, **self._spawn_payload(self.toaster_info),
                            "position": list(self.toaster.position), **self.toaster.spawn_extras()})
        self.toaster_device = ToasterDevice(self.toaster, self.link.send, self.feedback)

    def _stop_toaster(self) -> None:
        device, self.toaster_device = self.toaster_device, None
        if device:
            device.stop()

    def _browser_anchor(self) -> list[float]:
        """Where the catalog first appears: centred over the table, just inside its far edge (x, y,
        bottom z in metres). The bottom sits 0.3 m above the top so the tiles clear what stands on
        the desk (Studierstube's laptop screen) from standing eye height."""
        t = self.config.table
        return [t.center[0], t.center[1] + t.size[1] / 2 - 0.08, t.height + 0.3]

    def _cancel_downloads(self, spoken: bool) -> None:
        with self._downloads_lock:
            flags = list(self._downloads.values())
        for flag in flags:
            flag.set()
        if not flags and spoken:
            self.feedback.post("info", "Nothing to cancel", "No download is running")

    def _safely(self, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as exc:
            log.exception("Background task failed")
            self.feedback.post("error", "Something went wrong", str(exc))

    def _spawn(self, command: Command) -> None:
        heard = f"“{command.text}”" if command.text else f"Add {command.count} × {command.query}"
        self.feedback.post("thinking", heard, f"Searching SimReady for '{command.query}'…")
        match = self.catalog.best_match(command.query)
        if not match:
            closest = ", ".join(e.name for e, _ in self.catalog.search(command.query, 3))
            self.feedback.post("error", f"No SimReady asset for '{command.query}'",
                               f"Closest: {closest}" if closest else "")
            return
        self._spawn_entry(match[0], command.count, heard)

    def _spawn_entry(self, entry, count: int, heard: str) -> None:
        self.feedback.post("thinking", heard, f"{entry.name} · {entry.repo_id}")
        info = self._infos.get(entry.key)
        if info is None:
            self.feedback.post("thinking", f"Found: {entry.name}", f"Downloading from {entry.repo_id}…")
            with self._downloads_lock:
                cancel = self._downloads.setdefault(entry.key, threading.Event())
            try:
                root = self.catalog.fetch(entry, download_progress(self.feedback, entry.name, cancel), cancel)
                info = describe_asset(root, self.config.generated_dir)
            except DownloadCancelled:
                self.feedback.post("info", f"Cancelled {entry.name}", "Download stopped · nothing was added")
                return
            except NotDynamicError:
                self.feedback.post("error", f"Can't pick up '{entry.name}'",
                                   "That asset is static (no rigid body). Try something smaller")
                return
            finally:
                with self._downloads_lock:
                    if self._downloads.get(entry.key) is cancel:
                        del self._downloads[entry.key]
            if cancel.is_set():  # cancelled just as the last file arrived
                self.feedback.post("info", f"Cancelled {entry.name}", "Download stopped · nothing was added")
                return
            self._infos[entry.key] = info
        if not self.link.connected.is_set():
            self.feedback.post("error", "The VR app is not connected", "Start cpp\\build\\vrsim_cpp.exe")
            return
        self.link.send({"t": "spawn", "name": entry.name, "count": count, **self._spawn_payload(info)})

    @staticmethod
    def _spawn_payload(info) -> dict:
        payload = asdict(info)
        payload["root_usd"] = info.root_usd.as_posix()
        return payload

    # --- asset browser -------------------------------------------------------------------------
    def _show_entry_info(self, request: int, entry) -> None:
        """Size and download size of the catalog tile under the pointer, on the info panel. Waits a
        moment so sweeping across tiles doesn't flicker, and stays quiet while downloading."""
        time.sleep(0.25)
        if request != self._hover_request or self._downloads:
            return
        dims = self.catalog.dimensions_of(entry)
        if dims is None and entry.key in self._infos:
            dims = self._infos[entry.key].size
        parts = [_format_dimensions(dims) if dims else "dimensions known after download"]
        size = self.catalog.package_size(entry)
        if size:
            parts.append(f"{_format_bytes(size[0])} download ({size[1]} files)")
        if self.catalog.is_downloaded(entry):
            parts.append("already downloaded")
        parts.append(entry.repo_id)
        if request == self._hover_request and not self._downloads:
            self.feedback.post("info", entry.name, " · ".join(parts), hold=6.0)

    def _request_page(self, index: int) -> None:
        pages = max(1, -(-len(self.browser_entries) // PAGE_SIZE))
        with self._page_lock:
            self._page_request += 1
            request = self._page_request
            self._page_index = min(max(index, 0), pages - 1)
            index = self._page_index
        self.link.send({"t": "browser_loading", "anchor": self._browser_anchor()})  # spinner until the page arrives
        self._page_pool.submit(self._safely, self._build_page, request, index, list(self.browser_entries),
                               self.browser_title)

    def _build_page(self, request: int, index: int, entries: list, title: str) -> None:
        if request != self._page_request:
            return  # superseded by a newer flick / query
        page = self.browser.build(entries, index, title)
        if request != self._page_request:
            return
        self.page = page
        self.link.send({**page.message(), "anchor": self._browser_anchor()})
        self.browser.prefetch(entries, index + 1)  # make the next flick instant

    def _answer_query(self, query: str) -> None:
        entries = self.catalog.find_entries(query)
        if entries:  # open the in-VR browser on the results (all assets for a generic query)
            self.browser_entries = entries
            self.browser_title = f"'{query}' · {len(entries)} assets" if query else f"SimReady · {len(entries)} assets"
            self._request_page(0)
        names = self.catalog.find(query)
        shown = ", ".join(names[:8]) + (f" +{len(names) - 8} more" if len(names) > 8 else "")
        if not query:
            self.feedback.post("info", f"{len(names)} SimReady assets available", f"e.g. {shown}", hold=12.0)
        elif names:
            self.feedback.post("info", f"{len(names)} match{'es' if len(names) != 1 else ''} for '{query}'",
                               f"{shown} · say “add a …”", hold=12.0)
        else:
            closest = ", ".join(e.name for e, _ in self.catalog.search(query, 3))
            self.feedback.post("error", f"No SimReady assets for '{query}'",
                               f"Closest: {closest}" if closest else "")


def _format_dimensions(dims) -> str:
    """'12.1 × 8.1 × 9.7 cm' (metres from 1 m up)."""
    if max(dims) >= 1.0:
        return " × ".join(f"{d:.2f}" for d in dims) + " m"
    return " × ".join(f"{d * 100:.1f}" for d in dims) + " cm"


def _format_bytes(n: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice/catalog sidecar for the C++ VR app")
    parser.add_argument("--mic", help="Input device name or index")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--no-console", action="store_true")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx2", "httpx", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = Config()
    if args.mic:
        config.mic_device = int(args.mic) if args.mic.isdigit() else args.mic
    sidecar = Sidecar(config, refresh=args.refresh_catalog)
    if not args.no_console:
        start_console(sidecar.commands)
    sidecar.serve()


if __name__ == "__main__":
    main()
