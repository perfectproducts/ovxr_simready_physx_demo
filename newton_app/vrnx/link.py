"""The app's end of the sidecar connection: JSON lines over localhost TCP (see vrsim/sidecar.py).

A background thread (re)connects and reads; the frame loop polls parsed messages without blocking
and sends without waiting. While the sidecar is down, sends are dropped.
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time

log = logging.getLogger(__name__)


class SidecarLink:
    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.address = (host, port)
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._inbox: "queue.Queue[dict]" = queue.Queue()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="sidecar-link").start()

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _run(self) -> None:
        warned = False
        while not self._stop.is_set():
            try:
                sock = socket.create_connection(self.address, timeout=2.0)
            except OSError:
                if not warned:
                    log.info("Waiting for the sidecar on %s:%d (python -m vrnx.sidecar)...", *self.address)
                    warned = True
                time.sleep(1.0)
                continue
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._sock = sock
            log.info("Sidecar connected")
            warned = False
            buf = b""
            try:
                while not self._stop.is_set():
                    chunk = sock.recv(1 << 20)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.strip():
                            try:
                                self._inbox.put(json.loads(line))
                            except json.JSONDecodeError:
                                log.warning("Bad sidecar line: %.80s", line)
            except OSError:
                pass
            with self._lock:
                self._sock = None
            sock.close()
            log.info("Sidecar disconnected; reconnecting...")

    def poll(self) -> list[dict]:
        out = []
        while True:
            try:
                out.append(self._inbox.get_nowait())
            except queue.Empty:
                return out

    def send(self, message: dict) -> None:
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        with self._lock:
            if self._sock is None:
                return
            try:
                self._sock.sendall(data)
            except OSError as exc:
                log.warning("send failed: %s", exc)

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            if self._sock is not None:
                self._sock.close()
