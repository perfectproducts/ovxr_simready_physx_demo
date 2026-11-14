"""3D sound for the headset: short sounds placed in the world, heard from the user's head.

The sidecar plays them on the default output device (the Quest's "Headphones (Oculus Virtual Audio
Device)" while Link sends audio to the headset; ``VRSIM_SPEAKER`` / ``--speaker`` picks another).
The apps send the head's world pose with their toaster readings (``toaster_in`` field ``head``:
x, y, z, qx, qy, qz, qw, OpenXR view axes: +x right, +y up, -z forward), and every 10 ms block is
rendered binaurally for where the head is then, so turning toward the sound moves it:

- interaural time difference (Woodworth's spherical head, up to ~0.66 ms),
- head shadow: the far ear is quieter and low-passed, more the further the source is to the side,
- a softer, duller sound from behind (front/back cue), and 1/r loudness beyond 0.6 m.

The ding is ``assets/sounds/ding.wav`` (or the file ``VRSIM_DING_WAV`` names), mixed to mono and
resampled; without one, a bell synthesized here.

Gains, delays and filters are interpolated across each block, so head motion doesn't click.

    uv run python -m vrsim.spatial_audio      (a ding circling the listener, for a quick listen)
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from .config import PROJECT_ROOT

log = logging.getLogger(__name__)

SOUNDS_DIR = PROJECT_ROOT / "assets" / "sounds"  # <name>.wav here replaces a built-in sound

HEAD_RADIUS = 0.0875  # [m]
SPEED_OF_SOUND = 343.0  # [m/s]
REF_DISTANCE = 0.6  # [m] full loudness up to here, 1/r beyond
BLOCK = 480  # frames per callback at 48 kHz: 10 ms
LISTENER_STALE_S = 1.0  # no head pose for this long: play centred (mono)


def ding(rate: int, f0: float = 1975.0, seconds: float = 1.8) -> np.ndarray:
    """A small bell, like a toaster's: inharmonic partials with their own decays and a short strike."""
    t = np.arange(int(rate * seconds)) / rate
    partials = ((1.0, 1.0, 1.4), (2.0, 0.32, 0.7), (2.76, 0.22, 0.45), (5.40, 0.10, 0.18), (8.93, 0.05, 0.08))
    out = sum(a * np.exp(-t / d) * np.sin(2 * np.pi * f0 * r * t + r) for r, a, d in partials)
    strike = np.random.default_rng(7).standard_normal(len(t)) * np.exp(-t / 0.003) * 0.25
    out = out + lfilter([1, -1], [1, -0.6], strike)  # high-passed click of the hammer
    out *= np.minimum(1.0, t / 0.0015)  # 1.5 ms attack: no step at the start
    return (0.6 * out / np.abs(out).max()).astype(np.float32)


def load_wav(path: Path, rate: int) -> np.ndarray:
    """A WAV file as mono float32 at ``rate`` (channels averaged: the source is a point, placed by
    the renderer). 8/16/24/32-bit PCM and float files; peaks above 0.9 are scaled down."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    file_rate, data = wavfile.read(str(path))
    if data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.integer):
        x = data.astype(np.float32) / float(np.iinfo(data.dtype).max + 1)
    else:
        x = data.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if file_rate != rate:
        g = math.gcd(int(file_rate), int(rate))
        x = resample_poly(x, rate // g, file_rate // g).astype(np.float32)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 0.9:
        x *= 0.9 / peak
    return x


def _quat_rotate_inv(q, v) -> np.ndarray:
    """Rotate world vector ``v`` into the frame of unit quaternion ``q`` (x, y, z, w)."""
    x, y, z, w = q
    u = np.array([-x, -y, -z])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _one_pole(cutoff: float, rate: int) -> float:
    return float(np.exp(-2.0 * np.pi * min(cutoff, 0.45 * rate) / rate))


class _Voice:
    def __init__(self, samples: np.ndarray, position, gain: float):
        self.samples = samples
        self.position = np.asarray(position, dtype=np.float64)
        self.gain = gain
        self.pos = 0  # next sample index at zero delay
        self.prev: np.ndarray | None = None  # gains (2), delays (2) [samples], filter coefs (2) at the last block end
        self.zi = [np.zeros(1), np.zeros(1)]

    def params(self, head_p, head_q, rate: int) -> np.ndarray:
        if head_p is None:  # no listener: centred, at a moderate level
            return np.array([0.7, 0.7, 0.0, 0.0, 0.0, 0.0])
        d = _quat_rotate_inv(head_q, self.position - head_p)  # in the head frame
        r = max(float(np.linalg.norm(d)), 1e-3)
        lateral = float(np.clip(d[0] / r, -1.0, 1.0))  # +1 right, -1 left
        front = float(np.clip(-d[2] / r, -1.0, 1.0))  # +1 ahead, -1 behind
        theta = np.arcsin(lateral)
        itd = HEAD_RADIUS / SPEED_OF_SOUND * (abs(theta) + np.sin(abs(theta))) * rate  # [samples]
        side = abs(lateral)
        near, far = 1.0 + 0.1 * side, 1.0 - 0.4 * side
        far_cut = 18000.0 * (1500.0 / 18000.0) ** side  # head shadow
        rear_cut = 18000.0 * (6000.0 / 18000.0) ** max(0.0, -front)
        rear_gain = 1.0 - 0.15 * max(0.0, -front)
        g = self.gain * rear_gain * min(1.0, REF_DISTANCE / r)
        near_coef, far_coef = _one_pole(rear_cut, rate), _one_pole(min(far_cut, rear_cut), rate)
        if lateral >= 0:  # source on the right: the left ear is the far one
            return np.array([g * far, g * near, itd, 0.0, far_coef, near_coef])
        return np.array([g * near, g * far, 0.0, itd, near_coef, far_coef])

    def render(self, n: int, head_p, head_q, rate: int) -> tuple[np.ndarray, bool]:
        target = self.params(head_p, head_q, rate)
        prev = target if self.prev is None else self.prev
        ramp = (np.arange(1, n + 1) / n)[:, None]
        p = prev + (target - prev) * ramp  # (n, 6), per-sample parameters
        self.prev = target
        idx = self.pos + np.arange(n, dtype=np.float64)
        grid = np.arange(len(self.samples))
        out = np.empty((n, 2), np.float32)
        for ear in range(2):
            x = np.interp(idx - p[:, 2 + ear], grid, self.samples, left=0.0, right=0.0)
            a = float(p[-1, 4 + ear])  # the filter changes slowly; one coefficient per block
            y, self.zi[ear] = lfilter([1.0 - a], [1.0, -a], x, zi=self.zi[ear] * 1.0)
            out[:, ear] = y * p[:, ear]
        self.pos += n
        return out, self.pos - 64 < len(self.samples)


class SpatialAudio:
    """Mixes the playing sounds into one stereo output stream; ``play`` and ``set_listener`` are thread-safe."""

    def __init__(self, device=None):
        self.enabled = False
        self.rate = 48000
        self._voices: list[_Voice] = []
        self._lock = threading.Lock()
        self._head_p = self._head_q = None
        self._head_time = 0.0
        self._sounds: dict[str, np.ndarray] = {}
        self.stream = None
        try:
            import sounddevice as sd

            if isinstance(device, str) and device.isdigit():
                device = int(device)
            try:
                self.stream = sd.OutputStream(samplerate=self.rate, channels=2, dtype="float32", device=device,
                                              blocksize=BLOCK, latency="low", callback=self._callback)
            except sd.PortAudioError:  # a device that only runs at its own rate
                self.rate = int(sd.query_devices(device, kind="output")["default_samplerate"])
                self.stream = sd.OutputStream(samplerate=self.rate, channels=2, dtype="float32", device=device,
                                              latency="low", callback=self._callback)
            self.stream.start()
            self.enabled = True
            log.info("3D sound on '%s' at %d Hz", sd.query_devices(self.stream.device)["name"], self.rate)
        except Exception as exc:
            log.warning("3D sound off: no output device (%s). Pick one with --speaker / VRSIM_SPEAKER.", exc)

    def sound(self, name: str) -> np.ndarray:
        """``assets/sounds/<name>.wav`` (or ``VRSIM_<NAME>_WAV``) when there is one, else the built-in sound."""
        if name not in self._sounds:
            path = Path(os.environ.get(f"VRSIM_{name.upper()}_WAV") or SOUNDS_DIR / f"{name}.wav")
            samples = None
            if path.is_file():
                try:
                    samples = load_wav(path, self.rate)
                    log.info("3D sound '%s': %s (%.2f s)", name, path, len(samples) / self.rate)
                except Exception as exc:
                    log.warning("3D sound '%s': %s unreadable (%s), using the built-in one", name, path, exc)
            if samples is None:
                samples = {"ding": ding}[name](self.rate)
            self._sounds[name] = samples
        return self._sounds[name]

    def set_listener(self, pose) -> None:
        """Head pose in the world: x, y, z, qx, qy, qz, qw."""
        if pose is None or len(pose) != 7:
            return
        v = np.asarray(pose, dtype=np.float64)
        if not np.all(np.isfinite(v)):
            return
        q = v[3:] / (np.linalg.norm(v[3:]) or 1.0)
        with self._lock:
            self._head_p, self._head_q, self._head_time = v[:3], q, time.monotonic()

    def play(self, name: str, position, gain: float = 1.0) -> None:
        if not self.enabled:
            return
        voice = _Voice(self.sound(name), position, gain)
        with self._lock:
            self._voices.append(voice)

    def _callback(self, outdata, frames, time_info, status) -> None:
        outdata.fill(0.0)
        with self._lock:
            voices = list(self._voices)
            fresh = time.monotonic() - self._head_time < LISTENER_STALE_S
            head_p, head_q = (self._head_p, self._head_q) if fresh else (None, None)
        if not voices:
            return
        alive = []
        for v in voices:
            try:
                block, more = v.render(frames, head_p, head_q, self.rate)
                outdata += block
                if more:
                    alive.append(v)
            except Exception:
                log.exception("3D sound: voice dropped")
        np.clip(outdata, -1.0, 1.0, out=outdata)
        with self._lock:
            self._voices = [v for v in self._voices if v in alive or v not in voices]

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


if __name__ == "__main__":
    # Listener at the origin facing +Y (Z up): a ding ahead, right, behind, left.
    logging.basicConfig(level=logging.INFO)
    audio = SpatialAudio()
    head = [0, 0, 0, *(np.sin(np.pi / 4), 0, 0, np.cos(np.pi / 4))]  # view -z -> world +y, view +y -> world +z
    audio.set_listener(head)
    for label, pos in (("ahead", (0, 1, 0)), ("right", (1, 0, 0)), ("behind", (0, -1, 0)), ("left", (-1, 0, 0))):
        print(label)
        audio.play("ding", pos)
        for _ in range(16):
            time.sleep(0.1)
            audio.set_listener(head)
    time.sleep(0.5)
    audio.close()
