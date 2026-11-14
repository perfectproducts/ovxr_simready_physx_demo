"""Runtime configuration. Values come from defaults, then environment variables, then CLI flags."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _wit_token() -> str:
    """WIT_AI_TOKEN env var, else CLIENT_ACCESS_TOKEN (or SERVER_ACCESS_TOKEN) from wit_ai_api.txt."""
    if os.environ.get("WIT_AI_TOKEN"):
        return os.environ["WIT_AI_TOKEN"]
    path = PROJECT_ROOT / "wit_ai_api.txt"
    if not path.exists():
        return ""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip().upper()] = value.strip().strip('"')
    return values.get("CLIENT_ACCESS_TOKEN") or values.get("SERVER_ACCESS_TOKEN", "")


def _load_hf_token() -> None:
    """Export the token in huggingface_api.txt (bare, or HF_TOKEN=...) as HF_TOKEN, which
    huggingface_hub reads on every request, unless HF_TOKEN is already set. It unlocks gated
    repos such as nvidia/simready-catalog and higher rate limits."""
    path = PROJECT_ROOT / "huggingface_api.txt"
    if os.environ.get("HF_TOKEN") or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.partition("=")[2] if "=" in line else line
        token = token.strip().strip('"')
        if token:
            os.environ["HF_TOKEN"] = token
            return


_load_hf_token()


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class TableConfig:
    """A static table in front of the user. Shared by the rendered scene and the physics world."""

    center: tuple[float, float, float] = (0.0, 0.9, 0.0)  # floor-level center; +Y is "forward" at start
    size: tuple[float, float] = (1.4, 0.8)  # x, y extent [m]
    height: float = 0.75
    top_thickness: float = 0.04


def _environment_usd() -> Path | None:
    """VRSIM_ENVIRONMENT (a .usd/.usdz path, or "none" for the generated floor and table), else
    the bundled Studierstube scene when present."""
    raw = os.environ.get("VRSIM_ENVIRONMENT")
    if raw is not None:
        return None if raw.strip().lower() in ("", "none") else Path(raw)
    default = PROJECT_ROOT / "assets" / "Studierstube.usdz"
    return default if default.exists() else None


@dataclass
class Decal:
    """A textured patch (PNG, alpha cut out) on the authored scene, in its coordinates (before
    ``EnvironmentConfig.offset``). Flat, or wrapped around a vertical cylinder of ``radius`` whose
    axis passes through ``center``."""

    image: Path
    center: tuple[float, float, float]
    size: tuple[float, float]  # width (arc length when wrapped), height [m]
    facing: float = -90.0  # direction the patch faces [deg from +X toward +Y]; -90 faces the user
    radius: float = 0.0


def _studierstube_decals() -> tuple[Decal, ...]:
    """OpenUSD logo beside the "OpenUSD" text on the laptop screen; NVIDIA logo around the mug,
    turned toward the standing spot. Sources: assets/logos/SOURCES.txt."""
    logos = PROJECT_ROOT / "assets" / "logos"
    return (Decal(logos / "openusd.png", (0.19, 0.4625, 1.075), (0.12, 0.12)),
            Decal(logos / "nvidia_white_text.png", (0.43, 0.41, 0.9), (0.09, 0.072), facing=-113.0, radius=0.0555))


@dataclass
class ToasterConfig:
    """The dbeyond smart toaster (assets/dbeyond) on the Studierstube desk, run by its FMUs.

    It stands at the desk's front left, in front of the books, lever and dial facing the user
    (toaster +X points toward -Y). The toast flies along toaster +Y and lands flat about 0.5 m
    further, on the free desk in front of the laptop; the yaw is a few degrees past -90 so it
    clears the laptop base. VRSIM_TOASTER=none leaves it out."""

    enabled: bool = field(default_factory=lambda: os.environ.get("VRSIM_TOASTER", "").strip().lower() not in ("none", "0", "off"))
    position: tuple[float, float, float] = (-0.62, 0.64, 0.783)  # toaster origin (bottom centre) [m]
    yaw: float = -97.0  # [deg] about +Z
    # FMU monitor panel (vrsim/toaster_monitor.py): centre in the toaster frame, on the side away from
    # the toast's flight; it stands on the desk and turns to the head. Left menu button hides it.
    monitor_offset: tuple[float, float] = field(default_factory=lambda: tuple(
        float(v) for v in (os.environ.get("VRSIM_MONITOR_OFFSET") or "0,-0.34").split(",")))  # [m] toaster x, y[, surface h]
    # Size factor of the FMU monitor panel (0.40 x 0.30 m at 1). VRSIM_MONITOR_SCALE sets it.
    monitor_scale: float = field(default_factory=lambda: float(os.environ.get("VRSIM_MONITOR_SCALE", "1") or 1))
    # Real seconds a toasting cycle takes, whatever the dial says (60-200 s of model time): while
    # the heater is on the FMUs run that much faster. 0 = real time. VRSIM_TOAST_SECONDS sets it.
    toast_seconds: float = field(default_factory=lambda: float(os.environ.get("VRSIM_TOAST_SECONDS", "0") or 0))
    # How high the ejected toast rises above the slot [m]; it lands on the same spot. 0 = the
    # ToastFlight FMU's authored launch (about 0.46 m). VRSIM_EJECT_HEIGHT sets it.
    eject_height: float = field(default_factory=lambda: float(os.environ.get("VRSIM_EJECT_HEIGHT", "0") or 0))
    # After the toast lands: a fresh slice in the slot and the toaster reset, ready for the next
    # cycle (the Newton app also leaves the toasted slice on the desk). VRSIM_NEW_TOAST=1 sets it.
    new_toast_after_eject: bool = field(default_factory=lambda: os.environ.get("VRSIM_NEW_TOAST", "").strip().lower()
                                        in ("1", "true", "on", "yes"))


@dataclass
class EnvironmentConfig:
    """An authored scene (Z-up, metres) replacing the generated floor and table.

    The user starts at the tracking origin facing +Y; ``offset`` moves the scene so its standing
    spot lands there. Defaults fit Studierstube: deck surface (z 0.04) to floor level, the desk's
    open side 0.45 m ahead. The scene has no physics, so its meshes get static convex colliders.
    """

    usd: Path | None = field(default_factory=_environment_usd)
    offset: tuple[float, float, float] = (0.0, 0.6, -0.04)
    ground_z: float = -0.28  # the ground around it: Studierstube's foundation bottom after the offset
    # The desk as a box for physics that doesn't read the scene's colliders (Newton) and for rays.
    table: TableConfig = field(default_factory=lambda: TableConfig(
        center=(0.0, 0.93, 0.0), size=(2.45, 0.944), height=0.783, top_thickness=0.065))
    hidden_prims: tuple[str, ...] = ("Sun", "Area", "env_light", "Camera")  # its lights: ours are tuned for ovrtx
    # Top-level prims (fnmatch patterns) that cast no shadow in Minimal rendering: it has a single
    # sun and no fill light, so the pergola's shadow on the desk renders almost black.
    shadowless_prims: tuple[str, ...] = ("Pavilion_post*", "Side_lintel*", "Cross_lintel*", "Pergola_rafter*")
    decals: tuple[Decal, ...] = field(default_factory=_studierstube_decals)
    # Scene objects that become grabbable props: their top-level prims are hidden in the environment
    # and the sidecar spawns them as one rigid body each (with the decals on them) where they stood.
    desk_props: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"NVIDIA mug": ("Ceramic_mug", "Torus", "Coffee")})
    desk_prop_mass: float = 0.35  # [kg]
    toaster: ToasterConfig = field(default_factory=ToasterConfig)
    # Meshes (paths below the scene's root) given a plain material of this linear colour: the mug
    # and its handle (Torus) go black so the NVIDIA logo reads on it.
    prim_colors: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {"Ceramic_mug/Cylinder_004": (0.012, 0.012, 0.013), "Torus/Torus": (0.012, 0.012, 0.013)})


def _sky_hdri() -> Path | None:
    """VRSIM_SKY_HDRI (an equirectangular .hdr/.exr, or "none"), else the bundled Poly Haven sky."""
    raw = os.environ.get("VRSIM_SKY_HDRI")
    if raw is not None:
        return None if raw.strip().lower() in ("", "none") else Path(raw)
    default = PROJECT_ROOT / "assets" / "hdri" / "kloofendal_48d_partly_cloudy_puresky_4k.hdr"
    return default if default.exists() else None


@dataclass
class LightingConfig:
    """Sky dome and sun. ovrtx applies no auto-exposure, so intensities are tuned by eye."""

    hdri: Path | None = field(default_factory=_sky_hdri)  # lights the scene and is the backdrop
    sky_intensity: float | None = None  # None: 1000 with the HDRI, 60 for the plain-colour dome
    sun_intensity: float = 1000.0
    # Sun direction (elevation, azimuth from +X toward +Y) [deg]. With an HDRI, match its sun so
    # the crisp DistantLight shadow agrees with the sky.
    sun_elevation: float = 48.0
    sun_azimuth: float = -105.0  # behind the user (who faces +Y): Minimal mode has no fill light for faces in shadow
    dome_rotation: float = 0.0  # rotation of the HDRI about +Z [deg]

    # --- ovrtx Minimal rendering ignores dome lights: a faked sky stands in -------------------
    # Minimal mode lights only with the first DistantLight (the sun): no dome, no other lights, no
    # working ambient term, and emission only as constant colours. The sky is a sphere of faces
    # bound to constant-emissive materials: the tone-mapped HDRI quantized to a palette
    # (vrsim.scene._fake_sky).
    backdrop: Path | None = field(default_factory=lambda: next(
        iter(sorted((PROJECT_ROOT / "assets" / "hdri").glob("*_backdrop.jpg"))), None))
    backdrop_radius: float = 300.0  # [m], beyond the ground plane's corners
    backdrop_brightness: float = 0.5  # emissive scale of the palette colours
    backdrop_rings: int = 192  # sphere faces vertically (and the image is resampled to match)
    backdrop_segments: int = 384  # sphere faces around
    backdrop_colors: int = 128  # palette size: one constant-emissive material each
    backdrop_sun_u: float = 0.595  # horizontal image position of the HDRI's sun, aligned to sun_azimuth


@dataclass
class Config:
    # --- Hugging Face / SimReady ------------------------------------------------------------
    # Optional index repo listing other repos (SIMREADY_CATALOG_REPO); "nvidia/simready-catalog"
    # doesn't exist on the Hub (404 even when signed in), so none by default.
    catalog_repo: str = field(default_factory=lambda: os.environ.get("SIMREADY_CATALOG_REPO", ""))
    # Repos in the SimReady package layout, crawled directly. nvidia/simready-assets is NVIDIA's
    # own public library (~2.7k packages); the others are partner repos.
    partner_repos: list[str] = field(
        default_factory=lambda: _env_list(
            "SIMREADY_PARTNER_REPOS",
            ["nvidia/simready-assets", "ALLSIDES/simready-scanned-objects", "Vertiv/simready-assets"],
        )
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("VRSIM_CACHE", PROJECT_ROOT / ".cache"))
    )
    min_match_score: float = 55.0
    # SimReady Central search (NVIDIA USD Search API on a HF Space, no token): semantic search over
    # more datasets than the crawl. Tried first; the local index is the fallback. "none" disables.
    central_url: str = field(
        default_factory=lambda: os.environ.get("SIMREADY_CENTRAL_URL", "https://nvidia-simready-central.hf.space")
    )
    central_timeout: float = 5.0
    central_limit: int = 120  # hits per browse query

    # --- Voice (Wit.ai, the service behind Meta Voice SDK) ---------------------------------
    wit_token: str = field(default_factory=_wit_token)
    wit_api_version: str = "20240304"
    mic_device: str | int | None = field(default_factory=lambda: os.environ.get("VRSIM_MIC") or None)
    # Output for 3D sounds (the toaster's ding); None = the system default, the headset over Link
    speaker_device: str | int | None = field(default_factory=lambda: os.environ.get("VRSIM_SPEAKER") or None)
    sample_rate: int = 16000
    max_utterance_s: float = 8.0

    # --- Physics ---------------------------------------------------------------------------
    device: str = "cuda:0"
    physics_hz: float = 72.0  # Quest 3 default refresh; physics advances in fixed frames of 1/physics_hz
    substeps: int = 8
    solver_iterations: int = 8
    friction: float = 0.8
    max_props: int = 40
    throw_scale: float = 1.3
    grab_radius: float = 0.12  # extra reach beyond an object's bounding sphere [m]
    ray_max_distance: float = 8.0
    grab_max_linear_speed: float = 12.0  # [m/s] cap so "force pull" grabs fly, not teleport
    grab_max_angular_speed: float = 30.0  # [rad/s]
    table: TableConfig = field(default_factory=TableConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)

    # --- XR --------------------------------------------------------------------------------
    foveation: bool = True
    application_name: str = "vrsim.simready.physics"

    def __post_init__(self) -> None:
        if self.environment.usd is not None:  # physics, rays and the HUD use the scene's desk
            self.table = self.environment.table

    @property
    def generated_dir(self) -> Path:
        path = self.cache_dir / "generated"
        path.mkdir(parents=True, exist_ok=True)
        return path
