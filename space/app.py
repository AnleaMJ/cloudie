"""Gradio page for the Cloudie Space.

Project code is imported from ``src`` only. The permanent banner is
``NO_TRAINED_MODEL_BANNER`` in ``src.inference``, the same constant the web
demo re-exports. This page is not a forecast or a warning system.

``RunCheckpoint`` loads ``paths.runs/<run_id>/`` or a Hub repo that mirrors
that run directory. A wind is shown for the synthetic software-check file,
flagged as such, and for a real run whose record passes the held-out storm
check. Visitors do not choose the file.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class SpaceLayout:
    """Locate the checkout or the synced Space bundle and put it on ``sys.path``."""

    @staticmethod
    def root() -> Path:
        here = Path(__file__).resolve().parent
        if (here / "src" / "inference.py").is_file():
            return here
        parent = here.parent
        if (parent / "src" / "inference.py").is_file():
            return parent
        raise FileNotFoundError(
            "src/inference.py is not beside this app. Run python space/sync_bundle.py."
        )

    @staticmethod
    def install() -> Path:
        root = SpaceLayout.root()
        entry = str(root)
        if entry not in sys.path:
            sys.path.insert(0, entry)
        return root


ROOT = SpaceLayout.install()

from src.config import load_config  # noqa: E402
from src.inference import (  # noqa: E402
    CONFIDENCE_NOTE,
    GROUPING_LABEL,
    NO_TRAINED_MODEL_BANNER,
    PAGE_LEAD,
    SCALE_NOTE,
    CoastalContext,
    RunCheckpoint,
)
from src.schema import wind_to_imd5  # noqa: E402


class SpaceConfig:
    """The NASA config. Smoke stays a local software check."""

    @staticmethod
    def load(root: Path | None = None) -> dict:
        base = root or SpaceLayout.root()
        return load_config(base / "config" / "config.yaml")

    @staticmethod
    def uncalibrated(config: dict) -> str:
        label = str(config["predict"]["confidence"]["uncalibrated_label"]).strip()
        if not label:
            raise ValueError("config predict.confidence.uncalibrated_label is empty")
        return label

class SpaceSamples:
    """Bundled frames. The synthetic sample is not a satellite image."""

    HINT = "The bundled frame is synthetic. It is not a satellite image."

    @staticmethod
    def label(name: str) -> str:
        if name == "synthetic_ir.npy":
            return "synthetic_ir.npy — bundled synthetic sample"
        return name

    @staticmethod
    def choices(root: Path | None = None) -> list[tuple[str, str]]:
        base = root or SpaceLayout.root()
        rows = [("(none)", "(none)")]
        sample = base / "cloudie" / "samples" / "synthetic_ir.npy"
        if sample.is_file():
            rows.append((SpaceSamples.label(sample.name), sample.name))
        return rows

    @staticmethod
    def load(name: str, root: Path) -> np.ndarray | None:
        if name.strip() in {"", "(none)", "none"}:
            return None
        if name != Path(name).name or ".." in name:
            raise ValueError("That example frame is not available.")
        path = root / "cloudie" / "samples" / name
        if not path.is_file():
            raise ValueError("That example frame is not available.")
        return np.load(path, allow_pickle=False)


class SpaceFrame:
    """Turn a Gradio upload or the bundled sample into one array."""

    @staticmethod
    def path_of(upload) -> Path | None:
        if upload is None:
            return None
        if isinstance(upload, (str, Path)):
            text = str(upload).strip()
            return Path(text) if text else None
        if isinstance(upload, dict):
            raw = upload.get("path") or upload.get("name")
            return Path(str(raw)) if raw else None
        name = getattr(upload, "name", None)
        if name:
            return Path(str(name))
        return None

    @staticmethod
    def choose(image, npy_file, sample: str, root: Path) -> np.ndarray | None:
        """An upload replaces the bundled sample, as on the web demo."""
        npy_path = SpaceFrame.path_of(npy_file)
        if npy_path is not None:
            if npy_path.suffix.lower() != ".npy":
                raise ValueError("The extra frame must be a .npy file.")
            return np.asarray(np.load(npy_path, allow_pickle=False))
        if image is not None:
            return np.asarray(image)
        return SpaceSamples.load(sample or "(none)", root)


class SpacePreview:
    """A display image. The gate still sees the original array."""

    @staticmethod
    def rgb(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[-1] in (3, 4):
            scaled = SpacePreview._uint8(array[..., :3])
            return scaled
        field = array[..., 0] if array.ndim == 3 else array
        scaled = SpacePreview._uint8(field)
        return np.stack([scaled, scaled, scaled], axis=-1)

    @staticmethod
    def _uint8(array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float64)
        if values.ndim == 2:
            finite = values[np.isfinite(values)]
            scaled = SpacePreview._scale(values, finite)
            return scaled.astype(np.uint8)
        channels = [
            SpacePreview._scale(values[..., index], values[..., index][np.isfinite(values[..., index])])
            for index in range(values.shape[-1])
        ]
        return np.stack(channels, axis=-1).astype(np.uint8)

    @staticmethod
    def _scale(values: np.ndarray, finite: np.ndarray) -> np.ndarray:
        if finite.size == 0:
            return np.zeros(values.shape, dtype=np.float64)
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            high = low + 1.0
        return np.clip((values - low) / (high - low), 0.0, 1.0) * 255.0



class SpaceCoast:
    """Nearest configured coastal states. A position is required. Not a landfall forecast."""

    @staticmethod
    def idle(config: dict) -> str:
        note = str(config["predict"]["landfall_note"]).strip()
        return (
            "Nearest coastal states are listed only when latitude and longitude "
            f"are both provided. {note}."
        )

    @staticmethod
    def coordinate(value) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    @staticmethod
    def line(config: dict, latitude, longitude) -> str:
        note = str(config["predict"]["landfall_note"]).strip()
        try:
            lat = SpaceCoast.coordinate(latitude)
            lon = SpaceCoast.coordinate(longitude)
        except (TypeError, ValueError):
            return f"Latitude and longitude must be numbers. {note}."
        if lat is None and lon is None:
            return SpaceCoast.idle(config)
        try:
            names, landfall, reference = CoastalContext.describe(config, lat, lon, None)
        except ValueError as exc:
            return f"{exc}. {note}."
        if not names:
            return f"No coastal reference was returned. {note}."
        heading = str(config["predict"]["coastal_heading"]).strip()
        parts = [f"{heading}: {', '.join(names)}", landfall or note]
        if reference:
            parts.append(str(reference).strip())
        return "\n\n".join(part for part in parts if part)


@dataclass(frozen=True)
class SpaceView:
    """Text the page shows. ``show_wind`` is false whenever the knots stay hidden."""

    permanent_banner: str
    status: str
    wind: str
    category: str
    confidence: str
    interval: str
    interval_note: str
    grouping: str
    coastal: str
    show_wind: bool
    preview: np.ndarray | None = None

    def outputs(self):
        return (
            self.status,
            self.wind,
            self.category,
            self.confidence,
            self.interval,
            self.interval_note,
            self.grouping,
            self.coastal,
            self.preview,
        )


class SpacePage:
    """Score one frame with the project inference code and the shared banner."""

    ASK_FOR_FRAME = "Add an infrared frame, then predict intensity."

    @staticmethod
    def idle(config: dict | None = None) -> SpaceView:
        loaded = config or SpaceConfig.load()
        uncalibrated = SpaceConfig.uncalibrated(loaded)
        return SpaceView(
            permanent_banner=NO_TRAINED_MODEL_BANNER,
            status="",
            wind="—",
            category="—",
            confidence=f"— ({uncalibrated})",
            interval="—",
            interval_note=f"{uncalibrated}. No numeric range is shown.",
            grouping=str(loaded["data"]["grouping_note"]).strip(),
            coastal=SpaceCoast.idle(loaded),
            show_wind=False,
        )

    @staticmethod
    def _blank(config: dict, status: str, coastal: str, preview, *, show_wind: bool = False) -> SpaceView:
        idle = SpacePage.idle(config)
        return SpaceView(
            permanent_banner=NO_TRAINED_MODEL_BANNER,
            status=status,
            wind=idle.wind,
            category=idle.category,
            confidence=idle.confidence,
            interval=idle.interval,
            interval_note=idle.interval_note,
            grouping=idle.grouping,
            coastal=coastal,
            show_wind=show_wind,
            preview=preview,
        )

    LABELS = {
        "D_DD": "Depression / Deep Depression",
        "CS": "Cyclonic Storm",
        "SCS": "Severe Cyclonic Storm",
        "VSCS": "Very Severe Cyclonic Storm",
        "ESCS_SUCS": "Extremely Severe / Super Cyclonic Storm",
    }

    @staticmethod
    def _category(config: dict, estimate) -> str:
        if estimate.wind_kt is None:
            return "—"
        edges = estimate.edges or tuple(float(edge) for edge in config["data"]["imd5_edges_kt"])
        names = estimate.names or tuple(str(name) for name in config["data"]["imd5_names"])
        code = wind_to_imd5(float(estimate.wind_kt), edges, names)
        if code is None:
            return "Below depression"
        return SpacePage.LABELS.get(code, code)

    @staticmethod
    def _from_estimate(config: dict, estimate, coastal: str, preview) -> SpaceView:
        uncalibrated = SpaceConfig.uncalibrated(config)
        show = estimate.outcome == "ok" and estimate.wind_kt is not None
        status_bits = []
        if estimate.operational_disclaimer:
            status_bits.append(str(estimate.operational_disclaimer))
        if show and estimate.is_synthetic:
            if estimate.honesty:
                status_bits.append(str(estimate.honesty))
            status_bits.append("The number is a software check. It is not a forecast.")
        elif estimate.banner:
            status_bits.append(str(estimate.banner))
        grouping = str(estimate.grouping_note or config["data"]["grouping_note"]).strip()
        grouping = f"{grouping} This is the project's own grouping, not an official IMD class."
        if not show:
            return SpaceView(
                permanent_banner=NO_TRAINED_MODEL_BANNER,
                status="\n".join(bit for bit in status_bits if bit),
                wind="—",
                category="—",
                confidence=f"— ({uncalibrated})",
                interval="—",
                interval_note=f"{uncalibrated}. No numeric range is shown.",
                grouping=grouping,
                coastal=coastal,
                show_wind=False,
                preview=preview,
            )
        percent = "—" if estimate.confidence is None else f"{estimate.confidence * 100:.0f}%"
        label = estimate.confidence_status or uncalibrated
        wind = f"{estimate.wind_kt:.1f} kt"
        if estimate.wind_kmh is not None:
            wind = f"{wind} ({estimate.wind_kmh:.1f} km/h)"
        if label == uncalibrated:
            interval_note = f"{uncalibrated}. No numeric range is shown. {CONFIDENCE_NOTE}"
        else:
            interval_note = (
                f"{label}. {uncalibrated}. No numeric range is shown. {CONFIDENCE_NOTE}"
            )
        return SpaceView(
            permanent_banner=NO_TRAINED_MODEL_BANNER,
            status="\n".join(bit for bit in status_bits if bit),
            wind=wind,
            category=SpacePage._category(config, estimate),
            confidence=f"{percent} ({label})",
            interval="—",
            interval_note=interval_note,
            grouping=grouping,
            coastal=coastal,
            show_wind=True,
            preview=preview,
        )

    @staticmethod
    def view(
        image,
        npy_file,
        sample: str,
        latitude,
        longitude,
        *,
        config: dict | None = None,
        root: Path | None = None,
        checkpoint: Path | None | object = None,
        resolve_checkpoint: bool = False,
    ) -> SpaceView:
        """Ask ``src.inference.RunCheckpoint`` to gate the frame and load a run.

        ``resolve_checkpoint`` is what the button uses. Tests pass ``checkpoint``
        themselves. A missing file yields no wind. A colour frame yields the
        gate refusal and no wind.
        """
        loaded = config or SpaceConfig.load()
        base = root or SpaceLayout.root()
        coastal = SpaceCoast.line(loaded, latitude, longitude)
        try:
            frame = SpaceFrame.choose(image, npy_file, sample or "(none)", base)
        except (OSError, ValueError) as exc:
            return SpacePage._blank(loaded, str(exc), coastal, None)
        if frame is None:
            return SpacePage._blank(loaded, SpacePage.ASK_FOR_FRAME, coastal, None)
        preview = SpacePreview.rgb(frame)
        try:
            lat = SpaceCoast.coordinate(latitude)
            lon = SpaceCoast.coordinate(longitude)
        except (TypeError, ValueError):
            lat, lon = None, None
        if lat is None or lon is None:
            lat, lon = None, None
        if resolve_checkpoint:
            estimate = RunCheckpoint.estimate(
                loaded,
                [frame],
                latitude=lat,
                longitude=lon,
                root=base,
                resolve=True,
            )
        else:
            explicit = None if checkpoint is None else Path(checkpoint)
            estimate = RunCheckpoint.estimate(
                loaded,
                [frame],
                latitude=lat,
                longitude=lon,
                root=base,
                path=explicit,
                resolve=False,
            )
        return SpacePage._from_estimate(loaded, estimate, coastal, preview)

    @staticmethod
    def on_click(image, npy_file, sample, latitude, longitude):
        view = SpacePage.view(
            image,
            npy_file,
            sample,
            latitude,
            longitude,
            resolve_checkpoint=True,
        )
        return view.outputs()


class SpaceApp:
    """The Blocks page. ``demo`` is the object the Space runtime launches."""

    CSS = """
    .gradio-container, main, .contain { max-width: 920px !important; margin-left: auto !important; margin-right: auto !important; }
    #no-trained-model-banner {
      background: #f0d2c4;
      color: #4a2c22;
      border-left: 4px solid #8d3d24;
      padding: 0.85rem 1rem;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }
    """

    @staticmethod
    def build():
        import gradio as gr

        config = SpaceConfig.load()
        idle = SpacePage.idle(config)
        disclaimer = str(config["predict"]["operational_disclaimer"]).strip()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The parameters have been moved from the Blocks constructor",
                category=UserWarning,
            )
            with gr.Blocks(title="Cloudie", theme=gr.themes.Soft(), css=SpaceApp.CSS) as demo:
                gr.Markdown("# Cloudie")
                gr.Markdown(PAGE_LEAD)
                gr.Markdown(NO_TRAINED_MODEL_BANNER, elem_id="no-trained-model-banner")
                gr.Markdown(SCALE_NOTE)
                gr.Markdown(disclaimer)
                gr.Markdown(CONFIDENCE_NOTE)
                with gr.Row():
                    image = gr.Image(
                        type="numpy",
                        label="Infrared image",
                        sources=["upload", "clipboard"],
                    )
                    with gr.Column():
                        sample = gr.Radio(
                            choices=SpaceSamples.choices(),
                            value="(none)",
                            label="Bundled sample",
                            info=SpaceSamples.HINT,
                        )
                        latitude = gr.Textbox(
                            label="Latitude",
                            value="",
                            placeholder="optional",
                            info="Optional. Nearest coastal states need both coordinates.",
                        )
                        longitude = gr.Textbox(
                            label="Longitude",
                            value="",
                            placeholder="optional",
                            info=str(config["predict"]["landfall_note"]).strip(),
                        )
                        npy_file = gr.File(
                            label="Or a .npy infrared frame",
                            file_types=[".npy"],
                            type="filepath",
                        )
                        button = gr.Button("Predict intensity", variant="primary")
                status = gr.Markdown(idle.status)
                with gr.Row():
                    wind = gr.Textbox(label="Wind speed", value=idle.wind, interactive=False)
                    category = gr.Textbox(
                        label=GROUPING_LABEL,
                        value=idle.category,
                        interactive=False,
                    )
                    confidence = gr.Textbox(
                        label="Confidence (uncalibrated)",
                        value=idle.confidence,
                        interactive=False,
                    )
                    interval = gr.Textbox(
                        label="Wind interval (uncalibrated)",
                        value=idle.interval,
                        interactive=False,
                    )
                interval_note = gr.Markdown(idle.interval_note)
                grouping = gr.Markdown(idle.grouping)
                coastal = gr.Markdown(idle.coastal)
                preview = gr.Image(label="Frame passed to the gate", interactive=False)
                button.click(
                    SpacePage.on_click,
                    inputs=[image, npy_file, sample, latitude, longitude],
                    outputs=[
                        status,
                        wind,
                        category,
                        confidence,
                        interval,
                        interval_note,
                        grouping,
                        coastal,
                        preview,
                    ],
                    api_name="predict",
                )
        return demo


demo = SpaceApp.build()


if __name__ == "__main__":
    demo.launch()
