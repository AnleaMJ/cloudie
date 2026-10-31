"""ASGI page for the Vercel Python runtime.

The repository-root ``app.py`` exposes ``app``. ``streamlit run cloudie/app.py``
remains the local page. This module does not import torch. A missing
checkpoint returns before the model runtime is loaded. The public
Infrared-1 JPEG is fetched only when the satellite button is used, and it
is not stored.

GET ``/`` and GET ``/run`` render the form. Enter submits predict, not the
satellite fetch. Config and weight paths stay off the page unless ``debug=1``.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from cloudie.app import (
    CONFIDENCE_NOTE,
    GROUPING_LABEL,
    NO_TRAINED_MODEL_BANNER,
    PAGE_LEAD,
    SCALE_NOTE,
    example_frames,
    public_banner,
)
from cloudie.config import checkpoint_path, load_config
from cloudie.imd_satellite import IMD_BUTTON, describe_frame, fetch_current_infrared
from cloudie.predict import Prediction, category_span, predict_images
from cloudie.schema import DEFAULT_EDGES_KT, DEFAULT_NAMES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "config/config.yaml"
DEFAULT_CHECKPOINT = checkpoint_path(load_config(ROOT / DEFAULT_CONFIG)).as_posix()
NO_FRAME = "Add an infrared frame, then predict intensity."
RUNTIME_BANNER = (
    "This deployment cannot load the model runtime. "
    "There is no model output. This is not a forecast."
)
REFUSAL = "not a tropical cyclone IR image"
_MAX_UPLOAD = 20 * 1024 * 1024
_PREVIEW_EDGE = 720


def create_app() -> FastAPI:
    """Build the ASGI app. Importing this module does not start a server."""
    app = FastAPI(title="Cloudie", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    @app.get("/run", response_class=HTMLResponse)
    def home(debug: str = "") -> HTMLResponse:
        return HTMLResponse(_document(debug=_flag(debug)))

    @app.post("/run", response_class=HTMLResponse)
    async def run(
        action: str = Form("score"),
        config: str = Form(""),
        checkpoint: str = Form(""),
        previous: str = Form(""),
        example: str = Form("(none)"),
        debug: str = Form(""),
        frames: list[UploadFile] | None = File(default=None),
    ) -> HTMLResponse:
        page = await _run(
            action=action,
            config=config,
            checkpoint=checkpoint,
            previous=previous,
            example=example,
            debug=_flag(debug),
            frames=frames,
        )
        return HTMLResponse(page)

    return app


def _flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve(text: str) -> Path:
    """Resolve a form path inside the repository. Rejects paths that leave it."""
    raw = Path(text.strip())
    candidate = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError("Config and checkpoint paths stay inside this project.")
    return candidate


def _decode_frame(name: str, data: bytes) -> np.ndarray:
    if len(data) > _MAX_UPLOAD:
        raise ValueError("An uploaded frame is larger than expected.")
    try:
        if name.lower().endswith(".npy"):
            return np.asarray(np.load(io.BytesIO(data), allow_pickle=False))
        picture = Image.open(io.BytesIO(data))
        return np.asarray(picture, dtype=np.float32)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError("Could not read an infrared frame.") from exc


async def _uploaded_images(frames: list[UploadFile] | None) -> list[np.ndarray]:
    named = [item for item in frames or [] if item.filename]
    named.sort(key=lambda item: item.filename or "")
    images: list[np.ndarray] = []
    for item in named:
        images.append(_decode_frame(item.filename or "frame", await item.read()))
    return images


def _example_image(name: str) -> np.ndarray | None:
    if name.strip() in {"", "(none)"}:
        return None
    matches = [path for path in example_frames(ROOT) if path.name == name]
    if not matches:
        raise ValueError("That example frame is not available.")
    return np.load(matches[0], allow_pickle=False)


def _previous_kt(text: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError("Last advisory wind must be a number of knots.") from exc


def _jpeg_data_uri(preview: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(preview, dtype=np.uint8), mode="RGB")
    image.thumbnail((_PREVIEW_EDGE, _PREVIEW_EDGE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _href(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return html.escape(url, quote=True)
    return None


def _action(value: str) -> str:
    """Missing or blank actions predict. Only the satellite button fetches IMD."""
    chosen = value.strip().lower()
    if chosen == "imd":
        return "imd"
    return "score"


async def _run(
    *,
    action: str,
    config: str,
    checkpoint: str,
    previous: str,
    example: str,
    debug: bool,
    frames: list[UploadFile] | None,
) -> str:
    chosen = _action(action)
    config_text = config.strip() or DEFAULT_CONFIG
    checkpoint_text = checkpoint.strip() or DEFAULT_CHECKPOINT
    fields = _fields(config_text, checkpoint_text, previous, example)
    try:
        config_path = _resolve(config_text)
        weights = _resolve(checkpoint_text)
        previous_kt = _previous_kt(previous)
        loaded = load_config(config_path)
        if chosen == "imd":
            frame = fetch_current_infrared()
            prediction = _predict(loaded, weights, [frame.image], previous_kt)
            return _document(
                fields,
                _imd_block(frame) + _prediction_block(prediction, loaded),
                debug=debug,
                interval_status=_interval_status(prediction, loaded),
            )
        images = await _uploaded_images(frames)
        if not images:
            chosen_frame = _example_image(example)
            images = [] if chosen_frame is None else [chosen_frame]
        if not images:
            return _document(fields, _message(NO_FRAME), debug=debug)
        prediction = _predict(loaded, weights, images, previous_kt)
        return _document(
            fields,
            _prediction_block(prediction, loaded),
            debug=debug,
            interval_status=_interval_status(prediction, loaded),
        )
    except ImportError:
        return _document(fields, _message(RUNTIME_BANNER), debug=debug)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return _document(fields, _message(str(exc)), debug=debug)


def _predict(
    config: dict,
    checkpoint: Path,
    images: list[np.ndarray],
    previous_kt: float | None,
) -> Prediction:
    return predict_images(
        config,
        checkpoint,
        images,
        previous_advisory_kt=previous_kt,
    )


def _fields(config: str, checkpoint: str, previous: str, example: str) -> dict[str, str]:
    return {
        "config": config or DEFAULT_CONFIG,
        "checkpoint": checkpoint or DEFAULT_CHECKPOINT,
        "previous": previous,
        "example": example or "(none)",
    }


def _interval_status(prediction: Prediction, config: dict) -> str:
    if prediction.confidence_status:
        return str(prediction.confidence_status)
    label = (config.get("predict") or {}).get("confidence", {}).get("uncalibrated_label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return "uncalibrated"


def _message(text: str) -> str:
    shown = public_banner(text)
    kind = "refusal" if REFUSAL in shown else "banner"
    return f'<p class="{kind}">{html.escape(shown)}</p>'


def _imd_block(frame) -> str:
    caption = html.escape(describe_frame(frame))
    href = _href(frame.source_url)
    link = f'<a href="{href}">{href}</a>' if href else html.escape(frame.source_url)
    return (
        '<figure class="scene">'
        f'<img src="{_jpeg_data_uri(frame.preview)}" alt="{caption}">'
        f"<figcaption>{caption}<br>{link}</figcaption>"
        "</figure>"
    )


def _prediction_block(prediction: Prediction, config: dict) -> str:
    shown = public_banner(prediction.banner)
    if not prediction.show_number or prediction.wind_kt is None:
        return _message(shown)
    edges, names = _edges_names(config)
    percent = (
        "—"
        if prediction.confidence is None
        else f"{prediction.confidence * 100:.0f}%"
    )
    label = prediction.category_label or "—"
    status = _interval_status(prediction, config)
    heading = f"{prediction.wind_kt:.1f} kt · {label} · {percent}"
    note = prediction.grouping_note or (
        "Intensity category is this project's grouping. It is not an official IMD class."
    )
    parts = [
        f"<h2>{html.escape(heading)}</h2>",
        '<dl class="metrics">',
        _metric("Wind speed", f"{prediction.wind_kt:.1f} kt"),
    ]
    if prediction.wind_kmh is not None:
        parts.append(_metric("Wind speed (km/h)", f"{prediction.wind_kmh:.1f}"))
    parts.extend(
        [
            _metric(GROUPING_LABEL, label),
            _metric("Confidence", f"{percent} ({status})"),
            "</dl>",
            f'<p class="grouping">{html.escape(note)}</p>',
            f"<p>{html.escape(label)} covers "
            f"{html.escape(category_span(prediction.project_class, edges, names))}. "
            "This is the project's own grouping, not an official IMD class.</p>",
            f"<p>{html.escape(CONFIDENCE_NOTE)}</p>",
        ]
    )
    if prediction.operational_disclaimer:
        parts.append(f"<p>{html.escape(prediction.operational_disclaimer)}</p>")
    if prediction.class_probabilities:
        items = []
        for name, share in prediction.class_probabilities:
            share_width = min(100.0, max(0.0, float(share) * 100))
            items.append(
                "<li>"
                f"<span>{html.escape(name)}</span>"
                f'<span class="bar" style="width:{share_width:.1f}%"></span>'
                f"<span>{share * 100:.0f}%</span>"
                "</li>"
            )
        parts.append(f'<ul class="shares">{"".join(items)}</ul>')
    if prediction.is_synthetic and shown:
        parts.append(f'<p class="banner">{html.escape(shown)}</p>')
    if prediction.review:
        notes = "".join(f"<li>{html.escape(item)}</li>" for item in prediction.review)
        parts.append(f"<details><summary>Review notes</summary><ul>{notes}</ul></details>")
    return '<section class="score">' + "".join(parts) + "</section>"


def _metric(name: str, value: str) -> str:
    return f"<div><dt>{html.escape(name)}</dt><dd>{html.escape(value)}</dd></div>"


def _edges_names(config: dict) -> tuple[tuple[float, ...], tuple[str, ...]]:
    edges = tuple(float(edge) for edge in config["data"]["imd5_edges_kt"])
    names = tuple(str(name) for name in config["data"]["imd5_names"])
    if not edges:
        return DEFAULT_EDGES_KT, DEFAULT_NAMES
    return edges, names


def _document(
    fields: dict[str, str] | None = None,
    result: str = "",
    *,
    debug: bool = False,
    interval_status: str = "uncalibrated",
) -> str:
    current = fields or _fields(DEFAULT_CONFIG, DEFAULT_CHECKPOINT, "", "(none)")
    examples = example_frames(ROOT)
    options = ["(none)", *[path.name for path in examples]]
    selected = current["example"]
    option_html = "".join(
        '<option value="{value}"{picked}>{label}</option>'.format(
            value=html.escape(name, quote=True),
            picked=" selected" if name == selected else "",
            label=html.escape(_example_label(name)),
        )
        for name in options
    )
    example_field = ""
    if examples:
        example_field = (
            "<label>Sample frame"
            f'<select name="example">{option_html}</select>'
            "</label>"
            '<p class="hint">The bundled frame is synthetic. It is not a satellite image.</p>'
        )
    path_fields = ""
    if debug:
        path_fields = (
            '<input type="hidden" name="debug" value="1">'
            "<label>Config"
            f'<input name="config" value="{html.escape(current["config"], quote=True)}">'
            "</label>"
            "<label>Trained weights"
            f'<input name="checkpoint" value="{html.escape(current["checkpoint"], quote=True)}">'
            "</label>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cloudie</title>
<style>
:root {{
  --ink: #1c1915;
  --paper: #efe6d6;
  --card: #f7f1e6;
  --line: #d9cbb6;
  --banner: #4a2c22;
  --banner-bg: #f0d2c4;
  --refusal: #6b1d1d;
  --refusal-bg: #f6d6d2;
  --accent: #8d3d24;
  --slot: #6d6458;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(1200px 400px at 10% -10%, #f6e7cf, transparent 50%),
    var(--paper);
  font-family: "Iowan Old Style", Palatino, Georgia, serif;
}}
main {{
  max-width: 42rem;
  margin: 0 auto;
  padding: 2.2rem 1.2rem 3rem;
}}
h1 {{
  font-size: 2.4rem;
  line-height: 1;
  margin: 0 0 0.4rem;
  letter-spacing: -0.03em;
}}
.lead, .hint, .grouping {{ color: #3f3830; }}
.permanent {{
  background: var(--banner-bg);
  color: var(--banner);
  border-left: 4px solid var(--accent);
  padding: 0.85rem 1rem;
  margin: 1.2rem 0;
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}}
.interval {{
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 0.2rem 1rem;
  align-items: end;
  border: 1px dashed var(--slot);
  background: rgba(255, 252, 247, 0.7);
  padding: 0.9rem 1rem 1rem;
  margin: 0 0 1rem;
}}
.interval h2 {{
  margin: 0;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-family: "Avenir Next", "Segoe UI", sans-serif;
  font-weight: 600;
}}
.interval .value {{
  margin: 0;
  font-size: 2rem;
  line-height: 1;
}}
.interval .note {{
  grid-column: 1 / -1;
  margin: 0.35rem 0 0;
  color: var(--slot);
  font-family: "Avenir Next", "Segoe UI", sans-serif;
  font-size: 0.92rem;
}}
form, .score, figure {{
  background: var(--card);
  border: 1px solid var(--line);
  padding: 1rem 1.15rem 1.15rem;
  margin: 0 0 1rem;
}}
label {{
  display: block;
  margin: 0.8rem 0 0.3rem;
  font-family: "Avenir Next", "Segoe UI", sans-serif;
  font-size: 0.92rem;
}}
input, select {{
  display: block;
  width: 100%;
  margin-top: 0.3rem;
  padding: 0.45rem 0.5rem;
  font: inherit;
  color: var(--ink);
  background: #fffdf8;
  border: 1px solid var(--line);
}}
.actions {{ display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 1rem; }}
button {{
  font-family: "Avenir Next", "Segoe UI", sans-serif;
  font-size: 1rem;
  padding: 0.55rem 0.9rem;
  cursor: pointer;
}}
button.primary {{
  background: var(--accent);
  color: #fffaf6;
  border: 1px solid var(--accent);
}}
button.secondary {{
  background: transparent;
  color: var(--ink);
  border: 1px solid var(--line);
}}
.banner, .refusal {{ padding: 0.8rem 1rem; }}
.banner {{ background: #fff6ee; border: 1px solid #e7c2ae; }}
.refusal {{
  background: var(--refusal-bg);
  color: var(--refusal);
  border: 1px solid #e3aaa4;
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(8rem, 1fr));
  gap: 0.8rem;
  margin: 0.6rem 0 0.2rem;
}}
dt {{
  font-size: 0.78rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}}
dd {{ margin: 0.15rem 0 0; font-size: 1.25rem; }}
.bar {{ display: inline-block; height: 0.55rem; background: var(--accent); }}
.shares {{ list-style: none; padding: 0; }}
.shares li {{
  display: grid;
  grid-template-columns: minmax(8rem, 14rem) 1fr 3rem;
  gap: 0.5rem;
  align-items: center;
  font-family: "Avenir Next", "Segoe UI", sans-serif;
  font-size: 0.92rem;
}}
img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<main>
<h1>Cloudie</h1>
<p class="lead">{html.escape(PAGE_LEAD)}</p>
<p class="permanent">{html.escape(NO_TRAINED_MODEL_BANNER)}</p>
<p>{html.escape(SCALE_NOTE)}</p>
<section class="interval" aria-label="Wind interval">
<h2>Wind interval</h2>
<p class="value">—</p>
<p class="note">{html.escape(interval_status)}. No numeric range is shown.</p>
</section>
{result}
<form method="post" action="/run" enctype="multipart/form-data">
{path_fields}
<label>Last advisory wind (kt)
<input name="previous" value="{html.escape(current["previous"], quote=True)}" inputmode="decimal">
</label>
<p class="hint">Optional. Used only to flag a jump or a category change. Enter predicts. It does not fetch a satellite image.</p>
{example_field}
<label>Infrared image
<input name="frames" type="file" accept=".npy,.png,.jpg,.jpeg,.tif,.tiff" multiple>
</label>
<div class="actions">
<button class="primary" name="action" value="score" type="submit">Predict intensity</button>
<button class="secondary" name="action" value="imd" type="submit">{html.escape(IMD_BUTTON)}</button>
</div>
<p class="hint">Only the second button fetches the latest Infrared-1 image from the IMD satellite page.</p>
</form>
</main>
</body>
</html>
"""


def _example_label(name: str) -> str:
    if name == "(none)":
        return "(none)"
    if name == "synthetic_ir.npy":
        return "synthetic_ir.npy — bundled synthetic sample"
    return name
