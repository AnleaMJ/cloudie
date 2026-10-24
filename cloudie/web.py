"""ASGI page for the Vercel Python runtime.

The repository-root ``app.py`` exposes ``app``. ``streamlit run cloudie/app.py``
remains the local page. A missing checkpoint returns before PyTorch is
imported. The public Infrared-1 JPEG is fetched when the form asks for it
and is not stored.
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

from cloudie.app import CONFIDENCE_NOTE, PAGE_LEAD, SCALE_NOTE, example_frames
from cloudie.config import load_config
from cloudie.imd_satellite import IMD_BUTTON, describe_frame, fetch_current_infrared
from cloudie.predict import Prediction, category_span, predict_images
from cloudie.schema import DEFAULT_EDGES_KT, DEFAULT_NAMES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "config/config.yaml"
DEFAULT_CHECKPOINT = "artifacts/checkpoint.pt"
NO_FRAME = "Add an infrared frame, then predict intensity."
RUNTIME_BANNER = (
    "This deployment cannot load the model runtime. "
    "There is no model output. This is not a forecast."
)
_MAX_UPLOAD = 20 * 1024 * 1024
_PREVIEW_EDGE = 720


def create_app() -> FastAPI:
    """Build the ASGI app. Importing this module does not start a server."""
    app = FastAPI(title="Cloudie", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return HTMLResponse(_document())

    @app.post("/run", response_class=HTMLResponse)
    async def run(
        action: str = Form(...),
        config: str = Form(DEFAULT_CONFIG),
        checkpoint: str = Form(DEFAULT_CHECKPOINT),
        previous: str = Form(""),
        example: str = Form("(none)"),
        frames: list[UploadFile] | None = File(default=None),
    ) -> HTMLResponse:
        page = await _run(
            action=action,
            config=config,
            checkpoint=checkpoint,
            previous=previous,
            example=example,
            frames=frames,
        )
        return HTMLResponse(page)

    return app


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
    return np.load(matches[0])


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


async def _run(
    *,
    action: str,
    config: str,
    checkpoint: str,
    previous: str,
    example: str,
    frames: list[UploadFile] | None,
) -> str:
    fields = _fields(config, checkpoint, previous, example)
    try:
        config_path = _resolve(config or DEFAULT_CONFIG)
        checkpoint_path = _resolve(checkpoint or DEFAULT_CHECKPOINT)
        previous_kt = _previous_kt(previous)
        loaded = load_config(config_path)
        if action == "imd":
            frame = fetch_current_infrared()
            prediction = _predict(loaded, checkpoint_path, [frame.image], previous_kt)
            return _document(
                fields,
                _imd_block(frame) + _prediction_block(prediction, loaded),
            )
        images = await _uploaded_images(frames)
        if not images:
            chosen = _example_image(example)
            images = [] if chosen is None else [chosen]
        if not images:
            return _document(fields, _message(NO_FRAME))
        prediction = _predict(loaded, checkpoint_path, images, previous_kt)
        return _document(fields, _prediction_block(prediction, loaded))
    except ImportError:
        return _document(fields, _message(RUNTIME_BANNER))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return _document(fields, _message(str(exc)))


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


def _message(text: str) -> str:
    return f'<p class="banner">{html.escape(text)}</p>'


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
    if not prediction.show_number or prediction.wind_kt is None:
        return _message(prediction.banner)
    edges, names = _edges_names(config)
    percent = (
        "—"
        if prediction.confidence is None
        else f"{prediction.confidence * 100:.0f}%"
    )
    label = prediction.category_label or "—"
    heading = f"{prediction.wind_kt:.1f} kt · {label} · {percent}"
    parts = [
        f"<h2>{html.escape(heading)}</h2>",
        "<dl class=\"metrics\">",
        _metric("Wind speed", f"{prediction.wind_kt:.1f} kt"),
        _metric("Intensity category", label),
        _metric("Confidence", percent),
        "</dl>",
        f"<p>{html.escape(label)} covers "
        f"{html.escape(category_span(prediction.project_class, edges, names))}.</p>",
        f"<p>{html.escape(CONFIDENCE_NOTE)}</p>",
    ]
    if prediction.confidence is not None:
        width = min(100.0, max(0.0, float(prediction.confidence) * 100))
        parts.append(f'<div class="meter" style="width:{width:.1f}%"></div>')
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
    if prediction.is_synthetic:
        parts.append(f"<p>{html.escape(prediction.banner)}</p>")
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


def _document(fields: dict[str, str] | None = None, result: str = "") -> str:
    current = fields or _fields(DEFAULT_CONFIG, DEFAULT_CHECKPOINT, "", "(none)")
    examples = example_frames(ROOT)
    options = ["(none)", *[path.name for path in examples]]
    selected = current["example"]
    option_html = "".join(
        "<option value=\"{value}\"{picked}>{label}</option>".format(
            value=html.escape(name, quote=True),
            picked=" selected" if name == selected else "",
            label=html.escape(name),
        )
        for name in options
    )
    example_field = ""
    if examples:
        example_field = (
            '<label>Example infrared frame'
            f'<select name="example">{option_html}</select>'
            "</label>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cloudie</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 46rem; color: #1c1915; background: #f7f3ea; }}
h1 {{ font-size: 2rem; margin-bottom: 0.2rem; }}
form, .score, figure {{ background: #fffdf8; border: 1px solid #e4dccb; padding: 1rem 1.2rem; }}
label {{ display: block; margin: 0.7rem 0; }}
input, select {{ display: block; width: 100%; margin-top: 0.25rem; font: inherit; }}
button {{ font: inherit; margin: 0.4rem 0.6rem 0.4rem 0; padding: 0.45rem 0.8rem; }}
.banner {{ background: #fff1e8; border: 1px solid #e7b8a0; padding: 0.8rem 1rem; }}
.metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; }}
dt {{ font-size: 0.85rem; }}
dd {{ margin: 0; font-size: 1.25rem; }}
.meter, .bar {{ display: inline-block; height: 0.7rem; background: #c4552a; }}
.meter {{ margin: 0.4rem 0 1rem; }}
.shares {{ list-style: none; padding: 0; }}
.shares li {{ display: grid; grid-template-columns: 16rem 1fr 3rem; gap: 0.5rem; align-items: center; }}
img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>Cloudie</h1>
<p>{html.escape(PAGE_LEAD)}</p>
<p>{html.escape(SCALE_NOTE)}</p>
{result}
<form method="post" action="/run" enctype="multipart/form-data">
<label>Config<input name="config" value="{html.escape(current["config"], quote=True)}"></label>
<label>Trained weights<input name="checkpoint" value="{html.escape(current["checkpoint"], quote=True)}"></label>
<label>Last advisory wind (kt)<input name="previous" value="{html.escape(current["previous"], quote=True)}"></label>
<p>Optional. Used only to flag a jump or a category change.</p>
<button name="action" value="imd" type="submit">{html.escape(IMD_BUTTON)}</button>
<p>Latest Infrared-1 image from the IMD satellite page.</p>
{example_field}
<label>Infrared image<input name="frames" type="file" accept=".npy,.png,.jpg,.jpeg,.tif,.tiff" multiple></label>
<button name="action" value="score" type="submit">Predict intensity</button>
</form>
</body>
</html>
"""
