"""Streamlit demo for cyclone wind speed and intensity category.

Run with ``streamlit run cloudie/app.py``. The page reads a trained
checkpoint and reports knots, the intensity category of that wind, and
the share of model samples in that category. It does not write a score
into the project documentation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cloudie.imd_satellite import IMD_BUTTON, describe_frame, fetch_current_infrared
from cloudie.predict import Prediction, category_span

ROOT = Path(__file__).resolve().parents[1]
PAGE_LEAD = "Predict cyclone wind speed and intensity category from an infrared image."
NO_TRAINED_MODEL_BANNER = (
    "No trained model yet. This page cannot produce estimates. "
    "Not a forecast or warning system."
)
CONFIDENCE_NOTE = (
    "Confidence mixes the image gate, dropout spread, and a residual-quantile "
    "width when one exists. It is uncalibrated until a held-out storm check exists. "
    "It is not a forecast interval."
)
SCALE_NOTE = (
    "Reported winds stay between 0 and 200 kt: below depression, depression, "
    "cyclonic storm, severe, very severe, and extremely severe or super cyclonic storm. "
    "Those names are this project's grouping, not an official IMD class."
)
GROUPING_LABEL = "Project grouping"


def public_banner(text: str) -> str:
    """Remove a server path appended after ``Path:``."""
    marker = " Path: "
    head, sep, _tail = text.partition(marker)
    return head.rstrip() if sep else text


def read_upload(upload) -> np.ndarray:
    """Read one uploaded ``.npy`` or image file. Does not apply a label."""
    import io

    name = str(upload.name).lower()
    data = upload.getvalue()
    if name.endswith(".npy"):
        return np.load(io.BytesIO(data))
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)), dtype=np.float32)


def example_frames(root: Path = ROOT) -> list[Path]:
    """Bundled synthetic sample, then smoke frames when that set exists."""
    from cloudie.config import load_config

    frames: list[Path] = []
    bundled = root / "cloudie" / "samples" / "synthetic_ir.npy"
    if bundled.is_file():
        frames.append(bundled)
    manifest = Path(load_config(root / "config" / "config_smoke.yaml")["paths"]["manifest"])
    if not manifest.is_absolute():
        manifest = root / manifest
    image_dir = manifest.parent / "images"
    if image_dir.is_dir():
        names = {path.name for path in frames}
        frames.extend(
            sorted(
                path
                for path in image_dir.glob("*.npy")
                if path.is_file() and path.name not in names
            )
        )
    return frames


def render_prediction(
    prediction: Prediction,
    edges: tuple[float, ...] | None = None,
    names: tuple[str, ...] | None = None,
) -> None:
    """Draw wind speed, intensity category, and category confidence."""
    import streamlit as st

    if not prediction.show_number or prediction.wind_kt is None:
        st.error(public_banner(prediction.banner))
        st.metric("Wind interval", "—")
        st.caption(prediction.confidence_status or "uncalibrated")
        return
    percent = (
        "—"
        if prediction.confidence is None
        else f"{prediction.confidence * 100:.0f}%"
    )
    status = prediction.confidence_status or "uncalibrated"
    st.subheader(f"{prediction.wind_kt:.1f} kt · {prediction.category_label} · {percent}")
    wind, category, confidence, interval = st.columns(4)
    wind.metric("Wind speed", f"{prediction.wind_kt:.1f} kt")
    category.metric(GROUPING_LABEL, prediction.category_label or "—")
    confidence.metric("Confidence", percent)
    interval.metric("Wind interval", "—")
    st.caption(status)
    if prediction.grouping_note:
        st.caption(prediction.grouping_note)
    if edges is not None and names is not None:
        st.caption(
            f"{prediction.category_label} covers "
            f"{category_span(prediction.project_class, edges, names)}. "
            "This is the project's grouping, not an official IMD class."
        )
    st.caption(CONFIDENCE_NOTE)
    if prediction.confidence is not None:
        st.progress(min(1.0, max(0.0, float(prediction.confidence))))
    if prediction.class_probabilities:
        chart = pd.DataFrame(
            {
                "Category": [label for label, _prob in prediction.class_probabilities],
                "Confidence": [prob for _label, prob in prediction.class_probabilities],
            }
        ).set_index("Category")
        st.bar_chart(chart, horizontal=True)
    if prediction.is_synthetic:
        st.caption(prediction.banner)
    if prediction.review:
        with st.expander("Review notes"):
            for item in prediction.review:
                st.write(item)


def main() -> None:
    import streamlit as st

    from cloudie.config import checkpoint_path, load_config
    from cloudie.predict import missing_checkpoint, predict_images
    from cloudie.schema import DEFAULT_EDGES_KT, DEFAULT_NAMES

    st.set_page_config(page_title="Cloudie", layout="wide")
    st.title("Cloudie")
    st.error(NO_TRAINED_MODEL_BANNER)
    st.header(PAGE_LEAD)
    st.caption(SCALE_NOTE)

    default_config = ROOT / "config" / "config.yaml"
    loaded_default = load_config(default_config)
    default_checkpoint = checkpoint_path(loaded_default, ROOT)
    with st.sidebar:
        st.header("Checkpoint")
        show_paths = st.checkbox("Show config and weights", value=False)
        if show_paths:
            config_path = st.text_input("Config", value=str(default_config))
            checkpoint_text = st.text_input(
                "Trained weights",
                value=str(default_checkpoint),
            )
            smoke_checkpoint = checkpoint_path(load_config(ROOT / "config" / "config_smoke.yaml"))
            st.caption(
                "A smoke check uses config/config_smoke.yaml and "
                f"{smoke_checkpoint.as_posix()}."
            )
        else:
            config_path = str(default_config)
            checkpoint_text = str(default_checkpoint)
        previous_text = st.text_input("Last advisory wind (kt)", value="")
        st.caption("Optional. Used only to flag a jump or a category change.")

    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_file():
        render_prediction(missing_checkpoint(checkpoint))
        st.info("Train a checkpoint, then point Trained weights at the .pt file.")
        return

    predict_clicked = st.button("Predict intensity", type="primary")
    pull = st.button(IMD_BUTTON)
    st.caption("Latest Infrared-1 image from the IMD satellite page. Predicting does not fetch it.")
    examples = example_frames()
    example_name = "(none)"
    if examples:
        example_name = st.selectbox(
            "Example infrared frame",
            options=["(none)", *[path.name for path in examples]],
            help="Local synthetic frames. An upload replaces the example.",
        )
    uploads = st.file_uploader(
        "Infrared image",
        type=["npy", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        help="One infrared frame. Four frames, named in time order, when the checkpoint is the ConvLSTM.",
    )
    has_example = example_name not in (None, "(none)")

    try:
        config = load_config(config_path)
    except (OSError, ValueError) as exc:
        st.error(f"Could not read the config. {exc}")
        return
    previous = None
    if previous_text.strip():
        try:
            previous = float(previous_text)
        except ValueError:
            st.error("Last advisory wind must be a number of knots.")
            return
    edges = tuple(float(edge) for edge in config["data"]["imd5_edges_kt"])
    names = tuple(str(name) for name in config["data"]["imd5_names"])
    if not edges:
        edges = DEFAULT_EDGES_KT
        names = DEFAULT_NAMES

    if pull:
        try:
            with st.spinner("Pulling the current IMD satellite image"):
                frame = fetch_current_infrared()
            picture, result = st.columns([1, 1])
            with picture:
                st.image(frame.preview, use_container_width=True)
                st.caption(describe_frame(frame))
                st.caption(frame.source_url)
            with result:
                prediction = predict_images(
                    config,
                    checkpoint,
                    [frame.image],
                    previous_advisory_kt=previous,
                )
                render_prediction(prediction, edges, names)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
        return

    if not predict_clicked:
        st.info("Pull the current IMD satellite image, or add an infrared frame.")
        return
    if not uploads and not has_example:
        st.error("Add an infrared frame, then predict intensity.")
        return
    try:
        if uploads:
            ordered = sorted(uploads, key=lambda item: item.name)
            images = [read_upload(item) for item in ordered]
        else:
            chosen = next(path for path in examples if path.name == example_name)
            images = [np.load(chosen)]
        prediction = predict_images(
            config,
            checkpoint,
            images,
            previous_advisory_kt=previous,
        )
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return
    render_prediction(prediction, edges, names)


if __name__ == "__main__":
    main()
