"""Streamlit demo. Smoke output and a missing checkpoint are not forecasts.

Run with ``streamlit run cloudie/app.py``. The page does not download imagery
and does not write a score into the project documentation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cloudie.predict import NOT_CALIBRATED, Prediction

ROOT = Path(__file__).resolve().parents[1]
REVIEW_NOTE = (
    "A person reviews any category change, any jump versus the last advisory, "
    "any crop that fails the missing-pixel gate, and any dropout spread that "
    "crosses a class boundary."
)


def read_upload(upload) -> np.ndarray:
    """Read one uploaded ``.npy`` or image file. Does not apply a label."""
    import io

    name = str(upload.name).lower()
    data = upload.getvalue()
    if name.endswith(".npy"):
        return np.load(io.BytesIO(data))
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)), dtype=np.float32)


def render_prediction(prediction: Prediction) -> None:
    """Draw a model output. The heading is not a forecast label."""
    import streamlit as st

    if prediction.is_synthetic or not prediction.show_number:
        st.error(prediction.banner)
    else:
        st.warning(prediction.banner)
    st.caption("present_as_forecast: false")
    st.caption(NOT_CALIBRATED)
    if prediction.show_number and prediction.wind_kt is not None:
        st.metric("Model output (kt)", f"{prediction.wind_kt:.1f}")
        st.caption("This number is not a forecast.")
        if prediction.project_class is not None:
            st.write(f"Project class: {prediction.project_class}. Not an IMD warning.")
        if prediction.dropout_std_kt is not None:
            st.write(f"Dropout standard deviation: {prediction.dropout_std_kt:.1f} kt.")
    if prediction.review:
        st.warning("A person needs to review this output.")
        for item in prediction.review:
            st.write(f"- {item}")


def main() -> None:
    import streamlit as st

    from cloudie.config import load_config
    from cloudie.predict import missing_checkpoint, predict_images

    st.set_page_config(page_title="Cloudie", layout="centered")
    st.title("Cloudie")
    st.caption("Not a forecast or a warning system.")
    st.markdown(REVIEW_NOTE)

    config_path = st.text_input("Config", value=str(ROOT / "config" / "config.yaml"))
    checkpoint_text = st.text_input(
        "Checkpoint", value=str(ROOT / "artifacts" / "checkpoint.pt")
    )
    previous_text = st.text_input("Last advisory wind (kt)", value="")
    uploads = st.file_uploader(
        "Infrared frames",
        type=["npy", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        help="One file for a single-frame checkpoint. Four files, named in time order, for the ConvLSTM.",
    )
    checkpoint = Path(checkpoint_text)
    if not checkpoint.is_file():
        render_prediction(missing_checkpoint(checkpoint))
        return
    st.warning(
        "A loaded checkpoint is still not a forecast. A smoke checkpoint is a software check."
    )
    if not uploads:
        st.info("Add a frame to compute a model output.")
        return
    if not st.button("Run model"):
        return
    try:
        config = load_config(config_path)
    except (OSError, ValueError) as exc:
        st.error(f"Could not read the config. This is not a forecast. {exc}")
        return
    previous = None
    if previous_text.strip():
        try:
            previous = float(previous_text)
        except ValueError:
            st.error("Last advisory wind must be a number of knots. This is not a forecast.")
            return
    ordered = sorted(uploads, key=lambda item: item.name)
    try:
        images = [read_upload(item) for item in ordered]
        prediction = predict_images(
            config,
            checkpoint,
            images,
            previous_advisory_kt=previous,
        )
    except (OSError, ValueError) as exc:
        st.error(f"{exc} This is not a forecast.")
        return
    render_prediction(prediction)


if __name__ == "__main__":
    main()
