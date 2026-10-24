"""The IMD button reads a supplied response. It does not download in CI."""

from __future__ import annotations

import io
from datetime import timezone
from urllib.error import URLError

import numpy as np
import pytest
from PIL import Image

from cloudie.imd_satellite import (
    IMD_BUTTON,
    IMD_IR1_URL,
    describe_frame,
    fetch_current_infrared,
    frame_from_response,
)


def _png(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def test_color_image_collapses_to_one_channel() -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 100
    rgb[..., 1] = 150
    rgb[..., 2] = 200
    frame = frame_from_response(
        _png(rgb), "Sat, 24 Oct 2026 19:55:13 GMT", IMD_IR1_URL
    )
    expected = 0.299 * 100 + 0.587 * 150 + 0.114 * 200
    assert frame.image.shape == (4, 6)
    assert frame.image.dtype == np.float32
    assert frame.image == pytest.approx(expected)
    assert frame.preview.shape == (4, 6, 3)
    assert frame.updated_at is not None
    assert frame.updated_at.tzinfo is not None
    assert frame.updated_at.astimezone(timezone.utc).hour == 19
    text = describe_frame(frame)
    assert "Infrared-1" in text
    assert "25 Oct 2026" in text
    assert "1:25 am IST" in text


def test_missing_update_time_is_reported() -> None:
    rgb = np.full((3, 3, 3), 40, dtype=np.uint8)
    frame = frame_from_response(_png(rgb), "not a date", IMD_IR1_URL)
    assert frame.updated_at is None
    assert "did not send an update time" in describe_frame(frame)


def test_empty_response_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        frame_from_response(b"", None, IMD_IR1_URL)


def test_fetch_reads_the_response_without_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    rgb = np.full((5, 5, 3), 12, dtype=np.uint8)
    payload = _png(rgb)

    class _Response:
        headers = {"Last-Modified": "Sat, 24 Oct 2026 19:55:13 GMT"}

        def read(self, _limit: int) -> bytes:
            return payload

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    monkeypatch.setattr("cloudie.imd_satellite.urlopen", lambda *args, **kwargs: _Response())
    frame = fetch_current_infrared()
    assert frame.source_url == IMD_IR1_URL
    assert frame.image.shape == (5, 5)
    assert frame.image == pytest.approx(12.0)


def test_fetch_failure_raises_and_returns_no_image(monkeypatch: pytest.MonkeyPatch) -> None:
    def _down(*args: object, **kwargs: object) -> None:
        raise URLError("down")

    monkeypatch.setattr("cloudie.imd_satellite.urlopen", _down)
    with pytest.raises(ValueError, match="IMD"):
        fetch_current_infrared()


def test_button_label() -> None:
    import cloudie.app as app

    assert app.IMD_BUTTON == IMD_BUTTON == "Pull current status from IMD"
