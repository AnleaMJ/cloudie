"""Current INSAT infrared image from the public IMD satellite page.

The page https://mausam.imd.gov.in/responsive/satellite.php shows the Asia
sector Infrared-1 JPEG, overwritten about once an hour. This module reads
that JPEG. It does not store the file. Tests must pass a response in;
they must not download the live image.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, UnidentifiedImageError

IMD_SATELLITE_PAGE = "https://mausam.imd.gov.in/responsive/satellite.php"
IMD_IR1_URL = "https://mausam.imd.gov.in/Satellite/3Dasiasec_ir1.jpg"
IMD_BUTTON = "Pull current status from IMD"
IST = ZoneInfo("Asia/Kolkata")
_MAX_BYTES = 20 * 1024 * 1024
_USER_AGENT = "Mozilla/5.0 (compatible; Cloudie/1.0; +https://mausam.imd.gov.in/responsive/satellite.php)"


@dataclass(frozen=True)
class SatelliteFrame:
    """One Infrared-1 sector image. ``image`` is a single display-count field."""

    image: np.ndarray
    preview: np.ndarray
    updated_at: datetime | None
    source_url: str


def _luminance(rgb: np.ndarray) -> np.ndarray:
    channels = rgb.astype(np.float32)
    return 0.299 * channels[..., 0] + 0.587 * channels[..., 1] + 0.114 * channels[..., 2]


def _parse_updated(last_modified: str | None) -> datetime | None:
    if not last_modified:
        return None
    try:
        parsed = parsedate_to_datetime(last_modified)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed


def frame_from_response(
    body: bytes, last_modified: str | None, source_url: str
) -> SatelliteFrame:
    """Decode one IMD JPEG or PNG. Color is collapsed to one channel."""
    if not body:
        raise ValueError("IMD satellite response was empty")
    if len(body) > _MAX_BYTES:
        raise ValueError("IMD satellite image is larger than expected")
    try:
        picture = Image.open(io.BytesIO(body))
        rgb = np.asarray(picture.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("IMD satellite response was not an image") from exc
    if rgb.ndim != 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
        raise ValueError("IMD satellite image is too small")
    return SatelliteFrame(
        image=_luminance(rgb),
        preview=rgb,
        updated_at=_parse_updated(last_modified),
        source_url=source_url,
    )


def format_updated(when: datetime | None) -> str:
    """Clock time in India Standard Time, where the satellite page is published."""
    if when is None:
        return "IMD did not send an update time."
    local = when.astimezone(IST)
    hour = local.strftime("%I").lstrip("0") or "0"
    return f"Updated {local.strftime('%d %b %Y')}, {hour}:{local.strftime('%M %p').lower()} IST"


def describe_frame(frame: SatelliteFrame) -> str:
    return f"INSAT-3DS Infrared-1, Asia sector. {format_updated(frame.updated_at)}"


def fetch_current_infrared(
    url: str = IMD_IR1_URL, timeout: float = 30.0
) -> SatelliteFrame:
    """Download the current Infrared-1 JPEG. Raises ValueError on failure."""
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            last_modified = response.headers.get("Last-Modified")
            body = response.read(_MAX_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"Could not pull the current IMD satellite image. {exc}") from exc
    return frame_from_response(body, last_modified, url)
