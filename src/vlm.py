from __future__ import annotations

import base64
import io
import logging
import time

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from openai import OpenAI, OpenAIError

from openai.types.responses import (
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputTextParam,
)

from PIL import Image

from .questions import YearAnswer

logger = logging.getLogger(__name__)


@dataclass
class VLMResponse:
    """The answer of one call, plus what it cost and what went wrong."""

    started_at: str
    image_bytes: int = 0            # after downscaling and re-encoding, before base64
    latency_s: float = 0.0          # time spent in the API call alone
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw_text_response: str = ""     # what the model wrote, kept for the dumps
    parsed_response: YearAnswer | None = None

    error: str | None = None


class VLMClient:
    def __init__(self,
                 model: str,
                 api_key: str,
                 max_tokens: int,
                 timeout: float,
                 max_retries: int,
                 image_format: str = "jpeg") -> None:

        self.model = model
        self.max_tokens = max_tokens
        self.image_format = image_format.lower()

        # Retries are handled by the SDK itself (429, 408, 5xx and connection errors).
        self.client = OpenAI(api_key=api_key,
                             timeout=timeout,
                             max_retries=max_retries)

    def encode_image_with(self, path: Path) -> tuple[str, int]:
        """Returns tuple[image data URL, payload size]"""
        is_png = self.image_format == "png"
        fmt = "PNG" if is_png else "JPEG"

        with Image.open(path) as im:
            if not is_png:
                im = im.convert("RGB")

            buffer = io.BytesIO()
            im.save(buffer, format=fmt, quality=85, optimize=True)

        payload = buffer.getvalue()
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:image/{fmt.lower()};base64,{encoded}", len(payload)

    def ask(self, images: Sequence[Path], system: str, user: str) -> VLMResponse:
        """One call with the images attached to the user turn, in order."""
        result = VLMResponse(
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        # A bad path, a truncated file or an unknown format all raise OSError here;
        # keep it off the API clock and turn it into a row rather than a crash.
        try:
            encoded = [self.encode_image_with(path) for path in images]
        except OSError as exc:
            result.error = str(exc)[:500]
            logger.error("image encoding failed — %s: %s",
                         type(exc).__name__, result.error[:300])
            return result

        result.image_bytes = sum(nbytes for _, nbytes in encoded)

        content: ResponseInputMessageContentListParam = [
            ResponseInputTextParam(type="input_text", text=user),
            *(
                ResponseInputImageParam(
                    type="input_image", image_url=uri, detail="auto"
                )
                for uri, _ in encoded
            ),
        ]

        call_started = time.monotonic()
        try:
            response = self.client.responses.parse(
                model=self.model,
                instructions=system,
                input=[{"role": "user", "content": content}],
                text_format=YearAnswer,
                max_output_tokens=self.max_tokens,
                reasoning={"effort": "medium"},
            )
        except OpenAIError as exc:
            result.error = str(exc)[:500]
            logger.error("VLM call failed — %s: %s",
                         type(exc).__name__, result.error[:300])
            return result

        finally:
            result.latency_s = time.monotonic() - call_started

        result.raw_text_response = response.output_text
        result.parsed_response = response.output_parsed

        if response.usage:
            result.prompt_tokens = response.usage.input_tokens
            result.completion_tokens = response.usage.output_tokens

        # A strict schema only fails to parse when the reply was cut short.
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", None) or "unknown"
            result.error = f"incomplete response: {reason}"
            logger.error("VLM response incomplete (%s); raise vlm.max_tokens", reason)

        return result