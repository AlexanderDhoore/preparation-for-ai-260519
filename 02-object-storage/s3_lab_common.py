from __future__ import annotations

import hashlib
import json
import os
import struct
import urllib3
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path("/root/preparation-for-ai")
DATASETS_DIR = REPO_ROOT / "datasets"


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or "replace-me" in value or "example.invalid" in value:
        raise SystemExit(f"Missing environment variable: {name}")
    return value


def load_s3():
    load_dotenv(REPO_ROOT / "participant.env")
    endpoint_url = require_env("S3_ENDPOINT_URL")
    participant_bucket = require_env("PARTICIPANT_BUCKET")
    shared_bucket = require_env("SHARED_BUCKET")

    verify_ssl = os.environ.get("S3_VERIFY_SSL", "true").strip().lower()
    verify = verify_ssl not in {"0", "false", "no"}
    if not verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        # TODO 1:
        # Open participant.env and replace these two placeholder names with the
        # names of the variables that contain your S3 access key and secret key.
        aws_access_key_id=require_env("TODO_FILL_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=require_env("TODO_FILL_S3_SECRET_ACCESS_KEY"),
        verify=verify,
    )
    return client, participant_bucket, shared_bucket


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def upload_bytes(
    client, bucket: str, key: str, content: bytes, content_type: str
) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=content,
        ContentType=content_type,
    )


def upload_text(client, bucket: str, key: str, text: str) -> None:
    upload_bytes(
        client,
        bucket,
        key,
        text.encode("utf-8"),
        "text/plain; charset=utf-8",
    )


def upload_json(client, bucket: str, key: str, value: object) -> None:
    upload_bytes(
        client,
        bucket,
        key,
        json.dumps(value, indent=2, sort_keys=True).encode("utf-8"),
        "application/json",
    )


def upload_csv(client, bucket: str, key: str, frame: pd.DataFrame) -> None:
    upload_bytes(
        client,
        bucket,
        key,
        frame.to_csv(index=False).encode("utf-8"),
        "text/csv; charset=utf-8",
    )


def upload_file(client, bucket: str, key: str, path: Path, content_type: str) -> None:
    upload_bytes(client, bucket, key, path.read_bytes(), content_type)


def read_object_bytes(client, bucket: str, key: str) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def read_text(client, bucket: str, key: str) -> str:
    return read_object_bytes(client, bucket, key).decode("utf-8")


def read_manifest(client, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(read_object_bytes(client, bucket, key)))


def list_keys(client, bucket: str, prefix: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(item["Key"] for item in page.get("Contents", []))
    return sorted(keys)


def object_metadata(client, bucket: str, key: str) -> dict[str, object]:
    response = client.head_object(Bucket=bucket, Key=key)
    return {
        "content_length": int(response["ContentLength"]),
        "content_type": response.get("ContentType", ""),
        "etag": str(response.get("ETag", "")).strip('"'),
    }


def read_wav_bytes(content: bytes) -> tuple[np.ndarray, int, int, int]:
    if content[:4] != b"RIFF" or content[8:12] != b"WAVE":
        raise ValueError("Expected RIFF/WAVE audio content.")

    offset = 12
    fmt: tuple[int, int, int, int, int, int] | None = None
    data: bytes | None = None

    while offset + 8 <= len(content):
        chunk_id = content[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", content, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        chunk = content[chunk_start:chunk_end]
        if chunk_id == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", chunk, 0)
        elif chunk_id == b"data":
            data = chunk
        offset = chunk_end + (chunk_size % 2)

    if fmt is None or data is None:
        raise ValueError("WAV object is missing fmt or data chunk.")

    audio_format, channels, sample_rate, _byte_rate, _block_align, bits = fmt
    if audio_format == 1 and bits == 16:
        values = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif audio_format == 3 and bits == 32:
        values = np.frombuffer(data, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV format={audio_format}, bits={bits}.")

    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return values.astype(np.float32), int(sample_rate), int(channels), int(bits)


def audio_features(content: bytes) -> dict[str, float | int]:
    samples, sample_rate, channels, bits = read_wav_bytes(content)
    if len(samples) == 0:
        raise ValueError("Cannot extract features from empty audio.")

    segment = samples[: min(len(samples), 65536)]
    spectrum = np.abs(np.fft.rfft(segment))
    frequencies = np.fft.rfftfreq(len(segment), d=1 / sample_rate)
    spectrum_total = float(spectrum.sum()) or 1.0

    def band_share(low: float, high: float) -> float:
        mask = (frequencies >= low) & (frequencies < high)
        return float(spectrum[mask].sum() / spectrum_total)

    absolute = np.abs(samples)
    return {
        "duration_s": float(len(samples) / sample_rate),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "bits_per_sample": bits,
        "rms_energy": float(np.sqrt(np.mean(samples * samples))),
        "peak_amplitude": float(np.max(absolute)),
        "zero_crossing_rate": float(np.mean(np.abs(np.diff(np.signbit(samples))))),
        "mean_absolute_amplitude": float(np.mean(absolute)),
        "amplitude_q25": float(np.quantile(absolute, 0.25)),
        "amplitude_q50": float(np.quantile(absolute, 0.50)),
        "amplitude_q75": float(np.quantile(absolute, 0.75)),
        "spectral_centroid": float((spectrum * frequencies).sum() / spectrum_total),
        "very_low_frequency_share": band_share(0, 250),
        "low_frequency_share": band_share(250, 500),
        "mid_frequency_share": band_share(500, 2000),
        "high_frequency_share": band_share(2000, 8000),
        "very_high_frequency_share": band_share(8000, 22050),
    }
