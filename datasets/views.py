import logging
import os
from io import BytesIO

import pandas as pd
from django.conf import settings
from django.shortcuts import render

logger = logging.getLogger(__name__)

# Keys that belong to Django auth — must not be cleared on new upload
_AUTH_KEYS = {"_auth_user_id", "_auth_user_backend", "_auth_user_hash"}

# Supported extensions
_SUPPORTED_EXTS = {".csv", ".xls", ".xlsx", ".json"}


def _read_file(file_obj, ext):
    """
    Parse an uploaded InMemoryUploadedFile / TemporaryUploadedFile into a
    DataFrame.  We read directly from the file object so no temporary disk
    write is needed and there are no path-resolution issues.
    """
    file_obj.seek(0)
    data = file_obj.read()
    buf = BytesIO(data)

    if ext == ".csv":
        return pd.read_csv(buf)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(buf)
    elif ext == ".json":
        return pd.read_json(buf)
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def upload_dataset(request):
    context = {}

    if request.method == "POST" and request.FILES.get("dataset"):
        file = request.FILES["dataset"]
        ext = os.path.splitext(file.name)[1].lower()

        # ── 1. Extension validation ──────────────────────────────────────────
        if ext not in _SUPPORTED_EXTS:
            context["error"] = (
                "Unsupported file format. Please upload a CSV, Excel (.xls / .xlsx), "
                "or JSON file."
            )
            return render(request, "datasets/upload.html", context)

        # ── 2. File size validation ──────────────────────────────────────────
        max_bytes = getattr(settings, "DATASET_MAX_UPLOAD_BYTES", 50 * 1024 * 1024)
        if file.size > max_bytes:
            max_mb = max_bytes // (1024 * 1024)
            context["error"] = (
                f"File is too large ({file.size // (1024 * 1024)} MB). "
                f"Maximum allowed size is {max_mb} MB."
            )
            return render(request, "datasets/upload.html", context)

        # ── 3. Parse directly from the in-memory file object ─────────────────
        # (Avoids saving to disk, no path-resolution issues)
        try:
            df = _read_file(file, ext)
        except Exception as exc:
            logger.error(
                "Failed to parse uploaded file '%s': %s",
                file.name, exc, exc_info=True,
            )
            context["error"] = (
                f"Could not read the file. Please ensure it is a valid, "
                f"non-corrupted {ext.upper()} file and try again."
            )
            return render(request, "datasets/upload.html", context)

        # ── 4. Basic dataset validation ──────────────────────────────────────
        if df.empty:
            context["error"] = "The uploaded file contains no data."
            return render(request, "datasets/upload.html", context)

        if len(df.columns) < 2:
            context["error"] = (
                "The dataset must have at least 2 columns (features + target)."
            )
            return render(request, "datasets/upload.html", context)

        # Drop fully-empty columns silently (common in messy Excel exports)
        df = df.dropna(axis=1, how="all")

        # Warn about duplicate column names and auto-rename
        if len(df.columns) != len(set(df.columns)):
            context["warning"] = (
                "Duplicate column names were detected and have been renamed automatically."
            )
            df.columns = _make_unique_columns(df.columns)

        # ── 5. Clear old session data (preserve auth keys) ───────────────────
        old_keys = [k for k in list(request.session.keys()) if k not in _AUTH_KEYS]
        for key in old_keys:
            del request.session[key]

        # ── 6. Store in session with orient=columns (downstream compat) ──────
        request.session["dataset"] = df.to_json(orient="columns")
        logger.info(
            "Dataset uploaded: '%s' — %d rows × %d columns",
            file.name, len(df), len(df.columns),
        )

        context["preview"] = df.to_html(classes="if-table")
        context["rows"] = len(df)
        context["cols"] = len(df.columns)
        context["upload_success"] = True
        context["filename"] = file.name

    return render(request, "datasets/upload.html", context)


def _make_unique_columns(columns):
    """Append _1, _2 … to duplicate column names."""
    seen = {}
    new_cols = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    return new_cols
