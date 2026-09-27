from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

PROMPT_KEYS = ("prompts", "prompt", "text", "input", "question", "query", "instruction", "content", "user")
SYSTEM_KEYS = ("system", "system_prompt", "system_message", "instruction_system", "sys")
RESPONSE_KEY = "responses"

SUPPORTED_INPUT_FORMATS = ("jsonl", "json", "csv", "tsv", "txt", "yaml", "yml", "parquet")
SUPPORTED_OUTPUT_FORMATS = ("jsonl", "json", "csv", "tsv", "txt", "yaml", "yml")
SUPPORTED_FORMATS = tuple(sorted(set(SUPPORTED_INPUT_FORMATS) | set(SUPPORTED_OUTPUT_FORMATS)))


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "yml":
        return "yaml"
    if suffix in ("jsonl", "json", "csv", "tsv", "txt", "yaml", "parquet"):
        return suffix
    if suffix == "jsonl":
        return "jsonl"
    return "jsonl"


def normalize_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    if "prompts" not in row and "prompt" not in row:
        for k in PROMPT_KEYS:
            if k in raw and raw[k] is not None:
                row["prompt"] = str(raw[k])
                if "prompts" not in row:
                    row["prompts"] = str(raw[k])
                break
    if "system" not in row:
        for k in SYSTEM_KEYS:
            if k in raw and raw[k] is not None:
                row["system"] = str(raw[k])
                break
    return row


def load_input(path: Path, fmt: str | None = None) -> list[dict[str, Any]]:
    fmt = (fmt or detect_format(path)).lower()
    if fmt == "yml":
        fmt = "yaml"
    if fmt == "jsonl":
        return _load_jsonl(path)
    if fmt == "json":
        return _load_json(path)
    if fmt in ("csv", "tsv"):
        return _load_csv(path, delimiter="\t" if fmt == "tsv" else ",")
    if fmt == "txt":
        return _load_txt(path)
    if fmt in ("yaml", "yml"):
        return _load_yaml(path)
    if fmt == "parquet":
        return _load_parquet(path)
    raise ValueError(f"unsupported input format '{fmt}'; supported: {', '.join(SUPPORTED_INPUT_FORMATS)}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if isinstance(obj, str):
                rows.append(normalize_row({"prompt": obj}))
            elif isinstance(obj, dict):
                rows.append(normalize_row(obj))
            else:
                raise ValueError(f"{path}:{line_no}: expected object or string, got {type(obj).__name__}")
    return rows


def _load_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON: {e}") from e
    if isinstance(data, dict):
        if "prompts" in data or "prompt" in data or "text" in data:
            return [normalize_row(data)]
        if "data" in data and isinstance(data["data"], list):
            data = data["data"]
        elif "rows" in data and isinstance(data["rows"], list):
            data = data["rows"]
        else:
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], (dict, str)):
                    data = v
                    break
            else:
                return [normalize_row(data)]
    if not isinstance(data, list):
        raise ValueError(f"{path}: JSON root must be object or array")
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            rows.append(normalize_row({"prompt": item}))
        elif isinstance(item, dict):
            rows.append(normalize_row(item))
        else:
            raise ValueError(f"{path}[{i}]: expected string or object")
    return rows


def _load_csv(path: Path, delimiter: str = ",") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
            delimiter = dialect.delimiter
        except Exception:
            pass
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header")
        fieldnames_lower = {h.lower(): h for h in reader.fieldnames}
        # map prompt column
        prompt_col = None
        for k in PROMPT_KEYS:
            if k in fieldnames_lower:
                prompt_col = fieldnames_lower[k]
                break
        if prompt_col is None:
            prompt_col = reader.fieldnames[0]
        system_col = None
        for k in SYSTEM_KEYS:
            if k.lower() in fieldnames_lower:
                system_col = fieldnames_lower[k.lower()]
                break
        for raw in reader:
            prompt = raw.get(prompt_col, "")
            if prompt is None or str(prompt).strip() == "":
                continue
            row: dict[str, Any] = {"prompts": str(prompt), "prompt": str(prompt)}
            if system_col and raw.get(system_col):
                row["system"] = str(raw[system_col])
            # carry over extra columns as metadata (excluding prompt/system cols)
            for k, v in raw.items():
                skip = k in (prompt_col, system_col, "prompt", "prompts", "system")
                if not skip and v is not None and str(v) != "":
                    row[k] = v
            rows.append(normalize_row(row))
    return rows


def _load_txt(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append({"prompts": line, "prompt": line})
    return rows


def _load_yaml(path: Path) -> list[dict[str, Any]]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required for yaml input: pip install pyyaml") from e
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{path}: YAML root must be list or object")
    rows: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, str):
            rows.append({"prompts": item, "prompt": item})
        elif isinstance(item, dict):
            rows.append(normalize_row(item))
        else:
            raise ValueError(f"{path}: YAML items must be string or object")
    return rows


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas+pyarrow required for parquet: pip install pandas pyarrow") from e
    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    cols_lower = {c.lower(): c for c in df.columns}
    prompt_col = None
    for k in PROMPT_KEYS:
        if k in cols_lower:
            prompt_col = cols_lower[k]
            break
    if prompt_col is None:
        prompt_col = df.columns[0]
    system_col = None
    for k in SYSTEM_KEYS:
        if k.lower() in cols_lower:
            system_col = cols_lower[k.lower()]
            break
    for _, r in df.iterrows():
        prompt = r[prompt_col]
        try:
            import pandas as _pd

            if _pd.isna(prompt):
                continue
        except Exception:
            pass
        if str(prompt).strip() == "":
            continue
        row: dict[str, Any] = {"prompts": str(prompt), "prompt": str(prompt)}
        if system_col and not _is_na(r[system_col]):
            row["system"] = str(r[system_col])
        for c in df.columns:
            if c not in (prompt_col, system_col):
                v = r[c]
                if not _is_na(v) and str(v) != "":
                    row[str(c)] = v if not _is_na(v) else ""
        rows.append(normalize_row(row))
    return rows


def _is_na(v: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(v))
    except Exception:
        return v is None


def write_output(path: Path, rows: list[dict[str, Any]], fmt: str | None = None) -> None:
    fmt = (fmt or detect_format(path)).lower()
    if fmt == "yml":
        fmt = "yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif fmt == "json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
            f.write("\n")
    elif fmt in ("csv", "tsv"):
        _write_csv(path, rows, delimiter="\t" if fmt == "tsv" else ",")
    elif fmt == "txt":
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(str(r.get(RESPONSE_KEY, r.get("responses", ""))) + "\n")
    elif fmt in ("yaml", "yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML required for yaml output") from e
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(rows, f, allow_unicode=True, sort_keys=False)
    else:
        raise ValueError(f"unsupported output format '{fmt}'; supported: {', '.join(SUPPORTED_OUTPUT_FORMATS)}")


def _write_csv(path: Path, rows: list[dict[str, Any]], delimiter: str = ",") -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    preferred = ["prompts", "prompt", "system", "responses", "error"]
    ordered = [k for k in preferred if k in all_keys] + [k for k in all_keys if k not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in ordered})


def validate_rows(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not rows:
        errors.append("input has 0 rows")
        return False, errors
    for i, r in enumerate(rows):
        try:
            p = r.get("prompts") or r.get("prompt") or r.get("text") or r.get("input")
            if not p or not str(p).strip():
                errors.append(f"row {i}: missing prompt (expected 'prompts'/'prompt'/'text' column)")
        except Exception as e:
            errors.append(f"row {i}: {e}")
        if len(errors) >= 10:
            errors.append("... truncated")
            break
    return len(errors) == 0, errors


def preview_rows(rows: list[dict[str, Any]], n: int = 3) -> str:
    lines: list[str] = []
    for i, r in enumerate(rows[:n]):
        p = str(r.get("prompts", r.get("prompt", "")))[:80].replace("\n", " ")
        s = str(r.get("system", ""))[:40].replace("\n", " ")
        lines.append(f"  [{i}] prompt={p!r}" + (f" system={s!r}" if s else ""))
    if len(rows) > n:
        lines.append(f"  ... +{len(rows) - n} more")
    return "\n".join(lines)


def read_system_prompt_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
