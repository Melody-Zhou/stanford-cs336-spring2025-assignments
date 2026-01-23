import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class BenchmarkRow:
    specs: str
    model_size: str
    batch_size: int
    context_length: int
    vocab_size: int
    amp: str
    mode: str  # "forward" / "forward+backward" / (optionally "backward")
    warmup_steps: int
    measure_steps: int
    mean_ms: float
    std_ms: float
    tok_per_step: int
    tok_per_s: float
    device: str
    impl: str   # "eager" / "compiled"


class BenchmarkReporter:
    """
    One-stop utility for:
      - appending benchmark results to JSONL
      - reading JSONL into DataFrame
      - rendering a Markdown table
      - writing Markdown to file
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        md_path: Optional[str | Path] = None,
        *,
        float_fmt: str = ".3f",
        sort_cols: Optional[List[str]] = None,
        cols: Optional[List[str]] = None,
        title: str = "#### Benchmark results",
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.md_path = Path(md_path) if md_path else None

        self.float_fmt = float_fmt
        self.sort_cols = sort_cols or ["specs", "model_size", "context_length", "mode", "amp", "impl"]
        self.cols = cols or [
            "specs",
            "model_size",
            "context_length",
            "batch_size",
            "amp",
            "impl",
            "mode",
            "mean_ms",
            "std_ms",
            "tok_per_s",
            "device",
        ]
        self.title = title

        # Ensure directories exist
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.md_path is not None:
            self.md_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- IO ----------
    def append(self, row: BenchmarkRow) -> None:
        """Append one record to JSONL."""
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    def read_df(self) -> pd.DataFrame:
        """Read JSONL into a DataFrame."""
        records: List[Dict[str, Any]] = []
        if not self.jsonl_path.exists():
            return pd.DataFrame()

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return pd.DataFrame.from_records(records)

    # ---------- Rendering ----------
    def render_markdown(self, df: Optional[pd.DataFrame] = None) -> str:
        """Render the markdown table (optionally from a provided df)."""
        if df is None:
            df = self.read_df()
        if df.empty:
            return f"{self.title}\n\n(no rows)\n"

        df = df.copy()

        MODEL_SIZE_ORDER = ["small", "medium", "large", "xl", "2.7b"]
        MODE_ORDER = ["forward", "backward"]

        if "model_size" in df.columns:
            df["model_size"] = pd.Categorical(df["model_size"], categories=MODEL_SIZE_ORDER, ordered=True)

        if "mode" in df.columns:
            df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)     

        # sort + select columns
        if self.sort_cols:
            df = df.sort_values(self.sort_cols)
        if self.cols:
            df = df[self.cols]

        # float formatting
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].map(lambda x: f"{x:{self.float_fmt}}" if pd.notna(x) else "")

        md = []
        md.append(self.title)
        md.append("")
        table = df.to_markdown(index=False)
        table = self.center_align_markdown(table)
        md.append(table)
        # md.append(df.to_markdown(index=False))  # uv add tabulate
        md.append("")
        return "\n".join(md)

    def write_markdown(self, content: Optional[str] = None) -> None:
        """Write markdown to md_path (requires md_path)."""
        if self.md_path is None:
            raise ValueError("md_path is None; pass md_path in BenchmarkReporter(...).")
        if content is None:
            content = self.render_markdown()
        self.md_path.write_text(content, encoding="utf-8")

    # ---------- Convenience ----------
    def append_and_maybe_write(self, row: BenchmarkRow, *, write_md: bool = False) -> None:
        """
        Common pattern:
          - append a new JSONL row
          - optionally refresh markdown file
        """
        self.append(row)
        if write_md:
            self.write_markdown()

    def center_align_markdown(self, md: str) -> str:
        lines = md.splitlines()
        if len(lines) < 2:
            return md

        header = lines[0]
        sep = lines[1]

        cols = header.count("|") - 1
        centered = "| " + " | ".join([":---:" for _ in range(cols)]) + " |"

        return "\n".join([header, centered] + lines[2:])


@dataclass
class AttentionRow:
    d_model: int
    seq_len: int
    fwd_ms: Optional[float]
    bwd_ms: Optional[float]
    mem_before_bwd_mb: Optional[float]
    status: str  # "ok" / "oom" / "error:<Type>"
    impl: str = "eager"


class AttentionBenchmarkReporter:
    """
    Similar spirit to BenchmarkReporter:
      - append JSONL
      - render markdown table
      - write markdown to file
    """

    def __init__(self, jsonl_path: str | Path, md_path: str | Path, *, title: str = "#### PyTorch attention (naive)"):
        self.jsonl_path = Path(jsonl_path)
        self.md_path = Path(md_path)
        self.title = title
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: AttentionRow) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    def read_df(self) -> pd.DataFrame:
        if not self.jsonl_path.exists():
            return pd.DataFrame()
        records: List[Dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _center_align_markdown(md: str) -> str:
        lines = md.splitlines()
        if len(lines) < 2:
            return md
        header = lines[0]
        cols = header.count("|") - 1
        centered = "| " + " | ".join([":---:" for _ in range(cols)]) + " |"
        return "\n".join([lines[0], centered] + lines[2:])

    def render_markdown(self) -> str:
        df = self.read_df()
        if df.empty:
            return f"{self.title}\n\n(no rows)\n"

        # Sort for readability
        sort_cols = ["d_model", "seq_len"]
        if "impl" in df.columns:
            sort_cols.append("impl")
        df = df.sort_values(sort_cols, ascending=True)

        # Format floats
        for c in ["fwd_ms", "bwd_ms", "mem_before_bwd_mb"]:
            if c in df.columns:
                df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")

        md = []
        md.append(self.title)
        md.append("")
        table = df.to_markdown(index=False)
        table = self._center_align_markdown(table)
        md.append(table)
        md.append("")
        return "\n".join(md)

    def write_markdown(self) -> None:
        self.md_path.write_text(self.render_markdown(), encoding="utf-8")


@dataclass
class FlashBenchRow:
    impl: str                   # "baseline" or "flash"
    dtype: str                  # "bf16" / "fp32"
    seq_len: int
    d_model: int
    fwd_ms: Optional[float]
    bwd_ms: Optional[float]
    e2e_ms: Optional[float]
    status: str                 # "ok" / "oom" / "error:<Type>"


class FlashBenchmarkReporter:
    def __init__(
        self,
        jsonl_path: str | Path,
        md_path: str | Path,
        *,
        title: str = "#### FlashAttention-2 (Triton) vs Baseline (PyTorch)"
    ):
        self.jsonl_path = Path(jsonl_path)
        self.md_path = Path(md_path)
        self.title = title
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: FlashBenchRow) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    def read_df(self) -> pd.DataFrame:
        if not self.jsonl_path.exists():
            return pd.DataFrame()
        records: List[Dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _center_align_markdown(md: str) -> str:
        lines = md.splitlines()
        if len(lines) < 2:
            return md
        header = lines[0]
        cols = header.count("|") - 1
        centered = "| " + " | ".join([":---:" for _ in range(cols)]) + " |"
        return "\n".join([lines[0], centered] + lines[2:])

    def render_markdown(self) -> str:
        df = self.read_df()
        if df.empty:
            return f"{self.title}\n\n(no rows)\n"

        # Sort for readability
        sort_cols = ["dtype", "seq_len", "d_model", "impl"]
        df = df.sort_values(sort_cols, ascending=True)

        # Format floats
        for c in ["fwd_ms", "bwd_ms", "e2e_ms"]:
            if c in df.columns:
                df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")

        md = []
        md.append(self.title)
        md.append("")
        table = df.to_markdown(index=False)
        table = self._center_align_markdown(table)
        md.append(table)
        md.append("")
        return "\n".join(md)

    def write_markdown(self) -> None:
        self.md_path.write_text(self.render_markdown(), encoding="utf-8")


@dataclass
class LeaderboardBenchRow:
    variant: str               # e.g. "baseline"
    dtype: str                 # "bf16"
    seq_len: int               # S
    n_heads: int               # H
    d_head: int                # Dh
    fwd_ms: Optional[float]
    bwd_ms: Optional[float]
    e2e_ms: Optional[float]
    status: str                # "ok" / "oom" / "error:<Type>"


class LeaderboardBenchmarkReporter:
    def __init__(self, jsonl_path: str | Path, md_path: str | Path, *, title: str):
        self.jsonl_path = Path(jsonl_path)
        self.md_path = Path(md_path)
        self.title = title
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: LeaderboardBenchRow) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    def read_df(self) -> pd.DataFrame:
        if not self.jsonl_path.exists():
            return pd.DataFrame()
        records: List[Dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _center_align_markdown(md: str) -> str:
        lines = md.splitlines()
        if len(lines) < 2:
            return md
        header = lines[0]
        cols = header.count("|") - 1
        centered = "| " + " | ".join([":---:" for _ in range(cols)]) + " |"
        return "\n".join([lines[0], centered] + lines[2:])

    def render_markdown(self) -> str:
        df = self.read_df()
        if df.empty:
            return f"{self.title}\n\n(no rows)\n"

        sort_cols = ["variant", "dtype", "seq_len", "n_heads", "d_head"]
        df = df.sort_values(sort_cols, ascending=True)

        # format floats
        for c in ["fwd_ms", "bwd_ms", "e2e_ms"]:
            if c in df.columns:
                df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")

        md = [self.title, "", self._center_align_markdown(df.to_markdown(index=False)), ""]
        return "\n".join(md)

    def write_markdown(self) -> None:
        self.md_path.write_text(self.render_markdown(), encoding="utf-8")