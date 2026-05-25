from pathlib import Path

from pandas import DataFrame
import pandas as pd

TRACE_COLUMNS = ["timestamp_ms", "stream_id", "payload_bytes", "latency_ms"]

TRACE_DTYPES = {
    "timestamp_ms": "uint32",
    "stream_id": "category",
    "payload_bytes": "uint16",
    "latency_ms": "float32",
}


class Importer:
    # Assumption: No missing values
    @staticmethod
    def import_csv(input_path: str | Path) -> DataFrame:
        return pd.read_csv(input_path, usecols=TRACE_COLUMNS, dtype=TRACE_DTYPES, na_filter=False)
