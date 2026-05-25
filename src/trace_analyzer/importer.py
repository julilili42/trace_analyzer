from pathlib import Path

from pandas import DataFrame
import pandas as pd


class Importer:
    @staticmethod
    def import_csv(input_path: str | Path) -> DataFrame:
        return pd.read_csv(input_path, usecols=['timestamp_ms', 'stream_id', 'payload_bytes', 'latency_ms'], dtype={
            "timestamp_ms": "uint32",
            "stream_id": "category",
            "payload_bytes": "uint16",
            "latency_ms": "float64",
        },)
