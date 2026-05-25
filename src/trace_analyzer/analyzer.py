from dataclasses import dataclass
from pandas import DataFrame
import numpy as np


@dataclass
class Payload:
    mean: float
    max: float
    sum_frames: int


@dataclass
class Latency:
    mean: float
    max: float


@dataclass
class Busload:
    min: float
    max: float
    mean: float


@dataclass
class Stats:
    payload: Payload
    latency: Latency
    busload: Busload


class Analyzer:
    def __init__(self, df: DataFrame):
        self.df = df

    def calc_stats(
        self,
        frame_size: int = 64,
        window_ms: float = 10,
        link_speed_mbit: float = 100,
    ) -> dict[str, Stats]:
        stats: dict[str, Stats] = {}
        stream_stats = self.df.groupby("stream_id", sort=False, observed=True).agg(
            payload_max=("payload_bytes", "max"),
            payload_mean=("payload_bytes", "mean"),
            payload_total=("payload_bytes", "sum"),
            latency_max=("latency_ms", "max"),
            latency_mean=("latency_ms", "mean"),

        )

        stream_stats["payload_sum_frames"] = np.ceil(
            stream_stats["payload_total"] / frame_size).astype(int)

        busload = self._calc_busload_stats(
            window_ms=window_ms,
            link_speed_mbit=link_speed_mbit,
        )

        for stream_id in stream_stats.index:
            stats[stream_id] = Stats(
                payload=Payload(
                    max=stream_stats.loc[stream_id, "payload_max"],
                    mean=stream_stats.loc[stream_id, "payload_mean"],
                    sum_frames=int(
                        stream_stats.loc[stream_id, "payload_sum_frames"]),
                ),
                latency=Latency(
                    max=stream_stats.loc[stream_id, "latency_max"],
                    mean=stream_stats.loc[stream_id, "latency_mean"],
                ),
                busload=Busload(
                    min=busload.loc[stream_id, "min"],
                    max=busload.loc[stream_id, "max"],
                    mean=busload.loc[stream_id, "mean"],
                ),
            )

        return stats

    def _calc_busload_stats(
        self,
        window_ms: float,
        link_speed_mbit: float,
    ) -> DataFrame:
        if window_ms <= 0:
            raise ValueError("window_ms must be greater than 0")

        if link_speed_mbit <= 0:
            raise ValueError("link_speed_mbit must be greater than 0")

        window_start_ms = (self.df["timestamp_ms"] // window_ms) * window_ms

        payload_per_window = (
            self.df.groupby(["stream_id", window_start_ms],
                            sort=False, observed=True)["payload_bytes"].sum()
        )

        scale = 8 / (link_speed_mbit * 1_000_000 * (window_ms / 1000)) * 100

        busload = (payload_per_window * scale).groupby(
            level="stream_id",
            sort=False,
            observed=True
        ).agg(
            ["min", "max", "mean"]
        )

        return busload

    def sum_frames(self, x, frame_size: int) -> int:
        if frame_size <= 0:
            raise ValueError("frame_size must be greater than 0")

        return int(np.ceil(x.sum() / frame_size))
