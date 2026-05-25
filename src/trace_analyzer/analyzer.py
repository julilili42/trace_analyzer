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
        stream_codes = self.df["stream_id"].cat.codes.to_numpy(copy=False).astype(np.int64,
                                                                                  copy=False)
        stream_names = self.df["stream_id"].cat.categories

        payload = self.df["payload_bytes"].to_numpy(copy=False)
        latency = self.df["latency_ms"].to_numpy(copy=False)

        # optimized mean + sum calculation
        # index of new array corresponds to streamcode, value corresponds to number of occurences
        row_count = np.bincount(stream_codes)
        # summation based on stream codes
        payload_sum = np.bincount(stream_codes, weights=payload)
        latency_sum = np.bincount(stream_codes, weights=latency)
        # division of two 1xn arrays, where n = number of stream_codes
        payload_mean = payload_sum / row_count
        latency_mean = latency_sum / row_count

        sum_frames = np.ceil(payload_sum / frame_size).astype(int)

        # optimized max calculation
        n_streams = len(stream_names)
        payload_max = np.zeros(n_streams, dtype=payload.dtype)
        latency_max = np.full(n_streams, -np.inf, dtype=latency.dtype)

        np.maximum.at(payload_max, stream_codes, payload)
        np.maximum.at(latency_max, stream_codes, latency)

        busload = self.calc_busload_stats(
            window_ms=window_ms,
            link_speed_mbit=link_speed_mbit,
        )

        for i, stream_id in enumerate(stream_names):
            stats[stream_id] = Stats(
                payload=Payload(
                    max=payload_max[i],
                    mean=payload_mean[i],
                    sum_frames=sum_frames[i]
                ),
                latency=Latency(
                    max=latency_max[i],
                    mean=latency_mean[i]
                ),
                busload=Busload(
                    min=busload.loc[stream_id, "min"],
                    max=busload.loc[stream_id, "max"],
                    mean=busload.loc[stream_id, "mean"],
                ),
            )

        return stats

    # assumption: timestamps are roughtly uniformly distributed
    def calc_busload_stats(
        self,
        window_ms: float,
        link_speed_mbit: float,
    ) -> DataFrame:
        if window_ms <= 0:
            raise ValueError("window_ms must be greater than 0")

        if link_speed_mbit <= 0:
            raise ValueError("link_speed_mbit must be greater than 0")

        stream = self.df["stream_id"]
        stream_codes = stream.cat.codes.to_numpy(copy=False).astype(np.int64,
                                                                    copy=False)
        stream_names = stream.cat.categories
        n_streams = len(stream_names)

        timestamps = self.df["timestamp_ms"].to_numpy(copy=False)
        payload = self.df["payload_bytes"].to_numpy(copy=False)

        if float(window_ms).is_integer():
            window_bucket = timestamps // int(window_ms)
        else:
            window_bucket = np.floor(timestamps / window_ms).astype(np.int64)

        window_bucket = window_bucket.astype(np.int64, copy=False)
        n_windows = int(window_bucket.max()) + 1

        pair_key = stream_codes * n_windows + window_bucket

        payload_per_pair = np.bincount(
            pair_key,
            weights=payload,
            minlength=n_streams * n_windows,
        )

        observed_pair_key = np.flatnonzero(payload_per_pair)
        payload_per_pair = payload_per_pair[observed_pair_key]
        stream_per_pair = observed_pair_key // n_windows

        scale = 8 / (link_speed_mbit * 1_000_000 * (window_ms / 1000)) * 100
        busload_values = payload_per_pair * scale

        busload_min = np.full(n_streams, np.inf)
        busload_max = np.full(n_streams, -np.inf)

        np.minimum.at(busload_min, stream_per_pair, busload_values)
        np.maximum.at(busload_max, stream_per_pair, busload_values)

        busload_sum = np.bincount(
            stream_per_pair,
            weights=busload_values,
            minlength=n_streams,
        )
        busload_count = np.bincount(stream_per_pair, minlength=n_streams)
        busload_mean = busload_sum / busload_count

        return DataFrame(
            {
                "min": busload_min,
                "max": busload_max,
                "mean": busload_mean,
            },
            index=stream_names,
        )
