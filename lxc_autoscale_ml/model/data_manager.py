import pandas as pd
import logging
import json
from datetime import UTC, datetime

import numpy as np


def _parse_timestamp(value):
    """One metrics timestamp as an aware UTC datetime.

    The monitor writes timezone-aware UTC timestamps; older monitors wrote
    naive local time. Both can sit in the same file for as long as the old
    rows are retained, and pandas refuses to mix them ("Cannot subtract
    tz-naive and tz-aware"), so every value is normalised to UTC here. A naive
    value is read as the host's local time, which is what the old monitor
    meant by it.
    """
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()  # naive -> local time zone
    return parsed.astimezone(UTC)


def _timestamps_to_utc(series):
    return pd.to_datetime(series.map(_parse_timestamp), utc=True)

def load_data(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        logging.info(f"Data loaded successfully from {file_path}.")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading data: {e}")
        return None

    if not data:
        logging.error(f"No data found in {file_path}.")
        return None

    records = []
    for snapshot_index, snapshot in enumerate(data):
        for container_id, metrics in snapshot.items():
            if container_id == "summary":
                continue  # Skip the summary, focus on container data

            try:
                record = {
                    "container_id": container_id,
                    # Which collection cycle the row came from. The monitor
                    # stamps containers of one cycle individually in older
                    # versions, so "same timestamp" cannot identify a cycle.
                    "snapshot": snapshot_index,
                    "timestamp": metrics["timestamp"],
                    "cpu_usage_percent": metrics["cpu_usage_percent"],
                    "memory_usage_mb": metrics["memory_usage_mb"],
                    "swap_usage_mb": metrics.get("swap_usage_mb", 0),
                    "swap_total_mb": metrics.get("swap_total_mb", 0),
                    "process_count": metrics["process_count"],
                    "io_reads": metrics["io_stats"]["reads"],
                    "io_writes": metrics["io_stats"]["writes"],
                    "network_rx_bytes": metrics["network_usage"]["rx_bytes"],
                    "network_tx_bytes": metrics["network_usage"]["tx_bytes"],
                    "filesystem_usage_gb": metrics["filesystem_usage_gb"],
                    "filesystem_total_gb": metrics["filesystem_total_gb"],
                    "filesystem_free_gb": metrics["filesystem_free_gb"],
                }
                records.append(record)
            except KeyError as e:
                logging.warning(f"Missing expected metric {e} in container {container_id}. Skipping.")
                continue

    if not records:
        logging.error("No valid records found in the data.")
        return None

    df = pd.DataFrame(records)
    try:
        df['timestamp'] = _timestamps_to_utc(df['timestamp'])
    except Exception as e:
        logging.error(f"Error converting timestamps: {e}")
        return None

    logging.info("Data preprocessed successfully.")
    return df


def _rolling_trend(series, window):
    """Slope of the rolling mean, or 0 when there is not enough data.

    np.polyfit raises on a single-sample group, which used to abort the whole
    preprocessing step and leave the caller with a half-transformed frame.
    """
    smoothed = series.rolling(window=window, min_periods=1).mean()
    if len(smoothed) < 2 or smoothed.nunique() < 2:
        return 0.0
    return np.polyfit(np.arange(len(smoothed)), smoothed, 1)[0]


def _spike_flags(series, window, threshold):
    """1 where a sample leaves the band of the `window` samples before it.

    Only a full window of history counts: the standard deviation of two or
    three samples is too unstable and flagged ordinary noise. The first
    `window` samples of a container are therefore never flagged. A flat
    history (standard deviation 0) flags any change.
    """
    history = series.shift(1).rolling(window=window, min_periods=window)
    mean, std = history.mean(), history.std()
    spike = (series - mean).abs() > threshold * std
    flat_then_moved = (std == 0) & (series != mean)
    return ((spike & (std > 0)) | flat_then_moved).astype(int)


def _safe_ratio(numerator, denominator):
    """Element-wise division that yields 0 instead of inf/NaN on a zero denominator."""
    ratio = numerator.divide(denominator).replace([np.inf, -np.inf], np.nan)
    return ratio.fillna(0.0)


def preprocess_data(df, config):
    if df.empty:
        logging.error("DataFrame is empty. Skipping preprocessing.")
        return df

    try:
        spike_config = config.get('spike_detection', {})
        spike_threshold = spike_config.get('spike_threshold', 2)
        # rolling_window lives under spike_detection in the shipped config; the
        # top-level lookup always missed it and silently used the default.
        rolling_window_size = spike_config.get(
            'rolling_window', config.get('rolling_window', 5))

        # Derived metrics. A container reporting zero processes or zero memory
        # would otherwise produce inf here, which StandardScaler turns into NaN
        # and IsolationForest then refuses to fit on.
        df['cpu_per_process'] = _safe_ratio(df['cpu_usage_percent'], df['process_count'])
        df['memory_per_process'] = _safe_ratio(df['memory_usage_mb'], df['process_count'])
        df['cpu_memory_ratio'] = _safe_ratio(df['cpu_usage_percent'], df['memory_usage_mb'])
        # Swap pressure relative to the container's own swap size, so a
        # container with 512 MB swap and one with 8 GB are comparable. No swap
        # configured (or enable_swap: false in the monitor) reports 0, not NaN.
        df['swap_usage_percent'] = _safe_ratio(df['swap_usage_mb'], df['swap_total_mb']) * 100

        # Fix for time_diff calculation
        try:
            df['time_diff'] = df.groupby('container_id')['timestamp'].diff().dt.total_seconds()
            # Assigning the result rather than fillna(inplace=True) on a column:
            # the latter is a chained assignment and stops working under
            # pandas copy-on-write.
            df['time_diff'] = df['time_diff'].fillna(0)
        except Exception as e:
            logging.error(f"Error calculating 'time_diff': {e}")
            df['time_diff'] = 0  # Default to 0 in case of errors

        # Rolling statistics for spike detection and trend analysis
        df['rolling_mean_cpu'] = df.groupby('container_id')['cpu_usage_percent'].transform(
            lambda x: x.rolling(window=rolling_window_size, min_periods=1).mean())
        df['rolling_std_cpu'] = df.groupby('container_id')['cpu_usage_percent'].transform(
            lambda x: x.rolling(window=rolling_window_size, min_periods=1).std()).fillna(0)
        df['rolling_mean_memory'] = df.groupby('container_id')['memory_usage_mb'].transform(
            lambda x: x.rolling(window=rolling_window_size, min_periods=1).mean())
        df['rolling_std_memory'] = df.groupby('container_id')['memory_usage_mb'].transform(
            lambda x: x.rolling(window=rolling_window_size, min_periods=1).std()).fillna(0)
        # Swap is tracked as a percentage rather than MB: a container that
        # starts swapping is the signal, and the percentage makes that signal
        # comparable across containers with different swap sizes.
        df['rolling_mean_swap'] = df.groupby('container_id')['swap_usage_percent'].transform(
            lambda x: x.rolling(window=rolling_window_size, min_periods=1).mean())
        df['rolling_std_swap'] = df.groupby('container_id')['swap_usage_percent'].transform(
            lambda x: x.rolling(window=rolling_window_size, min_periods=1).std()).fillna(0)

        # Spike detection, against the window *before* each sample.
        #
        # Comparing a value with rolling stats that already include it can
        # never fire with the shipped settings: in a window of n samples one
        # value lies at most (n-1)/sqrt(n) sample standard deviations from the
        # mean -- 1.79 for rolling_window 5, below spike_threshold 2. The flags
        # are 0/1 integers so the model, which trains on numeric columns only,
        # actually sees them.
        for column, flag in (('cpu_usage_percent', 'cpu_spike'),
                             ('memory_usage_mb', 'memory_spike'),
                             ('swap_usage_percent', 'swap_spike')):
            df[flag] = df.groupby('container_id')[column].transform(
                lambda x: _spike_flags(x, rolling_window_size, spike_threshold))

        # Trend detection using the slope of the rolling window
        df['cpu_trend'] = df.groupby('container_id')['cpu_usage_percent'].transform(
            lambda x: _rolling_trend(x, rolling_window_size))
        df['memory_trend'] = df.groupby('container_id')['memory_usage_mb'].transform(
            lambda x: _rolling_trend(x, rolling_window_size))
        df['swap_trend'] = df.groupby('container_id')['swap_usage_percent'].transform(
            lambda x: _rolling_trend(x, rolling_window_size))

        # Aggregated features
        df['max_cpu'] = df.groupby('container_id')['cpu_usage_percent'].transform(
            lambda x: x.rolling(window=rolling_window_size, min_periods=1).max())
        df['min_cpu'] = df.groupby('container_id')['cpu_usage_percent'].transform(
            lambda x: x.rolling(window=rolling_window_size, min_periods=1).min())
        df['max_memory'] = df.groupby('container_id')['memory_usage_mb'].transform(
            lambda x: x.rolling(window=rolling_window_size, min_periods=1).max())
        df['min_memory'] = df.groupby('container_id')['memory_usage_mb'].transform(
            lambda x: x.rolling(window=rolling_window_size, min_periods=1).min())

        # Deliberately NOT scaled here.
        #
        # This function used to standard-scale twelve columns in place, two of
        # which -- cpu_usage_percent and memory_usage_mb -- are read by
        # determine_scaling_action and compared against percentage thresholds.
        # A z-score never exceeds cpu_scale_up_threshold (75) and is almost
        # always below cpu_scale_down_threshold (30), so every container was
        # scaled DOWN on both axes every cycle regardless of real load. A
        # container at 99% CPU produced the same decision as one at 3%.
        #
        # The scaling was also redundant: train_anomaly_models wraps
        # IsolationForest in Pipeline([('scaler', StandardScaler()), ...]) and
        # fits it on this same frame, so the model standardises its own inputs
        # and its features are unchanged by this removal.

        logging.info("Feature engineering, spike detection, and trend detection completed.")
    except Exception:
        # Returning the half-transformed frame used to hand the model an
        # inconsistent feature set; the caller skips the cycle instead.
        logging.exception("Error during data preprocessing")
        return None

    return df
