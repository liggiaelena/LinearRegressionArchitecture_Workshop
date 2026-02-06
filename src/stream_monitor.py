"""
stream_monitor.py

A small, notebook-friendly toolkit for "DB playback streaming":
- Poll new rows from a Postgres table (Neon)
- Aggregate into 30s peak windows
- Trigger Alert/Error with persistence (T)
- Live-plot peak current + thresholds + events

Designed for VS Code / Jupyter notebooks.

Usage example (in your notebook):

from stream_monitor import DBPoller, PeakWindowAggregator, AlertEngine, LivePlotter, run_playback

poller = DBPoller(db_url=os.getenv("NEON_DB_URL"), table="robot_telemetry",
                  id_col="id", time_col="elapsed_seconds", value_col="axis1_smooth")

aggregator = PeakWindowAggregator(window_size_sec=30.0)
alerter = AlertEngine(minc=MinC, maxc=MaxC, t_alert=3, t_error=2)
plotter = LivePlotter(minc=MinC, maxc=MaxC, title="Peak Current per 30s (DB playback)")

run_playback(poller, aggregator, alerter, plotter, rows_per_tick=1, tick_seconds=1.0)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable, Dict, Any

import time
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text

import matplotlib.pyplot as plt

from IPython.display import clear_output, display


@dataclass
class TelemetryRow:
    """A single telemetry sample from DB."""
    row_id: int
    t: float
    y: float


class DBPoller:
    """
    Polls a Postgres table for new rows (id > last_id), ordered by id.
    This simulates streaming consumption from a DB.

    Notes:
    - Requires a monotonically increasing id column (BIGSERIAL is ideal).
    - For playback, you typically start with last_id=0.
    """
    def __init__(
        self,
        db_url: str,
        table: str,
        id_col: str = "id",
        time_col: str = "elapsed_seconds",
        value_col: str = "axis1_smooth",
    ) -> None:
        if not db_url:
            raise ValueError("db_url is empty. Set NEON_DB_URL env var or pass a DB URL string.")
        self.db_url = db_url
        self.table = table
        self.id_col = id_col
        self.time_col = time_col
        self.value_col = value_col

        self.engine = create_engine(self.db_url, pool_pre_ping=True)

    def fetch_new(self, last_id: int, limit: int = 1) -> List[TelemetryRow]:
        q = text(f"""
            SELECT {self.id_col} AS id, {self.time_col} AS t, {self.value_col} AS y
            FROM {self.table}
            WHERE {self.id_col} > :last_id
            ORDER BY {self.id_col}
            LIMIT :limit
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(q, {"last_id": last_id, "limit": limit}).fetchall()

        out: List[TelemetryRow] = []
        for r in rows:
            # SQLAlchemy Row supports mapping interface in many environments,
            # but attribute access is also common.
            m = getattr(r, "_mapping", None)
            if m is not None:
                out.append(TelemetryRow(int(m["id"]), float(m["t"]), float(m["y"])))
            else:
                out.append(TelemetryRow(int(r.id), float(r.t), float(r.y)))
        return out


class PeakWindowAggregator:
    """
    Aggregates raw samples into per-window peaks.

    window_id = floor(t / window_size_sec)

    When window_id changes, emits (window_end_time, peak_value).
    """
    def __init__(self, window_size_sec: float = 30.0) -> None:
        if window_size_sec <= 0:
            raise ValueError("window_size_sec must be > 0")
        self.window_size_sec = float(window_size_sec)

        self.current_window_id: Optional[int] = None
        self.buffer: List[float] = []

    def push(self, t: float, y: float) -> Optional[Tuple[float, float]]:
        w_id = int(t // self.window_size_sec)

        if self.current_window_id is None:
            self.current_window_id = w_id

        if w_id == self.current_window_id:
            self.buffer.append(y)
            return None

        # window ended -> compute peak and emit
        peak_val = max(self.buffer) if self.buffer else y
        peak_time = (self.current_window_id + 1) * self.window_size_sec

        # start new window
        self.current_window_id = w_id
        self.buffer = [y]

        return float(peak_time), float(peak_val)


class AlertEngine:
    """
    Threshold + persistence logic.

    - Alert when peak > minc for t_alert consecutive windows
    - Error when peak > maxc for t_error consecutive windows
    """
    def __init__(self, minc: float, maxc: float, t_alert: int = 3, t_error: int = 2) -> None:
        self.minc = float(minc)
        self.maxc = float(maxc)
        self.t_alert = int(t_alert)
        self.t_error = int(t_error)

        self.alert_streak = 0
        self.error_streak = 0

    def update(self, peak_time: float, peak_val: float) -> Dict[str, Any]:
        above_alert = peak_val > self.minc
        above_error = peak_val > self.maxc

        self.alert_streak = self.alert_streak + 1 if above_alert else 0
        self.error_streak = self.error_streak + 1 if above_error else 0

        fired_alert = self.alert_streak >= self.t_alert
        fired_error = self.error_streak >= self.t_error

        return {
            "above_alert": above_alert,
            "above_error": above_error,
            "alert_streak": self.alert_streak,
            "error_streak": self.error_streak,
            "fired_alert": fired_alert,
            "fired_error": fired_error,
        }


class LivePlotter:
    """
    Notebook-friendly live plot using persistent artists (fast & stable).

    Works well in Jupyter/VS Code without needing ax.clear() on every frame.
    """
    def __init__(self, minc: float, maxc: float, title: str = "Streaming Peaks") -> None:
        import matplotlib.pyplot as plt  # local import for notebook flexibility

        self.minc = float(minc)
        self.maxc = float(maxc)
        self.title = title

        self.peak_times: List[float] = []
        self.peak_vals: List[float] = []
        self.alert_points: List[Tuple[float, float]] = []
        self.error_points: List[Tuple[float, float]] = []
        self.annotations = []

        self.plt = plt
        self.plt.ion()
        self.fig, self.ax = self.plt.subplots(figsize=(11, 4))

        # Create artists once
        (self.line_peaks,) = self.ax.plot([], [], marker="o", linewidth=1.5, label="Peak Current (windowed)")

        self.sc_alert = self.ax.scatter([], [], s=90, marker="x", linewidths=2, color="orange", label="Alert", zorder=5)
        self.sc_error = self.ax.scatter([], [], s=120, marker="X", linewidths=2, color="red", label="Error", zorder=6)


        # Threshold lines once
        self.ax.axhline(self.minc, linestyle="--", label="MinC (Alert)")
        self.ax.axhline(self.maxc, linestyle="--", label="MaxC (Error)")

        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Peak Current")
        self.ax.set_title(self.title)
        self.ax.grid(True)
        self.ax.legend()

        # Force first draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_annotation(self, peak_time: float, peak_val: float, text: str):
        ann = self.ax.annotate(
            text,
            xy=(peak_time, peak_val),
            xytext=(peak_time, peak_val + 0.15),
            arrowprops=dict(arrowstyle="->"),
            fontsize=9,
            zorder=10
        )
        self.annotations.append(ann)


    def add_peak(self, peak_time: float, peak_val: float) -> None:
        self.peak_times.append(float(peak_time))
        self.peak_vals.append(float(peak_val))

    def add_event(self, peak_time: float, peak_val: float, kind: str) -> None:
        if kind == "alert":
            self.alert_points.append((float(peak_time), float(peak_val)))
        elif kind == "error":
            self.error_points.append((float(peak_time), float(peak_val)))

    def refresh(self, view_last_seconds: float = 2000.0) -> None:
        # Update line
        self.line_peaks.set_data(self.peak_times, self.peak_vals)

        # Update scatters
        if self.alert_points:
            at, av = zip(*self.alert_points)
            self.sc_alert.set_offsets(np.c_[at, av])
        else:
            self.sc_alert.set_offsets(np.empty((0, 2)))

        if self.error_points:
            et, ev = zip(*self.error_points)
            self.sc_error.set_offsets(np.c_[et, ev])
        else:
            self.sc_error.set_offsets(np.empty((0, 2)))

        # Update axes limits
        if self.peak_times:
            right = self.peak_times[-1]
            left = max(0.0, right - 2000)
            self.ax.set_xlim(left, right + 30)

            recent = self.peak_vals[-200:] if len(self.peak_vals) > 200 else self.peak_vals
            ymin = min(recent + [self.minc, self.maxc])
            ymax = max(recent + [self.minc, self.maxc])
            self.ax.set_ylim(max(0.0, ymin - 0.2), ymax + 0.2)
        

        clear_output(wait=True)
        display(self.fig)

    def set_status(self, status: str):
        self.ax.set_title(status)



def run_playback(
    poller: DBPoller,
    aggregator: PeakWindowAggregator,
    alerter: AlertEngine,
    plotter: Optional[LivePlotter] = None,
    rows_per_tick: int = 1,
    tick_seconds: float = 1.0,
    start_last_id: int = 0,
    max_ticks: Optional[int] = None,
    early_warning: Optional[EarlyWarningEngine] = None,
) -> None:
    """
    Playback rows from DB at a controlled rate (tick_seconds), as if it were streaming.

    - Poll `rows_per_tick` rows each tick
    - Update peak windows
    - Trigger alert/error on peaks
    - Update plot after each peak (window boundary)

    Tip:
    - For debugging faster, increase rows_per_tick (e.g., 50) or reduce window_size_sec.
    """
    last_id = int(start_last_id)
    ticks = 0
    events = []  # list of dicts


    while True:
        if max_ticks is not None and ticks >= max_ticks:
            print("🟦 Max ticks reached. Stopping playback.")
            break

        rows = poller.fetch_new(last_id=last_id, limit=int(rows_per_tick))
        if not rows:
            print("✅ End of playback (no more rows).")
            break

        for r in rows:
            last_id = r.row_id
            emitted = aggregator.push(r.t, r.y)
            if emitted is None:
                continue

            peak_time, peak_val = emitted

            # Update plot series
            if plotter is not None:
                plotter.add_peak(peak_time, peak_val)
                # Early warning (ETA to MaxC)
                if early_warning is not None and plotter is not None:
                    ew_res = early_warning.update(plotter.peak_times, plotter.peak_vals)
                    if ew_res["fired"]:
                        eta_min = ew_res["eta_sec"] / 60.0
                        print(f"🟨 EARLY WARNING: ETA≈{eta_min:.1f} min to MaxC (slope={ew_res['slope']:.6f})")
                        eta_min = ew_res["eta_sec"] / 60.0
                        plotter.set_status(f"🟨 EARLY WARNING: ETA≈{eta_min:.1f} min to MaxC")

                        plotter.add_event(peak_time, peak_val, "alert")


            # Alert logic
            res = alerter.update(peak_time, peak_val)

            # Always mark exceedances (helps visualization)
            if res["above_alert"] and plotter is not None:
                plotter.add_event(peak_time, peak_val, "alert")

            if res["above_error"] and plotter is not None:
                plotter.add_event(peak_time, peak_val, "error")

            # Still print only when it truly triggers (persistence reached)
            if res["fired_alert"] and plotter is not None:
                plotter.add_event(peak_time, peak_val, "alert")
                plotter.add_annotation(peak_time, peak_val, "ALERT")
                events.append({
                    "event": "ALERT",
                    "peak_time": peak_time,
                    "peak_val": peak_val,
                    "minc": alerter.minc,
                    "maxc": alerter.maxc,
                    "alert_streak": res["alert_streak"],
                    "error_streak": res["error_streak"]
                })

            if res["fired_error"] and plotter is not None:
                plotter.add_event(peak_time, peak_val, "error")
                plotter.add_annotation(peak_time, peak_val, "ERROR")
                events.append({
                    "event": "ERROR",
                    "peak_time": peak_time,
                    "peak_val": peak_val,
                    "minc": alerter.minc,
                    "maxc": alerter.maxc,
                    "alert_streak": res["alert_streak"],
                    "error_streak": res["error_streak"]
                })

            # Refresh view after each emitted peak
            if plotter is not None:
                plotter.refresh()

        ticks += 1
        time.sleep(float(tick_seconds))

        if events:
            pd.DataFrame(events).to_csv("alert_events.csv", index=False)
            print("✅ Saved: alert_events.csv")
        else:
            print("No events fired (nothing to save).")


class EarlyWarningEngine:
    """
    Predicts time-to-cross MaxC using a local linear fit on last K peaks.
    Triggers when ETA <= horizon_sec (e.g., 3600s).
    """
    def __init__(
        self,
        maxc: float,
        k: int = 20,
        horizon_sec: float = 3600.0,
        min_slope: float = 1e-4,
        cooldown_windows: int = 20
    ) -> None:
        self.maxc = float(maxc)
        self.k = int(k)
        self.horizon_sec = float(horizon_sec)
        self.min_slope = float(min_slope)
        self.cooldown_windows = int(cooldown_windows)

        self._cooldown = 0  # counts down after a warning fires

    def update(self, peak_times: list, peak_vals: list) -> dict:
        """
        Returns dict:
          - fired: bool
          - eta_sec: float | None
          - t_cross: float | None
          - slope: float | None
        """
        if self._cooldown > 0:
            self._cooldown -= 1

        if len(peak_times) < self.k:
            return {"fired": False, "eta_sec": None, "t_cross": None, "slope": None}

        t = np.array(peak_times[-self.k:], dtype=float)
        y = np.array(peak_vals[-self.k:], dtype=float)

        # Center time to improve numeric stability
        t0 = t[0]
        tc = t - t0

        # Fit y = a*tc + b  (b is value at t0)
        a, b = np.polyfit(tc, y, deg=1)

        if a <= self.min_slope:
            return {"fired": False, "eta_sec": None, "t_cross": None, "slope": float(a)}

        # Solve for crossing: maxc = a*(t_cross - t0) + b
        t_cross = t0 + (self.maxc - b) / a
        t_now = t[-1]
        eta = t_cross - t_now

        fired = (self._cooldown == 0) and (0 < eta <= self.horizon_sec)

        if fired:
            self._cooldown = self.cooldown_windows

        return {"fired": fired, "eta_sec": float(eta), "t_cross": float(t_cross), "slope": float(a)}
