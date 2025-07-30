from typing import Dict, List, Any
import mplcursors  
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re, numpy as np, matplotlib.cm as cm, matplotlib.colors as mcolors
__all__ = [
    "plot_harmonics",
    "plot_dips",
    "plot_dip_timeline",
    "plot_thd_avg_bar",
    "embed_fig",
    "CHART_REGISTRY",
]

def _normalize(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "")

def _to_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace("", "0")
        .astype(float)
    )

def _choose_column(df, candidates, target):
    norm = {_normalize(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize(cand)
        if key in norm:
            return norm[key]
    raise KeyError(f"DataFrame lacks '{target}' column; tried {candidates}.")

def plot_harmonics(
    df_dict: Dict[str, pd.DataFrame],
    _params: Dict[str, Any] | None = None,
) -> Figure:
    """
    Harmonics per-phase Avg/Max 折线图。

    Parameters
    ----------
    df_dict : {"harm": DataFrame, ...}
        controller / GUI 传入的表格字典。这里只用到 "harm"。
    _params : Dict[str, Any] | None
        预留给阈值等外部参数，当前未使用。

    Returns
    -------
    matplotlib.figure.Figure
    """
    df: pd.DataFrame | None = df_dict.get("harm")
    if df is None or df.empty:
        raise ValueError("No harmonics data was provided.")

    # 自动匹配列名，容错大小写 / 空格
    phase_col = _choose_column(df, ["phase", "channel", "ch"], "phase")
    order_col = _choose_column(df, ["order", "harm_order"], "order")
    avg_col   = _choose_column(df, ["avg", "avgpeak", "avg_peak"], "avg")
    max_col   = _choose_column(df, ["max", "maxpeak", "max_peak", "peak"], "max")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Harmonics per Phase – AVG vs. MAX")
    ax.set_xlabel("Harmonic Order")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, linestyle=":", alpha=0.6)

    # 为每个 phase 画两条线：AVG 与 MAX
    for phase in sorted(df[phase_col].unique(), key=str):
        sub = df[df[phase_col] == phase]
        # 按阶次升序，防止折线乱跳
        sub = sub.sort_values(order_col)

        ln_avg, = ax.plot(
            sub[order_col], sub[avg_col],
            marker="o", label=f"{phase} AVG"
        )
        ln_max, = ax.plot(
            sub[order_col], sub[max_col],
            marker="s", linestyle="--", label=f"{phase} MAX"
        )

    ax.legend(fontsize=8, ncol=2)
    harm_th = None
    if _params:
        harm_th = _params.get("harm_th") or _params.get("thd_th")
        if harm_th is not None:
            try:
                y_th = float(re.sub(r"[^\d.-]", "", str(harm_th)))
                ax.axhline(
                    y=y_th, color="red", linestyle="--",
                    linewidth=1.2, label=f"Threshold = {y_th}"
                )
            except ValueError:
                pass  # 忽略无法转成数字的阈值
    # ------------------

    # -------- mplcursors 交互 --------
    cursor = mplcursors.cursor(
        ax.get_lines(), multiple=True, hover=True
    )

    @cursor.connect("add")
    def _on_add(sel):
        evt = getattr(sel, "mouseevent", None) or getattr(sel, "event", None)
        is_click = evt is not None and getattr(evt, "button", None) == 1

        patch = sel.annotation.get_bbox_patch()
        patch.set(fc="yellow", alpha=0.95 if is_click else 0.45)
        sel.annotation.arrow_patch.set(arrowstyle="->" if is_click else "-")

        if not is_click:                  # ── Hover：保证只留 1 条
            for old in cursor.selections[:-1]:
                if old.annotation.get_bbox_patch().get_alpha() < 0.5:
                    try:
                        cursor.remove_selection(old)  # ≥0.4 推荐
                    except AttributeError:            # 极旧版兜底
                        try:
                            old.annotation.remove()
                        finally:
                            if old in cursor.selections:
                                cursor.selections.remove(old)

        sel.annotation.set_text(
            f"Order {int(sel.target[0])}\n"
            f"{sel.artist.get_label()}: {sel.target[1]:.2f}"
        )

    # ---------------------------------

    ax.legend(fontsize=8, ncol=3)  # 统一在这儿调一次
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig

def plot_dip_timeline(
    df_dict: Dict[str, pd.DataFrame],
    _params: Dict[str, Any] | None = None,
) -> Figure:
    df: pd.DataFrame | None = df_dict.get("dip")
    if df is None:
        df = pd.DataFrame()

    if df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        ax.set_title("Voltage Dip Timeline")
        ax.text(
            0.5, 0.5, "No Dip events found in this report.",
            ha="center", va="center", fontsize=12, transform=ax.transAxes
        )
        ax.axis("off")
        return fig

    time_col = _choose_column(df, ["time", "timestamp", "datetime"], "time")

    try:
        volt_col = _choose_column(df, ["worst_voltage", "voltage", "worst_v"], "worst_voltage")
        ylabel = "Worst Voltage (V)"
    except KeyError:
        volt_col = _choose_column(df, ["depth_percent", "dip_depth", "depth_%"], "depth")
        ylabel = "Depth (%)"


    dur_col = None
    try:
        dur_col = _choose_column(
            df, ["duration_ms", "duration", "ms"], "duration"
        )
    except KeyError:
        pass

    df[time_col] = df[time_col].astype(str).apply(
        lambda t: f"2000-01-01 {t.strip()}" if re.match(r"^\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?$", t.strip()) else t
    )
    df[time_col] = pd.to_datetime(df[time_col], format=None, errors="coerce")

    df = df.dropna(subset=[time_col, volt_col]).sort_values(time_col)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.set_title("Voltage Dip Timeline")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()

    sc = ax.scatter(
        df[time_col],
        df[volt_col],
        marker="o",
        s=45,
        edgecolors="k",
        alpha=0.85,
        label="Dip Event"
    )

    if _params:
        dip_th = _params.get("dip_th")
        if dip_th is not None:
            try:
                y_th = float(re.sub(r"[^\d.\-]", "", str(dip_th)))
                ax.axhline(
                    y=y_th,
                    color="red",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"Threshold = {y_th}"
                )
            except ValueError:
                pass
    ax.legend(fontsize=8)

    cursor = mplcursors.cursor(sc, multiple=True, hover=True)

    @cursor.connect("add")
    def _on_add(sel):
        evt = getattr(sel, "mouseevent", None) or getattr(sel, "event", None)
        is_click = evt is not None and getattr(evt, "button", None) == 1

        patch = sel.annotation.get_bbox_patch()
        patch.set(fc="yellow", alpha=0.95 if is_click else 0.45)
        sel.annotation.arrow_patch.set(arrowstyle="->" if is_click else "-")

        if not is_click:
            for old in cursor.selections[:-1]:
                if old.annotation.get_bbox_patch().get_alpha() < 0.5:
                    try:
                        cursor.remove_selection(old)
                    except AttributeError:
                        try:
                            old.annotation.remove()
                        finally:
                            if old in cursor.selections:
                                cursor.selections.remove(old)
        try:
            t_val = df.iloc[sel.index][time_col]
            time_str = pd.to_datetime(t_val).strftime("%H:%M:%S.%f")[:-3]
        except Exception:
            time_str = "—"

        y_val = sel.target[1]
        dur_txt = "—"
        if dur_col:
            try:
                dur = df.iloc[sel.index][dur_col]
                dur_txt = f"{float(dur):.1f} ms" if pd.notna(dur) else "—"
            except Exception:
                pass

        sel.annotation.set_text(
            f"time: {time_str}\n"
            f"{ylabel}: {y_val:.2f}\n"
            f"Duration: {dur_txt}"
        )

    fig.tight_layout()
    return fig

def embed_fig(fig: Figure, parent_tk: Any) -> None:
    canvas = FigureCanvasTkAgg(fig, master=parent_tk)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

_NUM_RE = re.compile(r"[^\d.\-]+")

def _to_float_series(s: pd.Series) -> pd.Series:
    """剥掉单位/空格，强转 float；失败记 NaN"""
    return pd.to_numeric(s.astype(str).str.replace(_NUM_RE, "", regex=True),
                         errors="coerce")

def plot_dips(df_dict: Dict[str, pd.DataFrame],
              _params: Dict[str, Any] | None = None) -> Figure:
    df = df_dict.get("dip")
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Voltage Dip Scatter")
        ax.axis("off")
        ax.text(0.5, 0.5, "No Dip events found.", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
        return fig

    depth_col = next((c for c in df.columns if _normalize(c) in
                     ["depthpercent", "dipdepth", "depth%"]), None)
    chan_col  = next((c for c in df.columns if _normalize(c) in
                     ["channel", "phase", "ch"]), None)
    dur_col   = next((c for c in df.columns if _normalize(c) in
                     ["durationms", "duration", "ms"]), None)
    volt_col  = next((c for c in df.columns if _normalize(c) in
                     ["worstvoltage", "voltage", "worstv"]), None)

    if volt_col is None:
        raise ValueError("Dip DataFrame缺少 worst_voltage 列，无法绘图。")

    df[volt_col]  = _to_float_series(df[volt_col])
    if depth_col:
        df[depth_col] = _to_float_series(df[depth_col])
    if dur_col:
        df[dur_col]   = _to_float_series(df[dur_col])

    if depth_col and df[depth_col].notna().any():
        x_col, x_lab = volt_col,  "Worst Voltage (V)"
        y_col, y_lab = depth_col, "Depth (%)"
        draw_x_th    = True
    elif dur_col and df[dur_col].notna().any():
        x_col, x_lab = volt_col,  "Worst Voltage (V)"
        y_col, y_lab = dur_col,   "Duration (ms)"
        draw_x_th    = False
    else:                                      
        df = df[df[volt_col].notna()].reset_index()
        x_col, x_lab = "index", "Event #"
        y_col, y_lab = volt_col, "Worst Voltage (V)"
        draw_x_th    = False

    if df.empty or df[[x_col, y_col]].dropna().empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Voltage Dip Scatter")
        ax.axis("off")
        ax.text(0.5, 0.5, "No valid Dip events after cleaning.",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        return fig

    if chan_col and df[chan_col].notna().any():
        chans  = sorted(df[chan_col].dropna().unique(), key=str)
        cmap   = cm.get_cmap("tab10", len(chans))
        norm   = mcolors.BoundaryNorm(range(len(chans)+1), cmap.N)
        colors = [cmap(norm(chans.index(v))) if pd.notna(v) else "#AAAAAA"
                  for v in df[chan_col]]
    else:
        cmap, colors = cm.get_cmap("tab10", 1), ["#1f77b4"] * len(df)
        chan_col = None

    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.set_title("Voltage Dip Scatter")
    ax.set_xlabel(x_lab); ax.set_ylabel(y_lab)
    ax.grid(True, linestyle=":", alpha=0.6)

    sc = ax.scatter(df[x_col], df[y_col],
                    c=colors, edgecolors="k", s=55, alpha=0.85)

    if chan_col:
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=cmap(norm(i)),
                              markeredgecolor="k", markersize=7, label=str(ch))
                   for i, ch in enumerate(chans)]
        ax.legend(handles=handles, title="Channel",
                  fontsize=8, title_fontsize=9)
    if y_col == depth_col:
        ymin = df[y_col].min()
        ymax = df[y_col].max()
        margin = (ymax - ymin) * 0.2 if ymax > ymin else 5
        ax.set_ylim(max(0, ymin - margin), ymax + margin)

    if draw_x_th and x_col == depth_col and _params and _params.get("dip_th") is not None:
        try:
            x_th = float(re.sub(_NUM_RE, "", str(_params["dip_th"])))
            ax.axvline(x=x_th, color="red", linestyle="--",
                    linewidth=1.2, label=f"Threshold = {x_th}%")
            ax.legend(fontsize=8)
        except ValueError:
            pass

    cursor = mplcursors.cursor(sc, multiple=True, hover=True)

    @cursor.connect("add")
    def _on_add(sel):
        evt   = getattr(sel, "mouseevent", None) or getattr(sel, "event", None)
        click = evt is not None and getattr(evt, "button", None) == 1
        patch = sel.annotation.get_bbox_patch()
        patch.set(fc="yellow", alpha=0.95 if click else 0.45)
        sel.annotation.arrow_patch.set(arrowstyle="->" if click else "-")

        if not click:
            for old in cursor.selections[:-1]:
                if old.annotation.get_bbox_patch().get_alpha() < 0.5:
                    try:    cursor.remove_selection(old)
                    except AttributeError:
                        old.annotation.remove(); cursor.selections.remove(old)

        i   = sel.index
        txt = []
        if chan_col: txt.append(f"Channel: {df.iloc[i][chan_col]}")
        txt.append(f"{x_lab}: {df.iloc[i][x_col]:.2f}")
        txt.append(f"{y_lab}: {df.iloc[i][y_col]:.2f}")
        sel.annotation.set_text("\n".join(txt))

    fig.tight_layout()
    return fig


def _normalize(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "")

def plot_thd_avg_bar(df_dict, _params=None):

    df = df_dict.get("thd_avg")
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.set_title("THD per Phase – AVG vs. MAX")
        ax.axis("off")
        ax.text(0.5, 0.5, "No THD data found.", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
        return fig

    phase_col = next((c for c in df.columns
                      if _normalize(c) in {"phase", "channel", "ph"}), None)
    if phase_col is None:
        raise ValueError("THD DataFrame lacks a phase column.")

    numeric_cols = [c for c in df.columns if c != phase_col]
    avg_col = max_col = None
    for c in numeric_cols:
        norm = _normalize(c)
        if "avg" in norm and avg_col is None:
            avg_col = c
        elif "max" in norm and max_col is None:
            max_col = c
    if avg_col is None and max_col is None:
        if len(numeric_cols) >= 2:
            avg_col, max_col = numeric_cols[:2]
        elif numeric_cols:
            avg_col = numeric_cols[0]
    elif avg_col is None and max_col:
        avg_col = next(c for c in numeric_cols if c != max_col)
    elif max_col is None and avg_col and len(numeric_cols) >= 2:
        max_col = next(c for c in numeric_cols if c != avg_col)

    df = df.copy()
    df[avg_col] = pd.to_numeric(
        df[avg_col].astype(str).str.replace(_NUM_RE, "", regex=True),
        errors="coerce")
    if max_col:
        df[max_col] = pd.to_numeric(
            df[max_col].astype(str).str.replace(_NUM_RE, "", regex=True),
            errors="coerce")

    phases = df[phase_col].astype(str).tolist()
    n = len(phases)
    idx = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6.5, n * 1.2), 3.8))
    ax.set_title("THD per Phase – AVG vs. MAX")
    ax.set_ylabel("THD (%)")
    ax.set_xticks(idx)
    ax.set_xticklabels(phases)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    color_avg = "#8abbdd"
    color_max = "#e0d884ac"
    color_nan = "#CCCCCC"

    bars_avg = ax.bar(idx - width / 2, df[avg_col],
                      width, label="AVG",
                      color=color_avg, edgecolor="k")
    bars_max = []
    if max_col:
        max_vals = df[max_col]
        bar_color = np.where(max_vals.notna(), color_max, color_nan)
        bars_max = ax.bar(idx + width / 2, max_vals.fillna(0),
                  width, label="MAX",
                  color=bar_color, edgecolor="k")

        for bar, is_valid in zip(bars_max, max_vals.notna()):
            bar.set_alpha(0.85 if is_valid else 0.4)


    # 5️⃣ 阈值线（来自 _params）
    if _params and _params.get("thd_th") is not None:
        try:
            y_th = float(re.sub(_NUM_RE, "", str(_params["thd_th"])))
            ax.axhline(y=y_th, color="red", linestyle="--", linewidth=1.2,
                       label=f"Threshold = {y_th}%")
        except ValueError:
            pass

    ax.legend(fontsize=8, ncol=3)

    # 6️⃣ mplcursors 交互
    cursor = mplcursors.cursor(list(bars_avg) + list(bars_max),
                               hover=True, multiple=True)

    @cursor.connect("add")
    def _on_add(sel):
        evt = getattr(sel, "mouseevent", None) or getattr(sel, "event", None)
        click = evt is not None and getattr(evt, "button", None) == 1
        patch = sel.annotation.get_bbox_patch()
        patch.set(fc="yellow", alpha=0.95 if click else 0.45)
        sel.annotation.arrow_patch.set(arrowstyle="->" if click else "-")

        if not click:
            for old in cursor.selections[:-1]:
                if old.annotation.get_bbox_patch().get_alpha() < 0.5:
                    try:
                        cursor.remove_selection(old)
                    except AttributeError:
                        old.annotation.remove()
                        if old in cursor.selections:
                            cursor.selections.remove(old)

        idx = int(round(sel.index))
        if idx < 0 or idx >= len(phases):
            sel.annotation.set_text("Index out of range")
            return
        phase = phases[idx]
        bar_type = "AVG" if sel.artist in bars_avg else "MAX"
        val = sel.annotation.xy[1]
        txt = "N/A" if np.isnan(val) else f"{val:.2f} %"
        sel.annotation.set_text(f"Phase: {phase}\n{bar_type}: {txt}")

    fig.tight_layout()
    return fig

__all__.extend(["plot_dips"])

