"""可视化模块 — 回测结果图表"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.offline import get_plotlyjs
from typing import Optional


def _pretty_indicator_name(column: str) -> str:
    mapping = {
        "ma": "均线MA",
        "ma_short": "短均线",
        "ma_long": "长均线",
        "slope": "均线斜率",
        "channel_high": "通道高点",
        "channel_low": "通道低点",
        "trend_high": "趋势上轨",
        "trend_low": "趋势下轨",
        "entry_high": "入场高点",
        "entry_low": "入场低点",
        "exit_high": "出场高点",
        "exit_low": "出场低点",
        "long_entry": "多头入场线",
        "long_exit": "多头出场线",
        "short_entry": "空头入场线",
        "short_exit": "空头出场线",
        "upper_band": "上轨",
        "lower_band": "下轨",
        "n_high": "N周期高点",
        "n_low": "N周期低点",
        "vwma1": "VWMA1",
        "vwma2": "VWMA2",
        "vwma3": "VWMA3",
        "vwma1_locked_long": "锁定均线(多)",
        "vwma1_locked_short": "锁定均线(空)",
        "obv": "OBV",
        "maobv": "MAOBV",
        "expma_short": "短EXPMA",
        "expma_long": "长EXPMA",
        "ema_20": "EMA20",
        "ema_50": "EMA50",
    }
    if column in mapping:
        return mapping[column]
    upper = column.upper()
    if upper.startswith("EMA_"):
        return upper.replace("_", "")
    if upper.startswith("EXPMA_"):
        return upper.replace("_", "")
    if upper.startswith("VWMA"):
        return upper
    return column


def _dynamic_price_overlay_specs(bars: pd.DataFrame) -> list[tuple[str, str, str, float, Optional[str]]]:
    specs: list[tuple[str, str, str, float, Optional[str]]] = []
    seen: set[str] = set()

    fixed_specs = [
        ("vwma1", "VWMA1", "#2563eb", 1.25, None),
        ("vwma2", "VWMA2", "#7c3aed", 1.25, None),
        ("vwma3", "VWMA3", "#0f766e", 1.25, None),
        ("vwma1_locked_long", "锁定均线(多)", "#dc2626", 1.15, "dash"),
        ("vwma1_locked_short", "锁定均线(空)", "#16a34a", 1.15, "dash"),
        ("n_high", "N周期高点", "#f97316", 1.0, "dot"),
        ("n_low", "N周期低点", "#0891b2", 1.0, "dot"),
    ]
    for item in fixed_specs:
        if item[0] in bars.columns:
            specs.append(item)
            seen.add(item[0])

    palette = ["#2563eb", "#7c3aed", "#0f766e", "#e11d48", "#ea580c", "#0891b2", "#65a30d", "#475569"]

    ma_like = []
    band_like = []
    ignore = {
        "open", "high", "low", "close", "volume", "open_interest", "signal_raw", "exit_signal",
        "position_state", "filter_pass", "strategy_k", "strategy_trade_extreme", "bars_held",
        "obv", "maobv", "slope", "vwma1_slope", "vwma2_slope", "vwma3_slope", "tr"
    }
    for column in bars.columns:
        if column in seen or column in ignore:
            continue
        lower = str(column).lower()
        if lower in {"stop_price", "strategy_stop_price"}:
            continue
        if any(token in lower for token in ["_slope", "reason", "date", "error"]):
            continue
        if any(token in lower for token in ["ma", "ema", "expma", "vwma"]) and lower not in {"maobv"}:
            ma_like.append(column)
            continue
        if any(token in lower for token in ["channel_", "trend_", "entry_", "exit_", "_band", "long_entry", "long_exit", "short_entry", "short_exit"]):
            band_like.append(column)

    for i, column in enumerate(ma_like):
        specs.append((column, _pretty_indicator_name(column), palette[i % len(palette)], 1.2, None))
        seen.add(column)
    for i, column in enumerate(band_like):
        specs.append((column, _pretty_indicator_name(column), palette[(i + len(ma_like)) % len(palette)], 1.0, "dot"))
        seen.add(column)

    return specs

def plot_backtest(result: dict, title: str = "回测结果") -> go.Figure:
    """
    生成回测结果图：净值曲线 + 回撤 + K线(带信号/止损)

    返回 plotly Figure 对象，可 .show() 显示或 .write_html() 保存
    """
    bars = result["bars"]
    trades = result["trades"]
    stats = result["stats"]

    # ── 子图布局 ──
    # 主图只放价格同量纲信息；OBV、权益和回撤放副图，避免把K线纵轴压扁。
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.62, 0.18, 0.20],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]],
        subplot_titles=("价格执行图：K线、均线、突破、入场、出场与止损线", "动能确认图：OBV 与 MAOBV", "绩效图：权益曲线与回撤"),
    )

    # ── 第1行：K线 ──
    fig.add_trace(go.Candlestick(
        x=bars["date"],
        open=bars["open"], high=bars["high"],
        low=bars["low"], close=bars["close"],
        name="K线", showlegend=False,
    ), row=1, col=1)

    # 透明命中层：Plotly 的 candlestick 在隐藏默认 hover 后，鼠标命中有时不稳定。
    # 这条不可见 close 线只负责把当前日期传给顶部行情状态栏。
    fig.add_trace(go.Scatter(
        x=bars["date"],
        y=bars["close"],
        mode="lines+markers",
        name="K线数据",
        line=dict(color="rgba(0,0,0,0)", width=1),
        marker=dict(size=12, color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="none",
    ), row=1, col=1)

    def add_line_if_exists(column: str, name: str, color: str, width: float = 1.3, dash=None, row: int = 1):
        if column not in bars.columns:
            return
        series = pd.to_numeric(bars[column], errors="coerce").replace(0, np.nan)
        if not series.notna().any():
            return
        fig.add_trace(go.Scatter(
            x=bars["date"], y=series,
            mode="lines", name=name,
            line=dict(color=color, width=width, dash=dash),
            meta={"price_overlay": row == 1, "source_column": column},
            hovertemplate=f"{name}<br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
        ), row=row, col=1)

    # 主图按当前策略实际产出的列自适应绘制均线、通道和关键执行价位。
    for column, name, color, width, dash in _dynamic_price_overlay_specs(bars):
        add_line_if_exists(column, name, color, width, dash)

    # ── 第2行：非价格量纲的成交量动能指标 ──
    add_line_if_exists("obv", "OBV", "#475569", 1.1, None, row=2)
    add_line_if_exists("maobv", "MAOBV", "#0ea5e9", 1.2, None, row=2)

    # 入场信号点：只使用真实交易明细，不使用持仓过程中的 entry_price 列。
    # bars["entry_price"] 在持仓期间会持续记录当前持仓成本，直接绘制会形成一条密集红带。
    entries = trades.copy() if not trades.empty else pd.DataFrame()
    if len(entries) > 0 and {"entry_date", "entry_price", "direction"}.issubset(entries.columns):
        long_entries = entries[entries["direction"] == "long"]
        short_entries = entries[entries["direction"] == "short"]
        if len(long_entries) > 0:
            fig.add_trace(go.Scatter(
                x=long_entries["entry_date"], y=long_entries["entry_price"],
                mode="markers", marker=dict(symbol="triangle-up", size=12, color="#d32f2f", opacity=0.92, line=dict(width=1, color="#ffffff")),
                name="多头入场",
                meta={"price_overlay": False},
                customdata=np.stack([long_entries.get("exit_reason", pd.Series("", index=long_entries.index)), long_entries.get("net_pnl", pd.Series(0, index=long_entries.index))], axis=-1),
                hovertemplate="多头入场<br>%{x}<br>价格=%{y:.2f}<br>退出=%{customdata[0]}<br>净盈亏=%{customdata[1]:,.2f}<extra></extra>",
            ), row=1, col=1)
        if len(short_entries) > 0:
            fig.add_trace(go.Scatter(
                x=short_entries["entry_date"], y=short_entries["entry_price"],
                mode="markers", marker=dict(symbol="triangle-down", size=12, color="#2e7d32", opacity=0.92, line=dict(width=1, color="#ffffff")),
                name="空头入场",
                meta={"price_overlay": False},
                customdata=np.stack([short_entries.get("exit_reason", pd.Series("", index=short_entries.index)), short_entries.get("net_pnl", pd.Series(0, index=short_entries.index))], axis=-1),
                hovertemplate="空头入场<br>%{x}<br>价格=%{y:.2f}<br>退出=%{customdata[0]}<br>净盈亏=%{customdata[1]:,.2f}<extra></extra>",
            ), row=1, col=1)
        exits = entries.dropna(subset=["exit_date", "exit_price"]) if {"exit_date", "exit_price"}.issubset(entries.columns) else pd.DataFrame()
        if len(exits) > 0:
            exit_colors = ["#c62828" if x > 0 else "#2e7d32" for x in exits.get("net_pnl", pd.Series(0, index=exits.index))]
            fig.add_trace(go.Scatter(
                x=exits["exit_date"], y=exits["exit_price"],
                mode="markers",
                marker=dict(symbol="x", size=11, color=exit_colors, line=dict(width=2)),
                name="出场",
                meta={"price_overlay": False},
                customdata=np.stack([exits.get("exit_reason", pd.Series("", index=exits.index)), exits.get("bars_held", pd.Series(0, index=exits.index)), exits.get("net_pnl", pd.Series(0, index=exits.index))], axis=-1),
                hovertemplate="出场<br>%{x}<br>价格=%{y:.2f}<br>原因=%{customdata[0]}<br>持仓=%{customdata[1]}根K线<br>净盈亏=%{customdata[2]:,.2f}<extra></extra>",
            ), row=1, col=1)

    stop_col = "stop_price" if "stop_price" in bars.columns else "strategy_stop_price" if "strategy_stop_price" in bars.columns else ""
    if stop_col:
        stop_series = pd.to_numeric(bars[stop_col], errors="coerce").replace(0, np.nan)
        if stop_series.notna().any():
            fig.add_trace(go.Scatter(
                x=bars["date"], y=stop_series,
                mode="lines", name="止损线",
                line=dict(color="#f59e0b", width=1.4, dash="dot"),
                meta={"price_overlay": True, "source_column": stop_col},
                hovertemplate="止损线<br>%{x}<br>%{y:.2f}<extra></extra>",
            ), row=1, col=1)

    # ── 第3行：绩效曲线 ──
    equity = result.get("equity_curve", pd.DataFrame()).copy()
    if not equity.empty and {"date", "equity"}.issubset(equity.columns):
        equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
        equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce")
        equity = equity.dropna(subset=["date", "equity"])
        if len(equity) > 0:
            peak = equity["equity"].cummax().replace(0, np.nan)
            drawdown = (equity["equity"] / peak - 1) * 100
            fig.add_trace(go.Scatter(
                x=equity["date"], y=equity["equity"],
                mode="lines", name="权益曲线",
                line=dict(color="#2563eb", width=1.4),
                hovertemplate="权益曲线<br>%{x}<br>%{y:,.2f}<extra></extra>",
            ), row=3, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(
                x=equity["date"], y=drawdown,
                mode="lines", name="回撤",
                line=dict(color="#16a34a", width=1.0),
                fill="tozeroy",
                fillcolor="rgba(22,163,74,0.10)",
                hovertemplate="回撤<br>%{x}<br>%{y:.2f}%<extra></extra>",
            ), row=3, col=1, secondary_y=True)

    # ── 布局设置 ──
    fig.update_layout(
        title="策略执行证据图",
        height=1120,
        hovermode="closest",
        margin=dict(l=54, r=46, t=96, b=34),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family='-apple-system, BlinkMacSystemFont, "Microsoft YaHei", sans-serif', color="#243244"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode="pan",
        modebar=dict(orientation="h"),
    )

    fig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1月", step="month", stepmode="backward"),
                dict(count=3, label="3月", step="month", stepmode="backward"),
                dict(count=1, label="1年", step="year", stepmode="backward"),
                dict(step="all", label="全部"),
            ],
            bgcolor="#f6faff",
            activecolor="#bbdefb",
            font=dict(size=12, color="#334e68"),
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.045, row=3, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#eef3f8", zeroline=False)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="OBV", row=2, col=1)
    fig.update_yaxes(title_text="权益", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="回撤%", row=3, col=1, secondary_y=True)
    fig.update_yaxes(nticks=10, row=1, col=1)
    fig.update_yaxes(nticks=4, row=2, col=1)
    fig.update_yaxes(nticks=4, row=3, col=1)
    fig.update_traces(hoverinfo="none")

    return fig


def plot_yearly_returns(equity: pd.DataFrame) -> go.Figure:
    """
    每年收益率柱状图
    """
    df = equity.copy()
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year")["equity"].apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)

    fig = go.Figure(data=[
        go.Bar(x=yearly.index, y=yearly.values * 100,
               marker_color=["#c62828" if v > 0 else "#2e7d32" for v in yearly.values])
    ])
    fig.update_layout(
        title="年度收益率", yaxis_title="收益率 (%)",
        height=320, margin=dict(l=48, r=24, t=54, b=28),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family='-apple-system, BlinkMacSystemFont, "Microsoft YaHei", sans-serif', color="#243244"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef3f8", zeroline=False)
    return fig


def save_report(result: dict, output_path: str = None):
    """生成完整HTML报告并保存"""
    from html import escape
    from pathlib import Path

    if output_path is None:
        output_path = Path(__file__).parent.parent / "reports" / "backtest_report.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotly_js_path = output_path.parent / "plotly.min.js"
    if not plotly_js_path.exists():
        plotly_js_path.write_text(get_plotlyjs(), encoding="utf-8")
    plotly_script_src = "/reports/web/plotly.min.js" if output_path.parent.name == "web" else "plotly.min.js"

    fig1 = plot_backtest(result)
    trades = result["trades"]
    equity = result["equity_curve"].copy()
    stats = result["stats"]

    yearly_rows = []
    if not equity.empty and "date" in equity.columns and "equity" in equity.columns:
        equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
        equity = equity.dropna(subset=["date"])
        equity["year"] = equity["date"].dt.year
        tr = trades.copy()
        if not tr.empty and "exit_date" in tr.columns:
            tr["exit_date"] = pd.to_datetime(tr["exit_date"], errors="coerce")
            tr["year"] = tr["exit_date"].dt.year
        for year, group in equity.groupby("year"):
            if len(group) < 2:
                continue
            ret = group["equity"].iloc[-1] / group["equity"].iloc[0] - 1
            dd = group["equity"] / group["equity"].cummax() - 1
            trade_count = len(tr[tr["year"] == year]) if not tr.empty and "year" in tr.columns else 0
            yearly_rows.append((int(year), ret * 100, dd.min() * 100, trade_count))

    direction_rows = []
    if not trades.empty and "direction" in trades.columns and "net_pnl" in trades.columns:
        for direction, group in trades.groupby("direction"):
            wins = group[group["net_pnl"] > 0]
            losses = group[group["net_pnl"] < 0]
            avg_loss = abs(losses["net_pnl"].mean()) if len(losses) else 0
            payoff = wins["net_pnl"].mean() / avg_loss if avg_loss else 0
            direction_rows.append((direction, group["net_pnl"].sum(), len(group), len(wins) / len(group), payoff))

    def h(value):
        return escape(str(value))

    def cls_num(value):
        try:
            return "good" if float(value) > 0 else "bad" if float(value) < 0 else "neutral"
        except Exception:
            return "neutral"

    exit_reason_labels = {
        "strategy_trailing_stop": "策略动态移动止损",
        "stop_loss": "固定止损",
        "strategy_exit": "策略主动离场",
        "reverse_signal": "反向信号",
        "final_close": "期末强制平仓",
        "exit_signal": "只平仓信号",
    }
    def exit_label(reason):
        return exit_reason_labels.get(str(reason or ""), str(reason or "未记录"))

    exit_reason_notes = {
        "strategy_trailing_stop": "由策略输出的动态止损线触发。",
        "stop_loss": "由通用固定止损触发。",
        "strategy_exit": "由策略代码中的主动离场条件触发。",
        "reverse_signal": "由反向信号触发。",
        "final_close": "回测结束时仍有持仓，按期末强制平仓处理。",
        "exit_signal": "由只平仓信号触发。",
    }

    def exit_note(reason):
        return exit_reason_notes.get(str(reason or ""), "来自交易明细记录。")

    html = f"""
    <html><head><meta charset="utf-8"><title>回测报告</title>
    <style>
        * {{ box-sizing:border-box; }}
        body {{ font-family:-apple-system,BlinkMacSystemFont,"Microsoft YaHei",sans-serif; margin:0; background:#f3f6fa; color:#243244; }}
        .page {{ max-width:none; width:100%; margin:0 auto; padding:22px 22px 42px; }}
        .hero {{ background:#ffffff; border:1px solid #dbe5ef; border-radius:12px; padding:22px 24px; box-shadow:0 10px 24px rgba(22,42,64,.06); }}
        .hero-top {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-start; flex-wrap:wrap; }}
        .eyebrow {{ font-size:12px; font-weight:700; color:#57708a; letter-spacing:.04em; text-transform:uppercase; }}
        h1 {{ color:#102a43; margin:6px 0 8px; font-size:28px; line-height:1.25; }}
        .subtitle {{ color:#52667a; line-height:1.7; margin:0; max-width:760px; font-size:14px; }}
        h2 {{ color:#102a43; margin:30px 0 12px; font-size:20px; line-height:1.35; }}
        h2::after {{ content:""; display:block; width:40px; height:3px; background:#1565c0; border-radius:99px; margin-top:8px; }}
        p {{ color:#52667a; line-height:1.7; }}
        .muted {{ background:#f8fbff; border:1px solid #dbe5ef; border-left:4px solid #1565c0; border-radius:8px; padding:10px 12px; color:#334e68; font-size:14px; }}
        .good {{ color:#c62828; }} .bad {{ color:#2e7d32; }} .neutral {{ color:#243244; }}
        .market-up {{ color:#c62828; }} .market-down {{ color:#2e7d32; }}
        .badge {{ display:inline-flex; align-items:center; justify-content:center; min-width:42px; height:42px; border-radius:8px; background:#eef6ff; color:#0f4fa8; font-size:24px; font-weight:800; }}
        .badge-text {{ text-align:right; color:#52667a; font-size:13px; line-height:1.6; }}
        .badge-text strong {{ color:#102a43; font-size:15px; }}
        .summary-bar {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin-top:18px; }}
        .summary-item {{ border:1px solid #e1e8f0; border-radius:8px; padding:12px 13px; background:#fbfdff; }}
        .summary-item .label {{ font-size:12px; color:#66788a; margin-bottom:6px; }}
        .summary-item .value {{ font-size:21px; font-weight:800; line-height:1.15; color:#102a43; }}
        .section {{ background:#ffffff; border:1px solid #dbe5ef; border-radius:12px; padding:18px 20px; margin-top:18px; box-shadow:0 8px 18px rgba(22,42,64,.04); }}
        .stats-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; margin:12px 0 4px; }}
        .stat-card {{ background:#fbfdff; padding:12px 13px; border-radius:8px; border:1px solid #e1e8f0; min-height:82px; }}
        .stat-card .value {{ font-size:20px; font-weight:800; line-height:1.2; }}
        .stat-card .label {{ font-size:12px; color:#66788a; margin-top:7px; }}
        table {{ width:100%; border-collapse:separate; border-spacing:0; background:#fff; border:1px solid #e1e8f0; border-radius:10px; overflow:hidden; margin:12px 0 4px; }}
        th,td {{ padding:11px 12px; border-bottom:1px solid #edf2f7; text-align:left; font-size:13px; vertical-align:top; }}
        th {{ background:#f6faff; color:#0f4fa8; font-weight:800; }}
        tr:nth-child(even) td {{ background:#fbfdff; }}
        tr:last-child td {{ border-bottom:none; }}
        .scorecard {{ background:#ffffff; border:1px solid #dbe5ef; border-radius:12px; padding:0; margin-top:18px; overflow:hidden; box-shadow:0 8px 18px rgba(22,42,64,.04); }}
        .score-head {{ display:grid; grid-template-columns:160px 1fr; gap:18px; align-items:stretch; padding:18px 20px; background:#f8fbff; border-bottom:1px solid #e1e8f0; }}
        .rating {{ display:flex; flex-direction:column; justify-content:center; align-items:center; background:#fff; border:1px solid #dbe5ef; border-radius:10px; min-height:120px; }}
        .rating .letter {{ font-size:48px; font-weight:900; color:#0f4fa8; line-height:1; }}
        .rating .score {{ color:#52667a; font-size:13px; margin-top:6px; }}
        .decision {{ font-size:18px; font-weight:800; color:#102a43; margin-bottom:6px; }}
        .flags {{ margin:10px 0 0; padding-left:18px; color:#9a3412; }}
        .dim-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:10px; padding:14px; }}
        .dim {{ border:1px solid #edf2f7; border-radius:8px; padding:12px 13px; background:#fff; }}
        .dim-title {{ display:flex; justify-content:space-between; gap:12px; font-weight:800; color:#102a43; margin-bottom:6px; }}
        .dim p {{ margin:0 0 7px; font-size:13px; color:#52667a; line-height:1.6; }}
        .evidence {{ color:#66788a; font-size:12px; line-height:1.7; }}
        .chart-panel {{ background:#fff; border:1px solid #dbe5ef; border-radius:12px; padding:10px 12px; margin-top:18px; box-shadow:0 8px 18px rgba(22,42,64,.04); }}
        .chart-ticker {{ position:sticky; top:0; z-index:5; display:flex; align-items:center; gap:14px; flex-wrap:wrap; min-height:38px; padding:8px 10px; margin:-2px 0 8px; background:rgba(255,255,255,.96); border:1px solid #dbe5ef; border-radius:8px; box-shadow:0 4px 12px rgba(22,42,64,.06); font-size:13px; color:#243244; }}
        .chart-ticker strong {{ color:#102a43; }}
        .chart-ticker .up {{ color:#c62828; font-weight:800; }}
        .chart-ticker .down {{ color:#2e7d32; font-weight:800; }}
        .chart-ticker .muted-txt {{ color:#66788a; }}
        .chart-help {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; color:#52667a; font-size:13px; margin:4px 0 10px; }}
        .chart-help span {{ background:#f6faff; border:1px solid #dbe5ef; border-radius:999px; padding:5px 10px; }}
        .trade-evidence tbody tr {{ cursor:pointer; }}
        .trade-evidence tbody tr:hover td {{ background:#eef6ff; }}
        .hint {{ font-size:13px; color:#66788a; margin:6px 0 0; }}
        @media (max-width:720px) {{ .page{{padding:16px}} .score-head{{grid-template-columns:1fr}} .badge-text{{text-align:left}} }}
    </style></head><body>
    <div class="page">
    <section class="hero">
      <div class="hero-top">
        <div>
          <div class="eyebrow">Backtest Evidence Report</div>
          <h1>策略回测报告</h1>
          <p class="subtitle">本报告由固定回测引擎生成，统计口径来自交易明细、资金曲线、信号K线数据和执行诊断。</p>
        </div>
      </div>
    """

    # 统计卡片
    if "error" not in stats:
        total_trades = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0)
        net_profit = stats.get("net_profit", 0)
        max_drawdown_pct = stats.get("max_drawdown_pct", 0)
        sharpe_ratio = stats.get("sharpe_ratio", 0)
        cards = [
            ("总交易", total_trades, ""),
            ("胜率", f"{win_rate:.1%}", ""),
            ("净利润", f"￥{net_profit:,.0f}", "good" if net_profit > 0 else "bad"),
            ("总收益率", f"{stats.get('total_return_pct', 0):.2f}%", "good" if stats.get("total_return_pct", 0) > 0 else "bad"),
            ("最大回撤", f"{max_drawdown_pct:.1f}%", "bad"),
            ("夏普比率", sharpe_ratio, "good" if sharpe_ratio > 1 else ""),
            ("收益回撤比", stats.get("return_drawdown_ratio", 0), ""),
            ("盈亏比", stats.get("payoff_ratio", 0), ""),
            ("盈利因子", stats.get("profit_factor", 0), ""),
            ("最长连续亏损", stats.get("max_consecutive_losses", 0), "bad"),
            ("总手续费", f"￥{stats.get('total_fees', 0):,.0f}", ""),
            ("成本/净利润", f"{stats.get('cost_profit_ratio', 0):.2f}%", ""),
            ("冲击成本", f"￥{stats.get('total_impact_cost', 0):,.0f}", ""),
            ("平均盈利", f"￥{stats.get('avg_win', 0):,.0f}", "good"),
            ("平均亏损", f"￥{stats.get('avg_loss', 0):,.0f}", "bad"),
        ]
        summary_cards = [
            ("总收益率", f"{stats.get('total_return_pct', 0):.2f}%", cls_num(stats.get("total_return_pct", 0))),
            ("净利润", f"￥{net_profit:,.0f}", cls_num(net_profit)),
            ("最大回撤", f"{max_drawdown_pct:.2f}%", "bad"),
            ("夏普比率", f"{sharpe_ratio:.2f}", cls_num(sharpe_ratio)),
            ("交易次数", total_trades, "neutral"),
            ("成本/净利润", f"{stats.get('cost_profit_ratio', 0):.2f}%", "bad" if stats.get("cost_profit_ratio", 0) > 30 else "neutral"),
        ]
        html += '<div class="summary-bar">'
        for label, value, cls in summary_cards:
            html += f'<div class="summary-item"><div class="label">{label}</div><div class="value {cls}">{value}</div></div>'
        html += "</div></section>"

        html += '<section class="section"><h2>核心指标</h2>'
        html += '<div class="stats-grid">'
        for label, value, cls in cards:
            html += f'<div class="stat-card"><div class="value {cls}">{value}</div><div class="label">{label}</div></div>'
        html += "</div>"
        if total_trades == 0 and stats.get("no_trade_diagnosis"):
            html += '<h2>无交易诊断</h2><table><tr><th>原因</th><th>证据</th><th>建议</th></tr>'
            for item in stats.get("no_trade_diagnosis", []):
                html += (
                    f'<tr><td>{h(item.get("title", ""))}</td>'
                    f'<td>{h(item.get("detail", ""))}</td>'
                    f'<td>{h(item.get("suggestion", ""))}</td></tr>'
                )
            html += "</table>"
        html += "</section>"

        behavior = stats.get("behavior_diagnostics") or {}
        if behavior:
            html += '<section class="section"><h2>策略行为诊断</h2>'
            html += '<div class="stats-grid">'
            behavior_cards = [
                ("平均持仓", f'{behavior.get("avg_bars_held", 0)} 根K线'),
                ("持仓中位数", f'{behavior.get("median_bars_held", 0)} 根K线'),
                ("最长持仓", f'{behavior.get("max_bars_held", 0)} 根K线'),
                ("原始信号", behavior.get("signal_count", 0)),
                ("实际开仓", behavior.get("entries_opened", 0)),
                ("信号转化率", f'{float(behavior.get("signal_to_entry_rate", 0) or 0):.1%}'),
            ]
            for label, value in behavior_cards:
                html += f'<div class="stat-card"><div class="value">{h(value)}</div><div class="label">{h(label)}</div></div>'
            html += '</div>'
            if behavior.get("flags"):
                html += '<table><tr><th>异常提示</th></tr>'
                for flag in behavior.get("flags", [])[:6]:
                    html += f'<tr><td>{h(flag)}</td></tr>'
                html += '</table>'
            if behavior.get("exit_reason"):
                top_exit = behavior.get("exit_reason", [])[0] or {}
                html += '<table><tr><th>退出方式</th><th>说明</th><th>交易次数</th><th>占比</th><th>净盈利</th><th>平均持仓</th></tr>'
                for row in behavior.get("exit_reason", []):
                    pnl = float(row.get("net_profit", 0) or 0)
                    pnl_cls = "good" if pnl > 0 else "bad" if pnl < 0 else "neutral"
                    reason = row.get("reason", "")
                    html += (
                        f'<tr><td>{h(exit_label(reason))}</td>'
                        f'<td>{h(exit_note(reason))}</td>'
                        f'<td>{h(row.get("trade_count", 0))}</td>'
                        f'<td>{float(row.get("share", 0) or 0):.1%}</td>'
                        f'<td class="{pnl_cls}">{pnl:,.2f}</td>'
                        f'<td>{h(row.get("avg_bars_held", 0))} 根K线</td></tr>'
                    )
                html += '</table>'
            html += '</section>'

        if not trades.empty and {"entry_date", "exit_date", "entry_price", "exit_price", "direction", "net_pnl"}.issubset(trades.columns):
            sample = trades.copy()
            sample["_abs_pnl"] = sample["net_pnl"].abs()
            picks = []
            if len(sample) > 0:
                picks.append(("首笔交易", sample.iloc[0]))
                picks.append(("最大盈利", sample.loc[sample["net_pnl"].idxmax()]))
                picks.append(("最大亏损", sample.loc[sample["net_pnl"].idxmin()]))
            seen = set()
            html += '<section class="section"><h2>逐笔交易证据</h2>'
            html += '<p class="hint">点击表格行可在下方K线图定位这笔交易；红色代表盈利，绿色代表亏损。</p>'
            html += '<table class="trade-evidence" id="tradeEvidence"><thead><tr><th>样本</th><th>方向</th><th>入场</th><th>出场</th><th>退出原因</th><th>仓位模型</th><th>手数</th><th>成本</th><th>净盈亏</th></tr></thead><tbody>'
            for label, row in picks:
                key = (str(row.get("entry_date")), str(row.get("exit_date")), float(row.get("net_pnl", 0) or 0))
                if key in seen:
                    continue
                seen.add(key)
                pnl = float(row.get("net_pnl", 0) or 0)
                pnl_cls = "good" if pnl > 0 else "bad" if pnl < 0 else "neutral"
                direction = "多头" if row.get("direction") == "long" else "空头" if row.get("direction") == "short" else h(row.get("direction", ""))
                entry_date = h(row.get("entry_date", ""))
                exit_date = h(row.get("exit_date", ""))
                html += (
                    f'<tr data-start="{entry_date}" data-end="{exit_date}">'
                    f'<td>{h(label)}</td><td>{direction}</td>'
                    f'<td>{entry_date}<br>{float(row.get("entry_price", 0) or 0):.2f}</td>'
                    f'<td>{exit_date}<br>{float(row.get("exit_price", 0) or 0):.2f}</td>'
                    f'<td>{h(exit_label(row.get("exit_reason", "")))}</td>'
                    f'<td>{h(row.get("sizing_model", "fixed"))}</td>'
                    f'<td>{h(row.get("position_size", ""))}</td>'
                    f'<td>{float(row.get("economic_cost", row.get("total_cost", row.get("fee", 0))) or 0):,.2f}</td>'
                    f'<td class="{pnl_cls}">{pnl:,.2f}</td></tr>'
                )
            html += '</tbody></table></section>'

    else:
        html += "</section>"

    if yearly_rows:
        html += '<section class="section"><h2>年度表现</h2><table><tr><th>年份</th><th>收益率</th><th>最大回撤</th><th>交易次数</th></tr>'
        for year, ret, dd, count in yearly_rows:
            ret_cls = "good" if ret > 0 else "bad" if ret < 0 else "neutral"
            html += f'<tr><td>{year}</td><td class="{ret_cls}">{ret:.2f}%</td><td class="bad">{dd:.2f}%</td><td>{count}</td></tr>'
        html += "</table></section>"

    if direction_rows:
        html += '<section class="section"><h2>多空贡献</h2><table><tr><th>方向</th><th>净盈利</th><th>交易次数</th><th>胜率</th><th>盈亏比</th></tr>'
        for direction, pnl, count, win_rate, payoff in direction_rows:
            label = "多头" if direction == "long" else "空头" if direction == "short" else direction
            pnl_cls = "good" if pnl > 0 else "bad" if pnl < 0 else "neutral"
            html += f'<tr><td>{label}</td><td class="{pnl_cls}">{pnl:,.2f}</td><td>{count}</td><td>{win_rate:.1%}</td><td>{payoff:.2f}</td></tr>'
        html += "</table></section>"

    diagnostics = stats.get("diagnostics") or {}
    data_policy = stats.get("data_policy") if isinstance(stats.get("data_policy"), dict) else {}
    execution_policy = stats.get("execution_policy") if isinstance(stats.get("execution_policy"), dict) else {}
    if diagnostics or data_policy or execution_policy:
        html += '<section class="section"><h2>数据与执行口径</h2>'
        if data_policy:
            quality = data_policy.get("quality") if isinstance(data_policy.get("quality"), dict) else {}
            html += '<table><tr><th>口径</th><th>内容</th></tr>'
            for key, label in [
                ("provider", "数据源"),
                ("contract_mode", "合约口径"),
                ("roll_policy", "换月说明"),
                ("research_level", "研究级别"),
                ("file", "本地文件"),
            ]:
                if data_policy.get(key):
                    html += f"<tr><td>{label}</td><td>{h(data_policy.get(key))}</td></tr>"
            if quality:
                html += (
                    f"<tr><td>数据范围</td><td>{h(quality.get('start', ''))} 至 {h(quality.get('end', ''))}，"
                    f"{h(quality.get('rows', 0))} 根K线，长间隔 {h(quality.get('long_calendar_gaps', 0))} 处，"
                    f"零成交量 {h(quality.get('zero_volume_rows', 0))} 行</td></tr>"
                )
            html += "</table>"
        if execution_policy:
            html += '<table><tr><th>撮合假设</th><th>内容</th></tr>'
            for key, label in [
                ("signal_timing", "信号时点"),
                ("entry_fill", "开仓成交"),
                ("exit_fill", "出场成交"),
                ("slippage", "滑点"),
                ("commission_rate", "手续费率"),
                ("margin_rate", "保证金率"),
                ("max_volume_participation", "最大成交量参与率"),
            ]:
                if key in execution_policy:
                    html += f"<tr><td>{label}</td><td>{h(execution_policy.get(key))}</td></tr>"
            html += "</table>"
    if diagnostics:
        labels = {
            "raw_rows": "原始K线",
            "valid_rows": "有效K线",
            "invalid_rows_removed": "清洗剔除",
            "signals_seen": "原始信号",
            "entries_opened": "实际开仓",
            "add_entries_opened": "实际加仓",
            "orders_rejected_insufficient_cash": "资金不足拒单",
            "orders_rejected_volume_participation": "成交量参与率拒单",
            "orders_blocked_price_limit": "涨跌停阻断",
            "bars_skipped_low_volume": "低成交量跳过",
            "forced_final_close": "期末强制平仓",
        }
        html += '<table><tr><th>执行诊断</th><th>数值</th></tr>'
        for key, label in labels.items():
            if key in diagnostics:
                html += f"<tr><td>{label}</td><td>{diagnostics[key]}</td></tr>"
        html += "</table>"
    if diagnostics or data_policy or execution_policy:
        html += "</section>"

    plot_config = {
        "displaylogo": False,
        "displayModeBar": True,
        "scrollZoom": True,
        "responsive": True,
        "modeBarButtonsToAdd": ["pan2d", "zoom2d", "resetScale2d"],
    }
    fig1_html = pio.to_html(fig1, include_plotlyjs=False, full_html=False, config=plot_config, div_id="mainBacktestChart")

    html += f"""
    <script src="{plotly_script_src}"></script>
    <section class="chart-panel">
      <div class="chart-ticker" id="chartTicker">
        <strong>行情状态</strong>
        <span class="muted-txt">移动鼠标查看当前K线数据</span>
      </div>
      <div class="chart-help"><span>默认拖动：左右平移</span><span>鼠标滚轮：缩放</span><span>底部滑块：快速选择区间</span><span>点击逐笔交易表：定位对应K线</span></div>
      {fig1_html}
    </section>
    <script>
      (function(){{
        const table = document.getElementById("tradeEvidence");
        const chart = document.getElementById("mainBacktestChart");
        const ticker = document.getElementById("chartTicker");
        if(!chart || !window.Plotly) return;
        let autoscaling = false;
        function asTime(value){{
          const t = new Date(value).getTime();
          return Number.isFinite(t) ? t : null;
        }}
        function currentRange(){{
          const layout = chart._fullLayout || {{}};
          const axis = layout.xaxis || {{}};
          const range = axis.range || [];
          if(range.length >= 2){{
            const start = asTime(range[0]);
            const end = asTime(range[1]);
            if(start !== null && end !== null) return [Math.min(start, end), Math.max(start, end)];
          }}
          return [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY];
        }}
        function addValue(values, target, index){{
          if(values == null) return;
          if(Array.isArray(values)){{
            const n = Number(values[index]);
            if(Number.isFinite(n)) target.push(n);
          }}else{{
            const n = Number(values);
            if(Number.isFinite(n)) target.push(n);
          }}
        }}
        function traceName(trace){{
          return String(trace.name || "");
        }}
        function isPriceOverlay(trace){{
          if(!trace) return false;
          if(trace.meta && trace.meta.price_overlay === true) return true;
          const name = traceName(trace);
          return ["VWMA1","VWMA2","VWMA3","锁定均线(多)","锁定均线(空)","N周期高点","N周期低点","止损线"].indexOf(name) >= 0;
        }}
        function autoscalePriceAxis(){{
          if(autoscaling || !chart.data) return;
          autoscaling = true;
          const range = currentRange();
          const candleYs = [];
          const overlayYs = [];
          chart.data.forEach(function(trace){{
            const yaxis = trace.yaxis || "y";
            if(yaxis !== "y") return;
            const xs = trace.x || [];
            for(let i = 0; i < xs.length; i++){{
              const t = asTime(xs[i]);
              if(t === null || t < range[0] || t > range[1]) continue;
              if(trace.type === "candlestick"){{
                addValue(trace.low, candleYs, i);
                addValue(trace.high, candleYs, i);
              }}else if(isPriceOverlay(trace)){{
                addValue(trace.y, overlayYs, i);
              }}
            }}
          }});
          const ys = candleYs.slice();
          if(candleYs.length >= 2 && overlayYs.length){{
            const cMin = Math.min.apply(null, candleYs);
            const cMax = Math.max.apply(null, candleYs);
            const cSpan = Math.max(cMax - cMin, Math.abs(cMax) * 0.01, 1);
            overlayYs.forEach(function(y){{
              if(y >= cMin - cSpan * 0.5 && y <= cMax + cSpan * 0.5) ys.push(y);
            }});
          }}
          if(ys.length >= 2){{
            const minY = Math.min.apply(null, ys);
            const maxY = Math.max.apply(null, ys);
            if(Number.isFinite(minY) && Number.isFinite(maxY) && maxY > minY){{
              const pad = Math.max((maxY - minY) * 0.08, Math.abs(maxY) * 0.002, 1);
              Plotly.relayout(chart, {{"yaxis.range":[minY - pad, maxY + pad]}}).finally(function(){{ autoscaling = false; }});
              return;
            }}
          }}
          autoscaling = false;
        }}
        function formatNumber(value, digits){{
          const n = Number(value);
          if(!Number.isFinite(n)) return "-";
          return n.toLocaleString("zh-CN", {{maximumFractionDigits: digits == null ? 2 : digits}});
        }}
        function traceValue(trace, index){{
          if(!trace) return null;
          if(Array.isArray(trace.y)){{
            const n = Number(trace.y[index]);
            return Number.isFinite(n) ? n : null;
          }}
          if(trace.type === "candlestick" && Array.isArray(trace.close)){{
            const n = Number(trace.close[index]);
            return Number.isFinite(n) ? n : null;
          }}
          return null;
        }}
        function pointIndex(point){{
          if(!point) return -1;
          const candidates = [point.pointNumber, point.pointIndex, point.pointInd];
          for(let i = 0; i < candidates.length; i++){{
            const n = Number(candidates[i]);
            if(Number.isInteger(n) && n >= 0) return n;
          }}
          return -1;
        }}
        function sameDay(a, b){{
          const ta = asTime(a);
          const tb = asTime(b);
          if(ta === null || tb === null) return false;
          return Math.abs(ta - tb) < 12 * 3600 * 1000;
        }}
        function candleTrace(){{
          return (chart.data || []).find(function(trace){{ return trace && trace.type === "candlestick"; }});
        }}
        function candleIndexForX(xValue){{
          const candle = candleTrace();
          if(!candle || !candle.x) return -1;
          for(let i = 0; i < candle.x.length; i++){{
            if(sameDay(candle.x[i], xValue)) return i;
          }}
          return -1;
        }}
        function renderCandleTicker(candle, i){{
          if(!ticker || !candle || i < 0) return false;
          const open = Number(candle.open && candle.open[i]);
          const high = Number(candle.high && candle.high[i]);
          const low = Number(candle.low && candle.low[i]);
          const close = Number(candle.close && candle.close[i]);
          if(!Number.isFinite(open) || !Number.isFinite(high) || !Number.isFinite(low) || !Number.isFinite(close)) return false;
          const prevClose = i > 0 ? Number(candle.close && candle.close[i - 1]) : open;
          const change = close - prevClose;
          const pct = prevClose ? change / prevClose * 100 : 0;
          const cls = change >= 0 ? "up" : "down";
          const arrow = change >= 0 ? "↑" : "↓";
          ticker.innerHTML =
            "<strong>" + String(candle.x[i]).slice(0, 10) + "</strong>" +
            "<span>开 " + formatNumber(open) + "</span>" +
            "<span>高 " + formatNumber(high) + "</span>" +
            "<span>低 " + formatNumber(low) + "</span>" +
            "<span>收 " + formatNumber(close) + "</span>" +
            "<span class='" + cls + "'>" + arrow + " " + formatNumber(change) + " (" + formatNumber(pct, 2) + "%)</span>";
          return true;
        }}
        function updateTicker(point){{
          if(!ticker || !point || !chart.data) return;
          const trace = chart.data[point.curveNumber];
          const i = pointIndex(point);
          if(!trace) return;
          const pointX = trace.x && i >= 0 ? trace.x[i] : point.x;
          const dateText = pointX ? String(pointX).slice(0, 10) : "";
          if(trace.type === "candlestick"){{
            renderCandleTicker(trace, i);
            return;
          }}
          const candle = candleTrace();
          const candleIndex = candleIndexForX(pointX);
          if(traceName(trace) === "K线数据" && candle && candleIndex >= 0){{
            renderCandleTicker(candle, candleIndex);
            return;
          }}
          const value = traceValue(trace, i);
          const name = traceName(trace) || "指标线";
          const suffix = isPriceOverlay(trace) ? "价格图层" : "图层";
          const candleText = candle && candleIndex >= 0
            ? "<span>收 " + formatNumber(candle.close[candleIndex]) + "</span>"
            : "";
          ticker.innerHTML =
            "<strong>" + dateText + "</strong>" +
            "<span>" + name + "</span>" +
            "<span>" + suffix + "</span>" +
            "<span>数值 " + formatNumber(value, 2) + "</span>" +
            candleText;
        }}
        chart.on("plotly_hover", function(eventData){{
          if(eventData && eventData.points && eventData.points.length){{
            updateTicker(eventData.points[0]);
          }}
        }});
        chart.on("plotly_relayout", function(eventData){{
          if(!eventData || autoscaling) return;
          if(eventData["xaxis.range"] || eventData["xaxis.range[0]"] || eventData["xaxis.autorange"]){{
            setTimeout(autoscalePriceAxis, 30);
          }}
        }});
        setTimeout(autoscalePriceAxis, 300);
        if(table){{
          table.querySelectorAll("tbody tr").forEach(function(row){{
            row.addEventListener("click", function(){{
              const start = new Date(row.getAttribute("data-start"));
              const end = new Date(row.getAttribute("data-end"));
              if(isNaN(start.getTime()) || isNaN(end.getTime())) return;
              const pad = Math.max(3, Math.ceil((end - start) / 86400000 * 0.25));
              const x0 = new Date(start.getTime() - pad * 86400000);
              const x1 = new Date(end.getTime() + pad * 86400000);
              Plotly.relayout(chart, {{"xaxis.range":[x0, x1]}}).then(autoscalePriceAxis);
              row.scrollIntoView({{block:"nearest", behavior:"smooth"}});
            }});
          }});
        }}
      }})();
    </script>
    </div></body></html>
    """

    output_path.write_text(html, encoding="utf-8")
    print(f"[报告] 已保存: {output_path.resolve()}")
    return str(output_path.resolve())
