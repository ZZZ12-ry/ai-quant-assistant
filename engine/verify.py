"""验证模块 — 让每一步计算结果可追溯、可手动复算"""
import sys; sys.path.insert(0, ".")
from engine.data import get_main_contract_data
from strategies.simple_ma.model import SimpleMAStrategy, load_params
from engine.backtest import BacktestEngine
import pandas as pd; import numpy as np
from pathlib import Path

def run_traceable_backtest(symbol: str, params: dict, label: str, out_dir: Path):
    """运行一次回测并保存所有中间数据"""
    df = get_main_contract_data(symbol, "20200101", "20260603")
    strategy = SimpleMAStrategy(params)
    bars = strategy.run(df)
    engine = BacktestEngine(100000)
    result = engine.run(bars, params)

    # 保存原始数据
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 交易明细
    trades = result["trades"]
    trades.to_csv(out_dir / f"{label}_trades.csv", index=False)
    
    # 权益曲线
    equity = result["equity_curve"]
    equity.to_csv(out_dir / f"{label}_equity.csv", index=False)
    
    # 带信号的K线（取关键列）
    key_cols = ["date","open","high","low","close","volume","VWMA1","VWMA3",
                "VWMA1_locked_long","MAOBV","cond1","cond2","cond3","cond4","cond5","cond6",
                "N_high","N_low","signal_raw","position","entry_price","stop_price"]
    available_cols = [c for c in key_cols if c in result["bars"].columns]
    result["bars"][available_cols].to_csv(out_dir / f"{label}_bars.csv", index=False)
    
    stats = result["stats"]
    return trades, equity, stats, out_dir

def build_verification_report(symbol: str, label: str, params: dict, out_dir: Path):
    """生成验证报告"""
    trades, equity, stats, _ = run_traceable_backtest(symbol, params, label, out_dir)
    
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
    <title>验证报告 - {label}</title>
    <style>
    body{{max-width:1000px;margin:0 auto;padding:20px;font-family:monospace;background:#f5f5f5}}
    h1{{border-bottom:2px solid #333;padding-bottom:8px}}
    .section{{background:#fff;border-radius:8px;padding:20px;margin:16px 0;box-shadow:0 2px 8px rgba(0,0,0,0.08)}}
    .formula{{background:#f0f0f0;padding:10px;border-radius:4px;margin:8px 0;font-family:monospace}}
    .data{{font-size:13px}}
    table{{font-size:13px;border-collapse:collapse;width:100%}}
    th,td{{padding:4px 8px;text-align:right;border-bottom:1px solid #ddd}}
    th{{text-align:center;background:#333;color:#fff}}
    td:first-child,th:first-child{{text-align:left}}
    .pos{{color:#d32f2f}} .neg{{color:#388e3c}}
    a{{color:#1565c0}}
    </style></head><body>
    <h1>验证报告 — {label}</h1>
    """
    
    stats = stats if "error" not in stats else {}
    n = len(trades)
    
    # ── 指标定义 + 计算公式 + 实际值 ──
    html += '<div class="section"><h2>统计指标追溯</h2>'
    
    metric_defs = [
        ("总交易次数", f"所有已平仓交易的数量", f"{n}笔",
         f"trades.csv 中共 {n} 行"),
        ("净利润", "所有交易平仓盈亏之和 - 总手续费", f"￥{stats.get('net_profit',0):,.0f}",
         f"SUM(trades[net_pnl]) = ￥{trades['net_pnl'].sum():,.0f}  (可打开trades.csv手动加总和验证)"),
        ("胜率", "盈利交易次数 / 总交易次数", f"{stats.get('win_rate',0)*100:.1f}%",
         f"盈利{stats.get('win_count',0)}笔 / 总{n}笔 = {stats.get('win_rate',0)*100:.1f}%"),
        ("盈利因子", "总盈利 / 总亏损绝对值", f"{stats.get('profit_factor','N/A')}",
         f"打开trades.csv，筛选net_pnl>0的行求和 = 总盈利；筛选net_pnl<0的行求和取绝对值 = 总亏损；两者相除"),
        ("最大回撤", "权益曲线从峰值到谷底的最大跌幅", f"￥{stats.get('max_drawdown',0):,.0f}",
         f"打开equity.csv，计算equity列的cummax，取(equity - cummax)的最小值"),
        ("盈亏比", "平均盈利 / 平均亏损的绝对值", f"{stats.get('payoff_ratio',0)}",
         f"盈利: ￥{stats.get('avg_win',0):,.0f} / 亏损: ￥{stats.get('avg_loss',0):,.0f} = {stats.get('payoff_ratio',0)}"),
    ]
    
    html += "<table><tr><th>指标</th><th>定义</th><th>值</th><th>验证方法</th></tr>"
    for name, definition, value, verify in metric_defs:
        html += f"<tr><td><strong>{name}</strong></td><td>{definition}</td><td>{value}</td><td>{verify}</td></tr>"
    html += "</table></div>"
    
    # ── 单笔交易验证示例 ──
    html += '<div class="section"><h2>单笔交易验证（以第1笔为例）</h2>'
    if n > 0:
        t = trades.iloc[0]
        contract_cost = params["contract_multiplier"]
        html += f"""
        <p><strong>开仓：</strong>{t['entry_date']}，方向{t['direction']}，价格{t['entry_price']}</p>
        <p><strong>平仓：</strong>{t['exit_date']}，价格{t['exit_price']}，原因{t['exit_reason']}</p>
        <p><strong>持仓：</strong>{t['bars_held']}根K线</p>
        <div class="formula">
        毛盈亏 = ({t['exit_price']} - {t['entry_price']}) × {contract_cost} × 方向<br>
        方向：多=1，空=-1<br>
        手续费 = {t['exit_price']} × {contract_cost} × 0.0001<br>
        净盈亏 = 毛盈亏 - 手续费
        </div>
        <p><strong>验证：</strong></p>
        <p>毛盈亏 = ({t['exit_price']} - {t['entry_price']}) × {contract_cost} = ￥{t['gross_pnl']:,.2f}</p>
        <p>手续费 = {t['exit_price']} × {contract_cost} × 0.0001 = ￥{t['fee']:,.2f}</p>
        <p><strong>净盈亏 = {t['net_pnl']:,.2f}</strong></p>
        <p>打开 {label}_trades.csv 第1行，核对以上数字。</p>
        """
    html += "</div>"
    
    # ── 策略信号验证 ──
    html += '<div class="section"><h2>策略信号验证（以第1次入场为例）</h2>'
    if n > 0:
        # 找到这笔交易对应K线的信号
        bars_path = out_dir / f"{label}_bars.csv"
        entry_time = pd.to_datetime(trades.iloc[0]["entry_date"])
        entry_bar = pd.read_csv(bars_path, parse_dates=["date"])
        entry_bar = entry_bar[entry_bar["date"] == entry_time]
        
        if len(entry_bar) > 0:
            eb = entry_bar.iloc[0]
            html += f"""
            <p><strong>入场日期：</strong>{entry_time.date()}</p>
            <p><strong>入场条件检查（需全部为True）：</strong></p>
            <table>
            <tr><th>条件</th><th>表达式</th><th>值</th><th>通过?</th></tr>
            <tr><td>cond1</td><td>close > VWMA1_locked_long</td><td>{eb.get('cond1','N/A')}</td></tr>
            <tr><td>cond2</td><td>VWMA1 > VWMA2</td><td>{eb.get('cond2','N/A')}</td></tr>
            <tr><td>cond3</td><td>VWMA2 > VWMA3</td><td>{eb.get('cond3','N/A')}</td></tr>
            <tr><td>cond4-6</td><td>各VWMA斜率 > 0</td><td colspan="2">查看bars.csv中cond4/cond5/cond6</td></tr>
            <tr><td>趋势</td><td>close > VWMA3 AND MAOBV上升</td><td colspan="2">查看bars.csv对应日期行</td></tr>
            <tr><td>突破</td><td>close > N_high (前N根K线最高价)</td><td>{eb.get('close','N/A')} > {eb.get('N_high','N/A')}</td></tr>
            </table>
            <p>打开 {label}_bars.csv，找到日期 {entry_time.date()} 的行，逐列核对以上条件。</p>
            """
    html += "</div>"
    
    # ── 数据文件索引 ──
    html += f'<div class="section"><h2>数据文件索引</h2>'
    html += f'<p>所有原始数据保存在：</p>'
    html += f'<ul>'
    html += f'<li><code>{label}_trades.csv</code> — 每笔交易明细（可打开逐行验证）</li>'
    html += f'<li><code>{label}_equity.csv</code> — 每日权益曲线（可验证最大回撤计算）</li>'
    html += f'<li><code>{label}_bars.csv</code> — 带指标和信号的K线数据（可验证入场条件）</li>'
    html += f'</ul>'
    html += f'<p>推荐验证流程：</p>'
    html += f'<ol>'
    html += f'<li>打开 {label}_trades.csv → 任选一行，手动复算 net_pnl = (exit_price - entry_price) * 10 - fee</li>'
    html += f'<li>打开 {label}_equity.csv → 验证最大回撤 = min(equity - cummax(equity))</li>'
    html += f'<li>打开 {label}_bars.csv → 找到入场日期，验证cond1~cond6是否满足入场条件</li>'
    html += f'</ol>'
    html += "</div></body></html>"
    
    report_path = out_dir / f"{label}_verification.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"[验证] {label} 验证报告: {report_path.resolve()}")
    print(f"[验证] {label} 数据文件: {out_dir.resolve()}\\{label}_*.csv")
    return str(report_path.resolve())

# ── 主入口 ──
if __name__ == "__main__":
    out_dir = Path("reports/verify")
    
    # 基准参数验证 (M=20, N=20)
    p20 = {"M": 20, "N": 20, "init_stop": 0.02, "K_min": 0.3, "decay_step": 0.1,
           "commission_rate": 0.0001, "slippage": 1, "contract_multiplier": 10}
    build_verification_report("RB0", "rb0_m20n20", p20, out_dir)
    
    # 最优参数验证 (M=25, N=10) 
    p25 = {"M": 25, "N": 10, "init_stop": 0.02, "K_min": 0.3, "decay_step": 0.1,
           "commission_rate": 0.0001, "slippage": 1, "contract_multiplier": 10}
    build_verification_report("HC0", "hc0_m25n10", p25, out_dir)
    
    print("\n所有验证报告和数据文件已生成，打开 .html 文件查看验证指引，打开 .csv 文件手动复算。")
