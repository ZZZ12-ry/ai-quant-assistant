"""回测引擎 v3 — 期货完整版"""
import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["date", "open", "high", "low", "close", "volume", "signal_raw"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"回测数据缺少必要字段: {missing}")
        if df.empty:
            raise ValueError("回测数据为空")
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce") if "date" in out.columns else out.index
        if "exit_signal" not in out.columns:
            out["exit_signal"] = 0
        for col in ["open", "high", "low", "close", "volume", "signal_raw", "exit_signal"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        for col in ["strategy_stop_price", "strategy_k", "strategy_trade_extreme", "strategy_take_profit_price", "strategy_take_profit_fraction"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["date", "open", "high", "low", "close"])
        if out.empty:
            raise ValueError("回测数据清洗后为空")
        invalid_price = (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
        if invalid_price.any():
            out = out.loc[~invalid_price].copy()
        bad_ohlc = (out["high"] < out[["open", "close", "low"]].max(axis=1)) | (out["low"] > out[["open", "close", "high"]].min(axis=1))
        if bad_ohlc.any():
            out = out.loc[~bad_ohlc].copy()
        if out.empty:
            raise ValueError("回测数据价格关系异常，清洗后为空")
        out["signal_raw"] = out["signal_raw"].fillna(0).astype(int).clip(-1, 1)
        out["exit_signal"] = out["exit_signal"].fillna(0).astype(int).clip(0, 1)
        return out.sort_values("date").reset_index(drop=True)

    def _read_params(self, params: dict, symbol: str = "") -> dict:
        """读取并校验参数，补齐默认值"""
        p = {
            "contract_multiplier": params.get("contract_multiplier", 10),
            "commission_rate": params.get("commission_rate", 0.0001),
            "fee_mode": params.get("fee_mode", "percent"),    # percent | per_lot
            "fee_per_lot": params.get("fee_per_lot", 0.0),     # 按手收费时每手金额
            "min_commission": params.get("min_commission", 0.0),
            "impact_cost": params.get("impact_cost", 0.0),
            "impact_threshold_1": params.get("impact_threshold_1", 100000.0),
            "impact_threshold_2": params.get("impact_threshold_2", 500000.0),
            "slippage": params.get("slippage", 1),
            "max_volume_participation": params.get("max_volume_participation", 0.1),
            "exit_mode": params.get("exit_mode", "stop_loss"),
            "margin_rate": params.get("margin_rate", 0.08),
            "init_stop": params.get("init_stop", 0.02),
            "K_min": params.get("K_min", 0.3),
            "decay_step": params.get("decay_step", 0.1),
            "price_limit_pct": params.get("price_limit_pct", 0.0),  # 0=不检查涨跌停
            "min_volume": params.get("min_volume", 0),      # 最小成交量过滤
            "max_position": params.get("max_position", 1),  # 最大持仓手数，支持加仓策略
            "position_sizing": params.get("position_sizing", "fixed"),  # fixed | risk
            "fixed_lots": params.get("fixed_lots", 1),
            "min_position": params.get("min_position", 1),
            "risk_per_trade": params.get("risk_per_trade", 0.01),
            "risk_stop_pct": params.get("risk_stop_pct", params.get("init_stop", 0.02)),
            "cash_usage_limit": params.get("cash_usage_limit", 0.95),
        }
        return p

    def _compute_costs(self, price: float, p: dict) -> dict:
        """计算单边交易成本，fee 保持为总成本以兼容旧统计。"""
        trade_value = abs(price * p["contract_multiplier"])
        if p["fee_mode"] == "per_lot" and p["fee_per_lot"] > 0:
            commission = float(p["fee_per_lot"])
        else:
            commission = trade_value * p["commission_rate"]
        commission = max(commission, float(p["min_commission"]))

        impact_rate = float(p["impact_cost"])
        if trade_value > p["impact_threshold_2"]:
            impact_rate *= 2.0
        elif trade_value > p["impact_threshold_1"]:
            impact_rate *= 1.5
        impact = trade_value * impact_rate
        return {
            "commission": commission,
            "slippage_cost": 0.0,
            "impact_cost": impact,
            "total_cost": commission + impact,
        }

    def _compute_fee(self, price: float, p: dict, is_entry: bool) -> float:
        """兼容旧调用：返回总交易成本。"""
        return self._compute_costs(price, p)["total_cost"]

    def _margin_required(self, price: float, p: dict) -> float:
        """计算保证金要求"""
        return price * p["contract_multiplier"] * p["margin_rate"]

    def _target_entry_lots(self, price: float, cash: float, p: dict) -> dict:
        """Calculate initial entry lots.

        fixed mode is used for strategy replication. risk mode sizes the first
        entry by account risk, then applies margin and max-position caps.
        """
        max_position = max(int(p.get("max_position", 1) or 1), 1)
        min_position = max(int(p.get("min_position", 1) or 1), 1)
        sizing = str(p.get("position_sizing", "fixed") or "fixed").lower()
        margin_per_lot = max(self._margin_required(price, p), 0.0)
        cash_cap = max(float(cash) * float(p.get("cash_usage_limit", 0.95) or 0.95), 0.0)
        margin_lots = int(cash_cap // margin_per_lot) if margin_per_lot > 0 else max_position

        if sizing != "risk":
            requested = max(int(p.get("fixed_lots", 1) or 1), 1)
            lots = min(requested, max_position, margin_lots)
            return {
                "lots": max(lots, 0),
                "sizing_model": "fixed",
                "risk_amount": 0.0,
                "risk_per_lot": 0.0,
                "raw_risk_lots": 0,
                "margin_lots": margin_lots,
            }

        risk_pct = max(float(p.get("risk_per_trade", 0.01) or 0.01), 0.0)
        stop_pct = max(float(p.get("risk_stop_pct", p.get("init_stop", 0.02)) or 0.02), 0.0001)
        risk_amount = max(float(self.initial_capital if cash <= 0 else cash) * risk_pct, 0.0)
        risk_per_lot = max(abs(float(price) * stop_pct * float(p["contract_multiplier"])), 0.0)
        raw_risk_lots = int(risk_amount // risk_per_lot) if risk_per_lot > 0 else 0
        if raw_risk_lots <= 0 and risk_amount > 0 and risk_per_lot > 0:
            raw_risk_lots = 1
        lots = min(raw_risk_lots, margin_lots, max_position)
        if lots < min_position:
            lots = min_position if min_position <= margin_lots and min_position <= max_position else 0
        return {
            "lots": max(int(lots), 0),
            "sizing_model": "risk",
            "risk_amount": risk_amount,
            "risk_per_lot": risk_per_lot,
            "raw_risk_lots": raw_risk_lots,
            "margin_lots": margin_lots,
        }

    def run(self, df: pd.DataFrame, params: dict, symbol: str = "") -> dict:
        raw_rows = len(df)
        df = self._validate_input(df)
        n = len(df)
        p = self._read_params(params, symbol)
        diagnostics = {
            "raw_rows": int(raw_rows),
            "valid_rows": int(n),
            "invalid_rows_removed": int(raw_rows - n),
            "bars_skipped_missing_price": 0,
            "bars_skipped_low_volume": 0,
            "signals_seen": int((df["signal_raw"] != 0).sum()),
            "exit_signals_seen": int((df["exit_signal"] != 0).sum()),
            "entries_opened": 0,
            "add_entries_opened": 0,
            "strategy_exits": 0,
            "strategy_stop_exits": 0,
            "strategy_take_profit_exits": 0,
            "signal_reversals": 0,
            "orders_rejected_insufficient_cash": 0,
            "orders_rejected_volume_participation": 0,
            "orders_rejected_position_sizing": 0,
            "orders_blocked_price_limit": 0,
            "forced_final_close": 0,
            "execution_model": "signal_on_close_fill_next_open; stop_intrabar_fill_stop_price",
            "position_sizing": str(p.get("position_sizing", "fixed")),
        }

        position = 0  # signed lots
        entry_price = 0.0
        entry_idx = 0
        entry_date = None
        stop_price = 0.0
        trade_extreme = 0.0
        entry_costs = {"commission": 0.0, "slippage_cost": 0.0, "impact_cost": 0.0, "total_cost": 0.0}
        cash = self.initial_capital
        margin_used = 0.0
        add_count = 0
        entry_signal_date = None
        entry_fill_mode = ""
        entry_sizing = {
            "sizing_model": str(p.get("position_sizing", "fixed")),
            "risk_amount": 0.0,
            "risk_per_lot": 0.0,
            "raw_risk_lots": 0,
            "margin_lots": 0,
        }
        trades = []

        df["position"] = 0
        df["stop_price"] = np.nan
        df["entry_price"] = np.nan
        equity_curve = []
        prev_close = None

        def signed_dir() -> int:
            return 1 if position > 0 else (-1 if position < 0 else 0)

        def add_costs(total: dict, inc: dict) -> dict:
            return {
                "commission": total["commission"] + inc["commission"],
                "slippage_cost": total["slippage_cost"] + inc["slippage_cost"],
                "impact_cost": total["impact_cost"] + inc["impact_cost"],
                "total_cost": total["total_cost"] + inc["total_cost"],
            }

        def execution_price(bar_open: float, direction: int, is_entry: bool) -> float:
            # 买入更贵、卖出更便宜；滑点通过成交价进入盈亏。
            side = direction if is_entry else -direction
            return float(bar_open) + float(p["slippage"]) * side

        def costs_for_fill(price: float, lots: int) -> dict:
            unit = self._compute_costs(price, p)
            slip = abs(float(p["slippage"]) * float(p["contract_multiplier"]) * lots)
            return {
                "commission": unit["commission"] * lots,
                "slippage_cost": slip,
                "impact_cost": unit["impact_cost"] * lots,
                "total_cost": unit["total_cost"] * lots,
            }

        def close_lots(exit_price: float, exit_date, exit_reason: str, idx: int, lots_to_close: int, signal_date=None, fill_mode: str = ""):
            nonlocal position, entry_price, entry_idx, entry_date, margin_used, cash, entry_costs, add_count, entry_signal_date, entry_fill_mode, entry_sizing
            if position == 0:
                return
            current_lots = abs(position)
            direction = signed_dir()
            close_lots_qty = max(1, min(int(lots_to_close), current_lots))
            gross_pnl = (exit_price - entry_price) * p["contract_multiplier"] * (direction * close_lots_qty)
            exit_costs = costs_for_fill(exit_price, close_lots_qty)
            exit_fee = exit_costs["total_cost"]
            ratio = close_lots_qty / current_lots if current_lots else 1.0
            allocated_entry_costs = {
                "commission": entry_costs["commission"] * ratio,
                "slippage_cost": entry_costs["slippage_cost"] * ratio,
                "impact_cost": entry_costs["impact_cost"] * ratio,
                "total_cost": entry_costs["total_cost"] * ratio,
            }
            total_fee = allocated_entry_costs["total_cost"] + exit_fee
            total_slippage = allocated_entry_costs["slippage_cost"] + exit_costs["slippage_cost"]
            total_impact = allocated_entry_costs["impact_cost"] + exit_costs["impact_cost"]
            net_pnl = gross_pnl - total_fee
            released_margin = margin_used * ratio
            cash += gross_pnl - exit_fee + released_margin
            trades.append({
                "entry_date": entry_date,
                "entry_signal_date": entry_signal_date,
                "entry_fill_mode": entry_fill_mode or "next_open",
                "entry_price": entry_price,
                "exit_date": exit_date,
                "exit_signal_date": signal_date if signal_date is not None else exit_date,
                "exit_fill_mode": fill_mode or ("stop_intrabar" if "stop" in exit_reason else "next_open"),
                "exit_price": exit_price,
                "direction": "long" if direction == 1 else "short",
                "position_size": close_lots_qty,
                "position_size_before_exit": current_lots,
                "add_count": add_count,
                "sizing_model": entry_sizing.get("sizing_model", "fixed"),
                "risk_amount": entry_sizing.get("risk_amount", 0.0),
                "risk_per_lot": entry_sizing.get("risk_per_lot", 0.0),
                "raw_risk_lots": entry_sizing.get("raw_risk_lots", 0),
                "margin_lots": entry_sizing.get("margin_lots", 0),
                "exit_reason": exit_reason,
                "bars_held": idx - entry_idx,
                "gross_pnl": gross_pnl,
                "fee": total_fee,
                "entry_commission": allocated_entry_costs["commission"],
                "exit_commission": exit_costs["commission"],
                "commission": allocated_entry_costs["commission"] + exit_costs["commission"],
                "slippage_cost": total_slippage,
                "impact_cost": total_impact,
                "total_cost": total_fee,
                "economic_cost": total_fee + total_slippage + total_impact,
                "net_pnl": net_pnl,
            })
            remaining_lots = current_lots - close_lots_qty
            entry_costs = {
                "commission": entry_costs["commission"] - allocated_entry_costs["commission"],
                "slippage_cost": entry_costs["slippage_cost"] - allocated_entry_costs["slippage_cost"],
                "impact_cost": entry_costs["impact_cost"] - allocated_entry_costs["impact_cost"],
                "total_cost": entry_costs["total_cost"] - allocated_entry_costs["total_cost"],
            }
            margin_used -= released_margin
            if remaining_lots <= 0:
                position = 0
                entry_price = 0.0
                entry_idx = 0
                entry_date = None
                margin_used = 0.0
                add_count = 0
                entry_signal_date = None
                entry_fill_mode = ""
                entry_costs = {"commission": 0.0, "slippage_cost": 0.0, "impact_cost": 0.0, "total_cost": 0.0}
                entry_sizing = {
                    "sizing_model": str(p.get("position_sizing", "fixed")),
                    "risk_amount": 0.0,
                    "risk_per_lot": 0.0,
                    "raw_risk_lots": 0,
                    "margin_lots": 0,
                }
            else:
                position = direction * remaining_lots

        def close_position(exit_price: float, exit_date, exit_reason: str, idx: int, signal_date=None, fill_mode: str = ""):
            if position == 0:
                return
            close_lots(exit_price, exit_date, exit_reason, idx, abs(position), signal_date=signal_date, fill_mode=fill_mode)

        def open_or_add(direction: int, bar, idx: int, signal_date=None, fill_mode: str = "next_open") -> bool:
            nonlocal position, entry_price, entry_idx, entry_date, margin_used, cash, entry_costs, trade_extreme, stop_price, add_count, entry_signal_date, entry_fill_mode, entry_sizing
            lots = abs(position)
            if lots >= int(p["max_position"]):
                return False
            fill = execution_price(bar["open"], direction, True)
            sizing_info = self._target_entry_lots(fill, cash, p) if position == 0 else {
                "lots": 1,
                "sizing_model": "add",
                "risk_amount": 0.0,
                "risk_per_lot": 0.0,
                "raw_risk_lots": 0,
                "margin_lots": 0,
            }
            order_lots = min(int(sizing_info.get("lots", 0) or 0), int(p["max_position"]) - lots)
            if order_lots <= 0:
                diagnostics["orders_rejected_position_sizing"] += 1
                return False
            volume = float(bar.get("volume", 0) or 0)
            max_participation = float(p.get("max_volume_participation", 0) or 0)
            if max_participation > 0 and volume > 0 and (lots + order_lots) > volume * max_participation:
                diagnostics["orders_rejected_volume_participation"] += 1
                return False
            required_margin = self._margin_required(fill, p) * order_lots
            fill_costs = costs_for_fill(fill, order_lots)
            total_cash_need = required_margin + fill_costs["total_cost"]
            if cash < total_cash_need:
                diagnostics["orders_rejected_insufficient_cash"] += 1
                return False
            cash -= total_cash_need
            margin_used += required_margin
            if position == 0:
                position = direction * order_lots
                entry_price = fill
                entry_idx = idx
                entry_date = bar.get("date", idx)
                entry_signal_date = signal_date if signal_date is not None else entry_date
                entry_fill_mode = fill_mode
                entry_sizing = sizing_info
                trade_extreme = bar["high"] if direction == 1 else bar["low"]
                stop_price = float("nan")
                diagnostics["entries_opened"] += 1
            else:
                entry_price = (entry_price * lots + fill * order_lots) / (lots + order_lots)
                position += direction * order_lots
                add_count += 1
                diagnostics["add_entries_opened"] += 1
            entry_costs = add_costs(entry_costs, fill_costs)
            return True

        for i in range(n):
            bar = df.iloc[i]

            if pd.isna(bar["close"]) or pd.isna(bar["open"]):
                diagnostics["bars_skipped_missing_price"] += 1
                equity_curve.append({"date": bar.get("date", i), "equity": cash + margin_used, "cash": cash, "margin": margin_used, "position": position})
                continue
            if bar.get("volume", 0) < p["min_volume"]:
                diagnostics["bars_skipped_low_volume"] += 1
                equity_curve.append({"date": bar.get("date", i), "equity": cash + margin_used, "cash": cash, "margin": margin_used, "position": position})
                continue

            limit_hit = False
            if p["price_limit_pct"] > 0 and prev_close is not None:
                limit_up = prev_close * (1 + p["price_limit_pct"])
                limit_down = prev_close * (1 - p["price_limit_pct"])
                if bar["close"] >= limit_up * 0.999 or bar["close"] <= limit_down * 1.001:
                    limit_hit = True
                    if i > 0 and (df.at[i - 1, "signal_raw"] != 0 or df.at[i - 1, "exit_signal"] != 0):
                        diagnostics["orders_blocked_price_limit"] += 1

            handled_prev_signal = False
            if i > 0 and position != 0 and not limit_hit:
                prev_exit = int(df.at[i - 1, "exit_signal"])
                prev_signal = int(df.at[i - 1, "signal_raw"])
                direction = signed_dir()
                if prev_exit != 0:
                    close_position(
                        execution_price(bar["open"], direction, False),
                        bar["date"],
                        "strategy_exit",
                        i,
                        signal_date=df.at[i - 1, "date"],
                        fill_mode="next_open",
                    )
                    diagnostics["strategy_exits"] += 1
                    handled_prev_signal = True
                elif prev_signal != 0 and prev_signal != direction:
                    close_position(
                        execution_price(bar["open"], direction, False),
                        bar["date"],
                        "signal_reversal",
                        i,
                        signal_date=df.at[i - 1, "date"],
                        fill_mode="next_open",
                    )
                    diagnostics["signal_reversals"] += 1
                    open_or_add(prev_signal, bar, i, signal_date=df.at[i - 1, "date"], fill_mode="next_open")
                    handled_prev_signal = True

            if p["exit_mode"] == "stop_loss" and position != 0:
                direction = signed_dir()
                bars_held = i - entry_idx
                k = max(1.0 - p["decay_step"] * bars_held, p["K_min"])
                custom_stop = None
                if "strategy_stop_price" in df.columns:
                    value = pd.to_numeric(pd.Series([bar.get("strategy_stop_price", np.nan)]), errors="coerce").iloc[0]
                    if pd.notna(value) and float(value) > 0:
                        custom_stop = float(value)
                if direction == 1:
                    trade_extreme = max(trade_extreme, bar["high"])
                    engine_stop = max(entry_price * (1 - p["init_stop"]), trade_extreme * (1 - p["init_stop"] * k))
                    stop_price = custom_stop if custom_stop is not None else engine_stop
                    if bar["low"] <= stop_price:
                        reason = "strategy_trailing_stop" if custom_stop is not None else "stop_loss"
                        close_position(stop_price, bar["date"], reason, i, signal_date=bar["date"], fill_mode="stop_intrabar")
                        if custom_stop is not None:
                            diagnostics["strategy_stop_exits"] += 1
                else:
                    trade_extreme = min(trade_extreme, bar["low"])
                    engine_stop = min(entry_price * (1 + p["init_stop"]), trade_extreme * (1 + p["init_stop"] * k))
                    stop_price = custom_stop if custom_stop is not None else engine_stop
                    if bar["high"] >= stop_price:
                        reason = "strategy_trailing_stop" if custom_stop is not None else "stop_loss"
                        close_position(stop_price, bar["date"], reason, i, signal_date=bar["date"], fill_mode="stop_intrabar")
                        if custom_stop is not None:
                            diagnostics["strategy_stop_exits"] += 1
                df.at[i, "stop_price"] = stop_price

            if position != 0:
                direction = signed_dir()
                tp_price = None
                tp_fraction = float(bar.get("strategy_take_profit_fraction", np.nan)) if "strategy_take_profit_fraction" in df.columns else np.nan
                if "strategy_take_profit_price" in df.columns:
                    value = pd.to_numeric(pd.Series([bar.get("strategy_take_profit_price", np.nan)]), errors="coerce").iloc[0]
                    if pd.notna(value) and float(value) > 0:
                        tp_price = float(value)
                tp_fraction = float(tp_fraction) if pd.notna(tp_fraction) else 0.0
                if tp_price is not None and tp_fraction > 0 and position != 0:
                    touched = (direction == 1 and float(bar["high"]) >= tp_price) or (direction == -1 and float(bar["low"]) <= tp_price)
                    if touched:
                        current_lots = abs(position)
                        lots_to_close = max(1, int(np.ceil(current_lots * min(tp_fraction, 1.0))))
                        close_lots(tp_price, bar["date"], "strategy_take_profit", i, lots_to_close, signal_date=bar["date"], fill_mode="take_profit_intrabar")
                        diagnostics["strategy_take_profit_exits"] += 1

            if i > 0 and not limit_hit and not handled_prev_signal:
                prev_signal = int(df.at[i - 1, "signal_raw"])
                if prev_signal != 0:
                    direction = signed_dir()
                    if position == 0:
                        open_or_add(prev_signal, bar, i, signal_date=df.at[i - 1, "date"], fill_mode="next_open")
                    elif prev_signal == direction:
                        open_or_add(prev_signal, bar, i, signal_date=df.at[i - 1, "date"], fill_mode="next_open")

            df.at[i, "position"] = position
            if position != 0:
                df.at[i, "entry_price"] = entry_price

            direction = signed_dir()
            lots = abs(position)
            if direction == 1:
                unrealized = (bar["close"] - entry_price) * p["contract_multiplier"] * lots
            elif direction == -1:
                unrealized = (entry_price - bar["close"]) * p["contract_multiplier"] * lots
            else:
                unrealized = 0.0
            equity_curve.append({"date": bar.get("date", i), "equity": cash + margin_used + unrealized, "cash": cash, "margin": margin_used, "position": position})
            prev_close = bar["close"]

        if position != 0:
            last = df.iloc[-1]
            close_position(float(last["close"]), last.get("date", n - 1), "final_close", n - 1, signal_date=last.get("date", n - 1), fill_mode="final_close")
            diagnostics["forced_final_close"] = 1
            if equity_curve:
                equity_curve[-1]["equity"] = cash
                equity_curve[-1]["cash"] = cash
                equity_curve[-1]["margin"] = 0.0
                equity_curve[-1]["position"] = 0

        df["stop_price"] = df["stop_price"].fillna(0)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curve)
        stats = self._compute_stats(trades_df, equity_df)
        stats["diagnostics"] = diagnostics
        stats["execution_policy"] = {
            "signal_timing": "策略在第 t 根K线收盘后产生 signal_raw / exit_signal",
            "entry_fill": "第 t+1 根K线开盘价成交，并按方向叠加滑点",
            "exit_fill": "反向信号/主动离场按下一根开盘价成交；策略止损与分批止盈按当根触及目标价成交；若同一根K线同时触及止损与止盈，当前实现先执行止损",
            "slippage": p["slippage"],
            "commission_rate": p["commission_rate"],
            "fee_mode": p["fee_mode"],
            "margin_rate": p["margin_rate"],
            "max_volume_participation": p["max_volume_participation"],
            "position_sizing": p["position_sizing"],
            "risk_per_trade": p["risk_per_trade"],
            "risk_stop_pct": p["risk_stop_pct"],
            "max_position": p["max_position"],
        }
        return {"bars": df, "trades": trades_df, "equity_curve": equity_df, "stats": stats}

    def _compute_stats(self, trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
        if trades.empty or len(trades) < 2:
            start_equity = equity["equity"].iloc[0] if len(equity) and "equity" in equity.columns else self.initial_capital
            end_equity = equity["equity"].iloc[-1] if len(equity) and "equity" in equity.columns else start_equity
            return {
                "total_trades": len(trades),
                "error": "交易次数不足",
                "start_equity": round(float(start_equity), 2),
                "end_equity": round(float(end_equity), 2),
                "net_profit": round(float(end_equity - start_equity), 2),
                "total_return_pct": round((float(end_equity) / float(start_equity) - 1) * 100, 2) if start_equity else 0,
            }
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] < 0]
        w, l = len(wins), len(losses)
        gp = wins["net_pnl"].sum() if w > 0 else 0
        gl = abs(losses["net_pnl"].sum()) if l > 0 else 0
        peak = equity["equity"].cummax()
        dd = equity["equity"] - peak
        md = dd.min()
        md_pct = (md / peak.max()) * 100 if peak.max() > 0 else 0
        dr = equity["equity"].pct_change().dropna()
        annual_volatility = dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        sharpe = (dr.mean() * 252) / annual_volatility if annual_volatility > 0 else 0
        start_equity = equity["equity"].iloc[0] if len(equity) else self.initial_capital
        end_equity = equity["equity"].iloc[-1] if len(equity) else start_equity
        total_return_pct = (end_equity / start_equity - 1) * 100 if start_equity else 0
        max_consecutive_losses = 0
        current_losses = 0
        for pnl in trades["net_pnl"].fillna(0):
            if pnl < 0:
                current_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_losses = 0
        return {
            "total_trades": len(trades), "win_count": w, "loss_count": l,
            "win_rate": round(w / len(trades), 4) if len(trades) > 0 else 0,
            "net_profit": round(trades["net_pnl"].sum(), 2),
            "total_return_pct": round(total_return_pct, 2),
            "annual_volatility_pct": round(annual_volatility * 100, 2),
            "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
            "max_drawdown": round(md, 2), "max_drawdown_pct": round(md_pct, 2),
            "return_drawdown_ratio": round(abs(total_return_pct / md_pct), 2) if md_pct else 0,
            "avg_win": round(wins["net_pnl"].mean(), 2) if w > 0 else 0,
            "avg_loss": round(losses["net_pnl"].mean(), 2) if l > 0 else 0,
            "payoff_ratio": round(abs(wins["net_pnl"].mean() / losses["net_pnl"].mean()), 2) if l > 0 and losses["net_pnl"].mean() != 0 else 0,
            "sharpe_ratio": round(sharpe, 2), "total_fees": round(trades["fee"].sum(), 2),
            "total_commission": round(trades["commission"].sum(), 2) if "commission" in trades.columns else round(trades["fee"].sum(), 2),
            "total_slippage_cost": round(trades["slippage_cost"].sum(), 2) if "slippage_cost" in trades.columns else 0,
            "total_impact_cost": round(trades["impact_cost"].sum(), 2) if "impact_cost" in trades.columns else 0,
            "cost_profit_ratio": round((trades["fee"].sum() / abs(trades["net_pnl"].sum())) * 100, 2) if trades["net_pnl"].sum() else 0,
            "max_consecutive_losses": int(max_consecutive_losses),
            "best_trade": round(trades["net_pnl"].max(), 2),
            "worst_trade": round(trades["net_pnl"].min(), 2),
            "start_equity": round(start_equity, 2),
            "end_equity": round(end_equity, 2),
        }
