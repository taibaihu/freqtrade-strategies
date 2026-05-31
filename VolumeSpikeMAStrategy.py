# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib


class VolumeSpikeMAStrategy(IStrategy):
    """
    Volume Spike + Daily MA Direction Strategy

    方向过滤（日线MA5/MA25）：
      - 价格在MA之间 -> 可多可空
      - 价格在MA之上 -> 只做多
      - 价格在MA之下 -> 只做空

    做空A（放量阳线，过滤立即触价）：close > open + 日线不在MA之上
	     扩展：触发止盈时若放量阴线(>=3倍20日均量)，止盈扩至 sig_open-2*sig_range
      -> 等待 low <= sig_open -> 入场
      止损 sig_open + sig_range, 止盈 sig_open - sig_range

    做多A（放量阴线，过滤立即触价）：close < open + 日线不在MA之下
      -> 等待 high >= sig_open -> 入场
      止损 sig_open - sig_range, 止盈 sig_open + sig_range

    做空B（放量阴线）：close < open + 日线不在MA之上
	     扩展：触发止盈时若放量阴线(>=3倍20日均量)，止盈扩至 sig_open-2*sig_range
      -> 等待反弹至开盘价(high >= sig_open) -> 入场
      止损 sig_open + sig_range, 止盈 sig_open - sig_range

    做多B（放量阳线，过滤立即触价）：close > open + 日线在MA之上
      -> 等待回踩至开盘价(low <= sig_open) -> 入场
      止损 sig_open - sig_range, 止盈 sig_open + 2*sig_range

    特殊K线分支（实体<100点 + 振幅>=500点）：
      - 阳线用 sig_low 代替 sig_open，sig_high 代替 sig_close
      - 阴线用 sig_high 代替 sig_open，sig_low 代替 sig_close

    量能信号：成交额 >= 1.2亿 USDC
    """

    INTERFACE_VERSION = 3

    timeframe = '5m'
    can_short: bool = True

    # 禁用ROI，完全依靠custom_exit止盈
    minimal_roi = {"0": 100.0}

    # 硬止损底线（实际止损由custom_stoploss控制）
    stoploss = -0.50

    use_custom_stoploss = True
    trailing_stop = False

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count = 200

    # 改用市价单入场，避免回测engine对限价单的clamp
    order_types = {
        "entry": "market",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {
        "entry": "IOC",
        "exit": "GTC",
    }

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return 1.0

    # 缓存每根K线的量能特征（避免回测中self.dp返回全量数据）
    def __init__(self, config: dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self._candle_vol_spike: Dict[str, Dict[str, str]] = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ---- 日线MA5/MA25（通过重采样5m→1d计算） ----
        daily = dataframe.resample('1D', on='date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna().reset_index()

        daily['ma5'] = ta.SMA(daily, timeperiod=5)
        daily['ma25'] = ta.SMA(daily, timeperiod=25)

        # 合并回5m
        dataframe = merge_informative_pair(dataframe, daily, self.timeframe, '1d', ffill=True)

        # ---- 量能均线（用于做空A/B放量扩展止盈）----
        dataframe['volume_sma20'] = dataframe['volume'].rolling(window=20).mean()

        # 缓存每根K线的量能特征：''=无, 'bear'=放量阴线, 'bull'=放量阳线
        pair_key = metadata.get('pair', '')
        if pair_key and self._candle_vol_spike.get(pair_key) is None:
            self._candle_vol_spike[pair_key] = {}
        for _, row in dataframe.iterrows():
            ts = row['date'].isoformat()
            is_bearish = row['close'] < row['open']
            is_high_vol = row['volume'] >= 3 * row.get('volume_sma20', float('inf'))
            candle_type = ''
            if is_high_vol:
                candle_type = 'bear' if is_bearish else 'bull'
            self._candle_vol_spike[pair_key][ts] = candle_type

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ---- 方向过滤（日线MA5/MA25） ----
        higher_ma = dataframe[['ma5_1d', 'ma25_1d']].max(axis=1)
        lower_ma = dataframe[['ma5_1d', 'ma25_1d']].min(axis=1)

        above_both = dataframe['close'] > higher_ma
        below_both = dataframe['close'] < lower_ma

        can_short = ~above_both   # below_both OR between
        can_long = ~below_both    # above_both OR between

        # ---- 量能信号 ----
        sig_range = (dataframe['close'] - dataframe['open']).abs()
        is_spike = (dataframe['volume'] * dataframe['close']) >= 120_000_000

        # ---- 特殊K线（小实体<100点 + 振幅>=500点，用高低价代替开收盘）----
        sig_wick = dataframe['high'] - dataframe['low']
        special_candle = is_spike & (sig_range < 100) & (sig_wick >= 500)

        short_signal = is_spike & (dataframe['close'] > dataframe['open']) & can_short & (sig_range >= 200)
        long_signal = is_spike & (dataframe['close'] < dataframe['open']) & can_long & (sig_range >= 200)

        # 特殊信号单独定义（振幅用wick代替body）
        short_signal_special = special_candle & (dataframe['close'] > dataframe['open']) & can_short
        long_signal_special = special_candle & (dataframe['close'] < dataframe['open']) & can_long

        short_signal = short_signal | short_signal_special
        long_signal = long_signal | long_signal_special

        # 过滤：放量信号K线的后一根K线立即触价的，去掉该信号
        immediate_touch = (
            ((dataframe['low'].shift(-1) <= dataframe['open']) & short_signal) |
            ((dataframe['high'].shift(-1) >= dataframe['open']) & long_signal)
        )
        short_signal = short_signal & ~immediate_touch
        long_signal = long_signal & ~immediate_touch

        # ===== 延迟入场逻辑 =====
        # 存储信号K线的开盘价和振幅（用于后续计算SL/TP）
        dataframe['_sig_open'] = np.nan
        dataframe['_sig_range'] = np.nan
        dataframe['_sig_dir'] = np.nan  # 用NaN而不是0，否则ffill不会传播

        # 普通信号：sig_open=开盘价, sig_range=|close-open|
        dataframe.loc[short_signal & ~short_signal_special, '_sig_open'] = dataframe['open']
        dataframe.loc[short_signal & ~short_signal_special, '_sig_range'] = sig_range
        dataframe.loc[short_signal & ~short_signal_special, '_sig_dir'] = -1

        # 特殊信号（阳线→实体小影线大）：sig_open=最低价, sig_range=high-low
        dataframe.loc[short_signal_special, '_sig_open'] = dataframe['low']
        dataframe.loc[short_signal_special, '_sig_range'] = sig_wick
        dataframe.loc[short_signal_special, '_sig_dir'] = -1

        # 普通做多信号：sig_open=开盘价, sig_range=|close-open|
        dataframe.loc[long_signal & ~long_signal_special, '_sig_open'] = dataframe['open']
        dataframe.loc[long_signal & ~long_signal_special, '_sig_range'] = sig_range
        dataframe.loc[long_signal & ~long_signal_special, '_sig_dir'] = 1

        # 特殊做多信号（阴线→实体小影线大）：sig_open=最高价, sig_range=high-low
        dataframe.loc[long_signal_special, '_sig_open'] = dataframe['high']
        dataframe.loc[long_signal_special, '_sig_range'] = sig_wick
        dataframe.loc[long_signal_special, '_sig_dir'] = 1

        # 向前填充信号，最多120根K线（10小时），覆盖隔夜信号
        dataframe['_sig_open'] = dataframe['_sig_open'].ffill(limit=120)
        dataframe['_sig_range'] = dataframe['_sig_range'].ffill(limit=120)
        dataframe['_sig_dir'] = dataframe['_sig_dir'].ffill(limit=120)

        # 标记每个独立信号（每个原始信号作为一个session）
        signal_start = short_signal | long_signal
        session = signal_start.cumsum()
        session = session.where(dataframe['_sig_dir'].notna(), 0)

        # 触价检测（排除信号K线自身：阳线open=low会导致误触）
        # 做空：价格下探到信号K线开盘价（low <= sig_open）
        touch_short = (
            (dataframe['_sig_dir'] == -1) & dataframe['_sig_open'].notna() &
            (dataframe['low'] <= dataframe['_sig_open']) &
            ~signal_start
        )
        touch_long = (
            (dataframe['_sig_dir'] == 1) & dataframe['_sig_open'].notna() &
            (dataframe['high'] >= dataframe['_sig_open']) &
            ~signal_start
        )

        # 每个session只取第一次触价，避免反复入场
        first_touch_short = pd.Series(False, index=dataframe.index)
        first_touch_long = pd.Series(False, index=dataframe.index)

        if session.max() > 0:
            cum_touch_short = touch_short.groupby(session).cumsum()
            first_touch_short = (cum_touch_short == 1) & touch_short & (session != 0)

            cum_touch_long = touch_long.groupby(session).cumsum()
            first_touch_long = (cum_touch_long == 1) & touch_long & (session != 0)

        # ---- 编码enter_tag ----
        # 格式: vps_{direction}:{signal_open}:{sig_range}
        if first_touch_short.any():
            idx_s = dataframe.loc[first_touch_short].index
            dataframe.loc[idx_s, 'enter_short'] = 1
            sig_open_vals = dataframe.loc[idx_s, '_sig_open']
            sig_range_vals = dataframe.loc[idx_s, '_sig_range']
            dataframe.loc[idx_s, 'enter_tag'] = (
                'vps_s:' + sig_open_vals.astype(str) + ':' + sig_range_vals.astype(str))

        if first_touch_long.any():
            idx_l = dataframe.loc[first_touch_long].index
            dataframe.loc[idx_l, 'enter_long'] = 1
            sig_open_vals = dataframe.loc[idx_l, '_sig_open']
            sig_range_vals = dataframe.loc[idx_l, '_sig_range']
            dataframe.loc[idx_l, 'enter_tag'] = (
                'vps_l:' + sig_open_vals.astype(str) + ':' + sig_range_vals.astype(str))

        # 清理辅助列
        dataframe.drop(columns=['_sig_open', '_sig_range', '_sig_dir', 'session'],
                       inplace=True, errors='ignore')

        # 确保enter列存在（vps_s/vps_l可能未创建）
        if 'enter_short' not in dataframe.columns:
            dataframe['enter_short'] = 0
        if 'enter_long' not in dataframe.columns:
            dataframe['enter_long'] = 0

        # ===== 放量阴线做空（反弹至开盘价入场）=====
        bear_short = is_spike & (dataframe['close'] < dataframe['open']) & can_short & (sig_range >= 200)
        bear_short_special = special_candle & (dataframe['close'] < dataframe['open']) & can_short

        for idx in bear_short[bear_short].index:
            pos = dataframe.index.get_loc(idx)
            sig_open = dataframe.iloc[pos]['open']
            sig_range_val = abs(dataframe.iloc[pos]['close'] - sig_open)

            for look_pos in range(pos + 1, min(pos + 121, len(dataframe))):
                if dataframe.iloc[look_pos]['high'] >= sig_open:
                    entry_idx = dataframe.index[look_pos]
                    if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                        dataframe.loc[entry_idx, 'enter_short'] = 1
                        dataframe.loc[entry_idx, 'enter_tag'] = (
                            f'vps_sb:{sig_open}:{sig_range_val}')
                    break

        # ===== 特殊做空B（小实体阴线→用最高价代替开盘价）=====
        for idx in bear_short_special[bear_short_special].index:
            pos = dataframe.index.get_loc(idx)
            sig_open = dataframe.iloc[pos]['high']   # sig_high
            sig_range_val = dataframe.iloc[pos]['high'] - dataframe.iloc[pos]['low']  # high-low

            for look_pos in range(pos + 1, min(pos + 121, len(dataframe))):
                if dataframe.iloc[look_pos]['high'] >= sig_open:
                    entry_idx = dataframe.index[look_pos]
                    if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                        dataframe.loc[entry_idx, 'enter_short'] = 1
                        dataframe.loc[entry_idx, 'enter_tag'] = (
                            f'vps_sb:{sig_open}:{sig_range_val}')
                    break

        # ===== 放量阳线做多（回踩至开盘价入场，过滤立即触价，日线在MA之上）=====
        bull_long = is_spike & (dataframe['close'] > dataframe['open']) & above_both & (sig_range >= 200)
        immediate = (dataframe['low'].shift(-1) <= dataframe['open']) & bull_long
        bull_long = bull_long & ~immediate

        for idx in bull_long[bull_long].index:
            pos = dataframe.index.get_loc(idx)
            sig_open = dataframe.iloc[pos]['open']
            sig_close = dataframe.iloc[pos]['close']
            sig_range_val = abs(sig_close - sig_open)

            for look_pos in range(pos + 1, min(pos + 121, len(dataframe))):
                if dataframe.iloc[look_pos]['low'] <= sig_open:
                    entry_idx = dataframe.index[look_pos]
                    if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                        dataframe.loc[entry_idx, 'enter_long'] = 1
                        dataframe.loc[entry_idx, 'enter_tag'] = (
                            f'vps_lb:{sig_open}:{sig_range_val}')
                    break

        # ===== 特殊做多B（小实体阳线→用最低价代替开盘价，过滤立即触价，日线在MA之上）=====
        bull_long_special = special_candle & (dataframe['close'] > dataframe['open']) & above_both
        immediate_special = (dataframe['low'].shift(-1) <= dataframe['low']) & bull_long_special
        bull_long_special = bull_long_special & ~immediate_special

        for idx in bull_long_special[bull_long_special].index:
            pos = dataframe.index.get_loc(idx)
            sig_open = dataframe.iloc[pos]['low']    # sig_low
            sig_close = dataframe.iloc[pos]['high']  # sig_high
            sig_range_val = sig_close - sig_open     # high - low

            for look_pos in range(pos + 1, min(pos + 121, len(dataframe))):
                if dataframe.iloc[look_pos]['low'] <= sig_open:
                    entry_idx = dataframe.index[look_pos]
                    if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                        dataframe.loc[entry_idx, 'enter_long'] = 1
                        dataframe.loc[entry_idx, 'enter_tag'] = (
                            f'vps_lb:{sig_open}:{sig_range_val}')
                    break

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # ----- 自定义止损 -----
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        if trade.enter_tag and trade.enter_tag.startswith('vps_'):
            parts = trade.enter_tag.split(':')
            if len(parts) >= 3:
                sig_open = float(parts[1])
                sig_range = float(parts[2])
                if trade.is_short:
                    stop_rate = sig_open + sig_range
                else:
                    stop_rate = sig_open - sig_range
                return stoploss_from_absolute(stop_rate, current_rate,
                                              trade.is_short, trade.leverage)
        return self.stoploss

    # ----- 自定义止盈 ----
    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        if trade.enter_tag and trade.enter_tag.startswith('vps_'):
            parts = trade.enter_tag.split(':')
            if len(parts) >= 3:
                sig_open = float(parts[1])
                sig_range = float(parts[2])
                if trade.enter_tag.startswith('vps_lb'):
                    # 做多B：止盈 sig_close + sig_range = sig_open + 2*sig_range
                    if current_rate >= sig_open + 2 * sig_range:
                        return 'take_profit'
                elif trade.enter_tag.startswith('vps_l'):
                    # 做多A：止盈 sig_open + sig_range
                    if current_rate >= sig_open + sig_range:
                        return 'take_profit'
                elif trade.is_short:
                    # 做空A/B：止盈 sig_open - sig_range
                    if current_rate <= sig_open - sig_range:
                        # 检查当前K线是否为放量阴线（量能>=3倍20日均量）
                        pair_key = pair
                        ts = current_time.isoformat()
                        is_spike = self._candle_vol_spike.get(pair_key, {}).get(ts, '') == 'bear'
                        if is_spike:
                            # 扩展止盈至 sig_open - 2*sig_range
                            if current_rate <= sig_open - 2 * sig_range:
                                return 'take_profit'
                            return None
                        return 'take_profit'
        return None

    # ----- 记录所有订单的出入场时间（北京时间）-----
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        if trade.enter_tag and trade.enter_tag.startswith('vps_'):
            parts = trade.enter_tag.split(':')
            if len(parts) >= 3:
                sig_open = float(parts[1])
                sig_range = float(parts[2])
                dir_cn = "做空" if trade.is_short else "做多"
                open_bt = trade.open_date + timedelta(hours=8)
                close_bt = current_time + timedelta(hours=8)
                profit_pct = ((rate / trade.open_rate) - 1) * (-1 if trade.is_short else 1) * trade.leverage * 100
                print(f"  [成交] {dir_cn} 信号开盘={sig_open:.1f} 振幅={sig_range:.1f}点 "
                      f"开仓={open_bt.strftime('%m-%d %H:%M')} "
                      f"平仓={close_bt.strftime('%m-%d %H:%M')} "
                      f"入场={trade.open_rate:.1f} 出场={rate:.1f} "
                      f"盈亏={profit_pct:.2f}% 原因={exit_reason}")
        return True

    # ----- 自定义仓位：每单0.01 BTC -----
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float],
                            max_stake: float, leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        target_margin = 0.01 * current_rate / leverage
        if min_stake is not None:
            target_margin = max(target_margin, min_stake)
        target_margin = min(target_margin, max_stake)
        return target_margin
