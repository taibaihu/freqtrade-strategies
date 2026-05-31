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


class SDOnlyStrategy(IStrategy):
    """
    做空D + 做多L独立策略

    做空D：日线不在MA5/MA25之下，5分钟实阴线跌破MA5/MA25，成交额7000万-9000万，实体>=250
           -> 入场：信号K线下一根开盘，止损 sig_open，止盈 sig_open-2*sig_range
           -> 扩展：触发止盈时若放量阴线(>=3倍20日均量)，止盈扩至 sig_open-3*sig_range

    做多L：日线在MA5/MA25之下，5分钟实阳线突破MA5/MA25，成交额7000万-9000万，实体>=250
           -> 入场：信号K线下一根开盘，止损 sig_open，止盈 sig_close+sig_range（即 sig_open+2*sig_range）
           -> 扩展：触发止盈时若放量阳线(>=3倍20日均量)，止盈扩至 sig_open+3*sig_range
    """

    INTERFACE_VERSION = 3

    timeframe = '5m'
    can_short: bool = True

    # 缓存每根K线的量能特征（避免回测中self.dp返回全量数据）
    def __init__(self, config: dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self._candle_vol_spike: Dict[str, Dict[str, bool]] = {}

    minimal_roi = {"0": 100.0}
    stoploss = -0.50

    use_custom_stoploss = True
    trailing_stop = False

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count = 200

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

        # ---- 5分钟均线 ----
        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma25'] = ta.SMA(dataframe, timeperiod=25)

        # ---- 量能均线 ----
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
        # ---- 方向过滤 ----
        higher_ma = dataframe[['ma5_1d', 'ma25_1d']].max(axis=1)
        lower_ma = dataframe[['ma5_1d', 'ma25_1d']].min(axis=1)

        above_both = dataframe['close'] > higher_ma      # 做多方向（日线在MA5/MA25之上）
        between = (dataframe['close'] > lower_ma) & (dataframe['close'] < higher_ma)  # 做空方向

        # 确保enter列存在
        dataframe['enter_short'] = 0
        dataframe['enter_long'] = 0

        # ---- 做空D：日线不在MA5/MA25之下，5分钟实阴线跌破MA5/MA25 ----
        not_below_both = dataframe['close'] >= lower_ma
        sd_condition = (
            not_below_both &
            ((dataframe['volume'] * dataframe['close']) >= 70_000_000) &
            ((dataframe['volume'] * dataframe['close']) < 90_000_000) &
            (dataframe['close'] < dataframe['open']) &
            ((dataframe['open'] - dataframe['close']) >= 250) &
            (dataframe['close'] < dataframe['sma5']) &
            (dataframe['close'] < dataframe['sma25'])
        )

        for idx in sd_condition[sd_condition].index:
            pos = dataframe.index.get_loc(idx)
            sig_open_val = max(dataframe.iloc[pos]['sma5'], dataframe.iloc[pos]['sma25'])
            sig_close_val = dataframe.iloc[pos]['close']
            sig_range_val = sig_open_val - sig_close_val

            if sig_range_val <= 0:
                continue

            # 信号K线出现后立即入场（下一根K线开盘入场）
            if pos + 1 < len(dataframe):
                entry_idx = dataframe.index[pos + 1]
                if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                    dataframe.loc[entry_idx, 'enter_short'] = 1
                    dataframe.loc[entry_idx, 'enter_tag'] = (
                        f'vps_sd:{sig_open_val}:{sig_range_val}')

        # ---- 做多L：日线在MA5/MA25之下，5分钟实阳线向上突破MA5/MA25 ----
        below_both = dataframe['close'] < lower_ma
        sl_condition = (
            below_both &
            ((dataframe['volume'] * dataframe['close']) >= 70_000_000) &
            ((dataframe['volume'] * dataframe['close']) < 90_000_000) &
            (dataframe['close'] > dataframe['open']) &
            ((dataframe['close'] - dataframe['open']) >= 250) &
            (dataframe['close'] > dataframe['sma5']) &
            (dataframe['close'] > dataframe['sma25'])
        )

        for idx in sl_condition[sl_condition].index:
            pos = dataframe.index.get_loc(idx)
            sig_open_val = min(dataframe.iloc[pos]['sma5'], dataframe.iloc[pos]['sma25'])
            sig_close_val = dataframe.iloc[pos]['close']
            sig_range_val = sig_close_val - sig_open_val

            if sig_range_val <= 0:
                continue

            # 信号K线出现后立即入场（下一根K线开盘入场）
            if pos + 1 < len(dataframe):
                entry_idx = dataframe.index[pos + 1]
                if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                    dataframe.loc[entry_idx, 'enter_long'] = 1
                    dataframe.loc[entry_idx, 'enter_tag'] = (
                        f'vps_sl:{sig_open_val}:{sig_range_val}')

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
                if trade.enter_tag.startswith('vps_sd') or trade.enter_tag.startswith('vps_sl'):
                    # 做空D/做多L：止损在 sig_open（均线突破位）
                    return stoploss_from_absolute(sig_open, current_rate,
                                                  trade.is_short, trade.leverage)
                elif trade.is_short:
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
                if trade.enter_tag.startswith('vps_sd'):
                    # 做空D：止盈 sig_open - 2*sig_range
                    if current_rate <= sig_open - 2 * sig_range:
                        # 检查当前K线是否为放量阴线（量能>=3倍20日均量）
                        pair_key = pair
                        ts = current_time.isoformat()
                        is_spike = self._candle_vol_spike.get(pair_key, {}).get(ts, '') == 'bear'
                        if is_spike:
                            # 扩展止盈至 sig_open - 3*sig_range
                            if current_rate <= sig_open - 3 * sig_range:
                                return 'take_profit'
                            return None
                        return 'take_profit'
                elif trade.enter_tag.startswith('vps_sl'):
                    # 做多L：止盈 sig_close + sig_range（即 sig_open + 2*sig_range）
                    if current_rate >= sig_open + 2 * sig_range:
                        # 检查当前K线是否为放量阳线（量能>=3倍20日均量）
                        pair_key = pair
                        ts = current_time.isoformat()
                        is_spike = self._candle_vol_spike.get(pair_key, {}).get(ts, '') == 'bull'
                        if is_spike:
                            # 扩展止盈至 sig_open + 3*sig_range
                            if current_rate >= sig_open + 3 * sig_range:
                                return 'take_profit'
                            return None
                        return 'take_profit'
        return None

    # ----- 记录订单 ----
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
