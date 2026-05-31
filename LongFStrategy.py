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


class LongFStrategy(IStrategy):
    """
    做多F（vps_lf）

    日线条件：收盘价不在 MA5/MA25 之下（close >= lower_ma）
    信号：收盘同时升破5分钟 MA5 和 MA25，同时自上一次MA5/MA25交叉以来
          的最低收开盘价与当前较大均线的差值 > 800点
          当前K线成交额 500万-1000万 USDC 且实体 < 100点（弱势升破）
    入场：信号K线下一根开盘立即入场
    止损：sig_open - sig_range（sig_range = sig_open - 记录低点）
    止盈：sig_open + sig_range
    扩展：触发止盈时放量阳线(>=3倍20日均量) -> sig_open + 3*sig_range
    """

    INTERFACE_VERSION = 3

    timeframe = '5m'
    can_short: bool = True

    def __init__(self, config: dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self._candle_vol_spike: Dict[str, Dict[str, str]] = {}

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
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ---- 日线MA5/MA25 ----
        daily = dataframe.resample('1D', on='date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna().reset_index()

        daily['ma5'] = ta.SMA(daily, timeperiod=5)
        daily['ma25'] = ta.SMA(daily, timeperiod=25)

        dataframe = merge_informative_pair(dataframe, daily, self.timeframe, '1d', ffill=True)

        # ---- 5分钟均线 ----
        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma25'] = ta.SMA(dataframe, timeperiod=25)
        dataframe['ma_min'] = dataframe[['sma5', 'sma25']].min(axis=1)
        dataframe['ma_max'] = dataframe[['sma5', 'sma25']].max(axis=1)

        # ---- 量能均线 ----
        dataframe['volume_sma20'] = dataframe['volume'].rolling(window=20).mean()

        # ---- 自上次MA交叉以来的最高/最低收开盘价 ----
        cross_state = (dataframe['sma5'] > dataframe['sma25']).astype(int)
        cross_change = cross_state.diff().abs().fillna(0)
        segment = cross_change.cumsum()

        dataframe['oc_max'] = dataframe[['open', 'close']].max(axis=1)
        dataframe['oc_min'] = dataframe[['open', 'close']].min(axis=1)
        dataframe['high_since_cross'] = dataframe.groupby(segment)['oc_max'].cummax()
        dataframe['low_since_cross'] = dataframe.groupby(segment)['oc_min'].cummin()

        # 缓存量能特征
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
        dataframe['enter_short'] = 0
        dataframe['enter_long'] = 0

        # 日线条件：收盘价不在 MA5/MA25 之下
        lower_ma = dataframe[['ma5_1d', 'ma25_1d']].min(axis=1)
        not_below_both = dataframe['close'] >= lower_ma

        # 收盘同时升破 MA5 和 MA25
        above_both_ma = (
            (dataframe['close'] > dataframe['sma5']) &
            (dataframe['close'] > dataframe['sma25'])
        )

        # 自上次交叉以来的最低收开盘价与当前较大均线的差值 > 800
        diff = dataframe['ma_max'] - dataframe['low_since_cross']
        range_ok = diff > 800

        # 当前K线成交额 < 1000万 且 实体 < 100点（弱势升破）
        vol = dataframe['volume'] * dataframe['close']
        weak_candle = (vol < 5_000_000) & ((dataframe['close'] - dataframe['open']).abs() < 100)

        condition = not_below_both & above_both_ma & range_ok & weak_candle

        for idx in condition[condition].index:
            pos = dataframe.index.get_loc(idx)
            ma_max_val = dataframe.iloc[pos]['ma_max']
            recorded_low = dataframe.iloc[pos]['low_since_cross']
            sig_range_val = ma_max_val - recorded_low
            if sig_range_val <= 0:
                continue
            if pos + 1 < len(dataframe):
                entry_idx = dataframe.index[pos + 1]
                if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                    dataframe.loc[entry_idx, 'enter_long'] = 1
                    dataframe.loc[entry_idx, 'enter_tag'] = f'vps_lf:{ma_max_val}:{sig_range_val}'

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # ----- 自定义止损/止盈 -----
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        if trade.enter_tag and trade.enter_tag.startswith('vps_lf'):
            parts = trade.enter_tag.split(':')
            if len(parts) >= 3:
                sig_open = float(parts[1])
                sig_range = float(parts[2])
                # 止损：sig_open - sig_range（多做，价位跌破此位置止损）
                stop_rate = sig_open - sig_range
                return stoploss_from_absolute(stop_rate, current_rate,
                                              trade.is_short, trade.leverage)
        return self.stoploss

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        if not trade.enter_tag or not trade.enter_tag.startswith('vps_lf'):
            return None

        parts = trade.enter_tag.split(':')
        if len(parts) < 3:
            return None
        sig_open = float(parts[1])
        sig_range = float(parts[2])

        # 止盈：sig_open + sig_range
        if current_rate >= sig_open + sig_range:
            pair_key = pair
            ts = current_time.isoformat()
            is_spike = self._candle_vol_spike.get(pair_key, {}).get(ts, '') == 'bull'
            if is_spike:
                if current_rate >= sig_open + 3 * sig_range:
                    return 'take_profit'
                return None
            return 'take_profit'

        return None

    # ----- 记录订单 ----
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        if trade.enter_tag and trade.enter_tag.startswith('vps_lf'):
            parts = trade.enter_tag.split(':')
            if len(parts) >= 3:
                sig_open = float(parts[1])
                sig_range = float(parts[2])
                dir_cn = "做多" if not trade.is_short else "做空"
                open_bt = trade.open_date + timedelta(hours=8)
                close_bt = current_time + timedelta(hours=8)
                profit_pct = ((rate / trade.open_rate) - 1) * (-1 if trade.is_short else 1) * trade.leverage * 100
                print(f"  [成交] {dir_cn}F 信号价={sig_open:.1f} 振幅={sig_range:.1f}点 "
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
