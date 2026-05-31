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


class CombinedStrategy(IStrategy):
    """
    VolumeSpikeMAStrategy + SDOnlyStrategy 合并策略

    === VolumeSpikeMA 分支（成交额 >= 1.2亿，延迟触价入场）===

    做空A（vps_s）：放量阳线+日线不在MA之上 -> 回踩开盘价入场
      止损 sig_open+sig_range, 止盈 sig_open-sig_range
      扩展：触发止盈时放量阴线(>=3倍20日均量) -> sig_open-2*sig_range

    做多A（vps_l）：放量阴线+日线不在MA之下 -> 反弹开盘价入场
      止损 sig_open-sig_range, 止盈 sig_open+sig_range

    做空B（vps_sb）：放量阴线+日线不在MA之上 -> 反弹开盘价入场
      止损 sig_open+sig_range, 止盈 sig_open-sig_range
      扩展：触发止盈时放量阴线(>=3倍20日均量) -> sig_open-2*sig_range

    做多B（vps_lb）：放量阳线+日线不在MA之下 -> 回踩开盘价入场
      止损 sig_open-sig_range, 止盈 sig_open+2*sig_range

    === SDOnly 分支（成交额 7000万-9000万，立即入场）===

    做空D（vps_sd）：日线不在MA/MA25之下 + 实阴线跌破MA5/MA25
      止损 sig_open, 止盈 sig_open-2*sig_range
      扩展：触发止盈时放量阴线 -> sig_open-3*sig_range

    做多D（vps_sl）：日线在MA5/MA25之下 + 实阳线突破MA5/MA25
      止损 sig_open, 止盈 sig_open+2*sig_range
      扩展：触发止盈时放量阳线 -> sig_open+3*sig_range

    === F 分支（成交额 < 1000万/500万-1000万，弱势突破，立即入场）===

    做空F（vps_sf）：日线不在MA之下 + 跌破双均线，high_since_cross-ma_min>600
      止损 sig_open+sig_range, 止盈 sig_open-sig_range
      扩展：触发止盈时放量阴线 -> sig_open-3*sig_range

    做多F（vps_lf）：日线不在MA之下 + 升破双均线，ema_max-low_since_cross>600
      止损 sig_open-sig_range, 止盈 sig_open+sig_range
      扩展：触发止盈时放量阳线 -> sig_open+3*sig_range
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
        "entry": "limit",
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
        return 100.0

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

        # shift(1)让日线数据滞后一天，避免用盘中的未完成日线来做判断
        daily['date'] = daily['date'] + pd.Timedelta(days=1)

        dataframe = merge_informative_pair(dataframe, daily, self.timeframe, '1d', ffill=True)

        # ---- 5分钟EMA5/EMA25（替代原SMA，用于所有入场条件和止损）----
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema25'] = ta.EMA(dataframe, timeperiod=25)
        dataframe['ema_min'] = dataframe[['ema5', 'ema25']].min(axis=1)
        dataframe['ema_max'] = dataframe[['ema5', 'ema25']].max(axis=1)

        # ---- 量能均线（用于放量扩展止盈）----
        dataframe['volume_sma20'] = dataframe['volume'].rolling(window=20).mean()

        # ---- 自上次EMA交叉以来的最高/最低收开盘价（用于F分支）----
        cross_state = (dataframe['ema5'] > dataframe['ema25']).astype(int)
        cross_change = cross_state.diff().abs().fillna(0)
        segment = cross_change.cumsum()
        dataframe['oc_max'] = dataframe[['open', 'close']].max(axis=1)
        dataframe['oc_min'] = dataframe[['open', 'close']].min(axis=1)
        dataframe['high_since_cross'] = dataframe.groupby(segment)['oc_max'].cummax()
        dataframe['low_since_cross'] = dataframe.groupby(segment)['oc_min'].cummin()

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
        pair_name = metadata.get('pair', 'unknown')

        # ---- 方向过滤（日线MA5/MA25） ----
        higher_ma = dataframe[['ma5_1d', 'ma25_1d']].max(axis=1)
        lower_ma = dataframe[['ma5_1d', 'ma25_1d']].min(axis=1)

        above_both = dataframe['close'] > higher_ma
        below_both = dataframe['close'] < lower_ma
        not_below_both = dataframe['close'] >= lower_ma
        between_both = (dataframe['close'] >= lower_ma) & (dataframe['close'] <= higher_ma)

        can_short = ~above_both   # below_both OR between（日线不在均线之上）
        can_long = ~below_both    # above_both OR between（日线不在均线之下）

        # 确保enter列存在
        dataframe['enter_short'] = 0
        dataframe['enter_long'] = 0

        # ================================================================
        # 第一部分：VolumeSpikeMA 分支（成交额 >= 1.2亿，延迟触价入场）
        # ================================================================
        sig_range = (dataframe['close'] - dataframe['open']).abs()
        is_spike = (dataframe['volume'] * dataframe['close']) >= 120_000_000

        # 特殊K线（小实体<100点 + 振幅>=500点）
        sig_wick = dataframe['high'] - dataframe['low']
        special_candle = is_spike & (sig_range < 100) & (sig_wick >= 500)

        # ---- 做空A ----
        short_signal = is_spike & (dataframe['close'] > dataframe['open']) & can_short & (sig_range >= 250)
        short_signal_special = special_candle & (dataframe['close'] > dataframe['open']) & can_short
        short_signal = short_signal | short_signal_special

        # ---- 做多A ----
        long_signal = is_spike & (dataframe['close'] < dataframe['open']) & can_long & (sig_range >= 250)
        long_signal_special = special_candle & (dataframe['close'] < dataframe['open']) & can_long
        long_signal = long_signal | long_signal_special

        # 过滤立即触价
        immediate_touch = (
            ((dataframe['close'].shift(-1) <= dataframe['open']) & short_signal) |
            ((dataframe['close'].shift(-1) >= dataframe['open']) & long_signal)
        )
        short_signal = short_signal & ~immediate_touch
        long_signal = long_signal & ~immediate_touch

        # 延迟入场逻辑
        dataframe['_sig_open'] = np.nan
        dataframe['_sig_range'] = np.nan
        dataframe['_sig_dir'] = np.nan

        # 做空A信号
        dataframe.loc[short_signal & ~short_signal_special, '_sig_open'] = dataframe['open']
        dataframe.loc[short_signal & ~short_signal_special, '_sig_range'] = sig_range
        dataframe.loc[short_signal & ~short_signal_special, '_sig_dir'] = -1
        # 做空A特殊信号
        dataframe.loc[short_signal_special, '_sig_open'] = dataframe['low']
        dataframe.loc[short_signal_special, '_sig_range'] = sig_wick
        dataframe.loc[short_signal_special, '_sig_dir'] = -1

        # 做多A信号
        dataframe.loc[long_signal & ~long_signal_special, '_sig_open'] = dataframe['open']
        dataframe.loc[long_signal & ~long_signal_special, '_sig_range'] = sig_range
        dataframe.loc[long_signal & ~long_signal_special, '_sig_dir'] = 1
        # 做多A特殊信号
        dataframe.loc[long_signal_special, '_sig_open'] = dataframe['high']
        dataframe.loc[long_signal_special, '_sig_range'] = sig_wick
        dataframe.loc[long_signal_special, '_sig_dir'] = 1

        # 向前填充信号，最多120根K线
        dataframe['_sig_open'] = dataframe['_sig_open'].ffill(limit=120)
        dataframe['_sig_range'] = dataframe['_sig_range'].ffill(limit=120)
        dataframe['_sig_dir'] = dataframe['_sig_dir'].ffill(limit=120)

        signal_start = short_signal | long_signal
        session = signal_start.cumsum()
        session = session.where(dataframe['_sig_dir'].notna(), 0)

        # 触价检测
        touch_short = (
            (dataframe['_sig_dir'] == -1) & dataframe['_sig_open'].notna() &
            (dataframe['close'] <= dataframe['_sig_open']) & ~signal_start
        )
        touch_long = (
            (dataframe['_sig_dir'] == 1) & dataframe['_sig_open'].notna() &
            (dataframe['close'] >= dataframe['_sig_open']) & ~signal_start
        )

        first_touch_short = pd.Series(False, index=dataframe.index)
        first_touch_long = pd.Series(False, index=dataframe.index)
        if session.max() > 0:
            cum_touch_short = touch_short.groupby(session).cumsum()
            first_touch_short = (cum_touch_short == 1) & touch_short & (session != 0)
            cum_touch_long = touch_long.groupby(session).cumsum()
            first_touch_long = (cum_touch_long == 1) & touch_long & (session != 0)

        if first_touch_short.any():
            idx_s = dataframe.loc[first_touch_short].index
            dataframe.loc[idx_s, 'enter_short'] = 1
            dataframe.loc[idx_s, 'enter_tag'] = (
                'vps_s:' + dataframe.loc[idx_s, '_sig_open'].astype(str) + ':'
                + dataframe.loc[idx_s, '_sig_range'].astype(str))
            for i in idx_s:
                row = dataframe.loc[i]

        if first_touch_long.any():
            idx_l = dataframe.loc[first_touch_long].index
            dataframe.loc[idx_l, 'enter_long'] = 1
            dataframe.loc[idx_l, 'enter_tag'] = (
                'vps_l:' + dataframe.loc[idx_l, '_sig_open'].astype(str) + ':'
                + dataframe.loc[idx_l, '_sig_range'].astype(str))
            for i in idx_l:
                row = dataframe.loc[i]

        dataframe.drop(columns=['_sig_open', '_sig_range', '_sig_dir', 'session'],
                       inplace=True, errors='ignore')

        # ---- 做空B（放量阴线+反弹开盘价入场）----
        bear_short = is_spike & (dataframe['close'] < dataframe['open']) & can_short & (sig_range >= 250)
        bear_short_special = special_candle & (dataframe['close'] < dataframe['open']) & can_short

        for idx in bear_short[bear_short].index:
            pos = dataframe.index.get_loc(idx)
            sig_open = dataframe.iloc[pos]['open']
            sig_close = dataframe.iloc[pos]['close']
            sig_range_val = abs(dataframe.iloc[pos]['close'] - sig_open)
            signal_time = dataframe.iloc[pos]['date']
            for look_pos in range(pos + 1, min(pos + 121, len(dataframe))):
                if dataframe.iloc[look_pos]['close'] >= sig_open:
                    entry_idx = dataframe.index[look_pos]
                    if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                        dataframe.loc[entry_idx, 'enter_short'] = 1
                        dataframe.loc[entry_idx, 'enter_tag'] = f'vps_sb:{sig_open}:{sig_range_val}'
                        entry_time = dataframe.loc[entry_idx, 'date']
                    break

        for idx in bear_short_special[bear_short_special].index:
            pos = dataframe.index.get_loc(idx)
            sig_open = dataframe.iloc[pos]['high']
            sig_range_val = dataframe.iloc[pos]['high'] - dataframe.iloc[pos]['low']
            signal_time = dataframe.iloc[pos]['date']
            for look_pos in range(pos + 1, min(pos + 121, len(dataframe))):
                if dataframe.iloc[look_pos]['close'] >= sig_open:
                    entry_idx = dataframe.index[look_pos]
                    if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                        dataframe.loc[entry_idx, 'enter_short'] = 1
                        dataframe.loc[entry_idx, 'enter_tag'] = f'vps_sb:{sig_open}:{sig_range_val}'
                        entry_time = dataframe.loc[entry_idx, 'date']
                    break

        # ---- 做多B（放量阳线+日线在MA之上+收盘确认回踩入场）----
        # 自上次MA交叉以来的最低收开盘价与信号开盘价的价差 > 250点
        pullback_ok = (dataframe['open'] - dataframe['low_since_cross']) > 250
        bull_long = is_spike & (dataframe['close'] > dataframe['open']) & not_below_both & (sig_range >= 250) & pullback_ok
        immediate = (dataframe['close'].shift(-1) <= dataframe['open']) & bull_long
        bull_long = bull_long & ~immediate

        for idx in bull_long[bull_long].index:
            pos = dataframe.index.get_loc(idx)
            sig_open_val = dataframe.iloc[pos]['open']
            sig_close_val = dataframe.iloc[pos]['close']
            sig_range_val = abs(sig_close_val - sig_open_val)
            signal_time = dataframe.iloc[pos]['date']
            for look_pos in range(pos + 1, min(pos + 121, len(dataframe))):
                close_val = dataframe.iloc[look_pos]['close']
                if close_val > sig_open_val and close_val <= sig_close_val:
                    entry_idx = dataframe.index[look_pos]
                    if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                        dataframe.loc[entry_idx, 'enter_long'] = 1
                        dataframe.loc[entry_idx, 'enter_tag'] = f'vps_lb:{sig_open_val}:{sig_range_val}'
                        entry_time = dataframe.loc[entry_idx, 'date']
                    break

        bull_long_special = special_candle & (dataframe['close'] > dataframe['open']) & not_below_both & pullback_ok
        immediate_special = (dataframe['close'].shift(-1) <= dataframe['low']) & bull_long_special
        bull_long_special = bull_long_special & ~immediate_special

        for idx in bull_long_special[bull_long_special].index:
            pos = dataframe.index.get_loc(idx)
            sig_open = dataframe.iloc[pos]['low']
            sig_range_val = dataframe.iloc[pos]['high'] - dataframe.iloc[pos]['low']
            signal_time = dataframe.iloc[pos]['date']
            for look_pos in range(pos + 1, min(pos + 121, len(dataframe))):
                if dataframe.iloc[look_pos]['close'] <= sig_open:
                    entry_idx = dataframe.index[look_pos]
                    if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                        dataframe.loc[entry_idx, 'enter_long'] = 1
                        dataframe.loc[entry_idx, 'enter_tag'] = f'vps_lb:{sig_open}:{sig_range_val}'
                        entry_time = dataframe.loc[entry_idx, 'date']
                    break

        # ================================================================
        # 第二部分：SDOnly 分支（成交额 7000万-9000万，立即入场）
        # ================================================================

        vol = dataframe['volume'] * dataframe['close']

        # ---- 做空D：日线不在MA5/MA25之下 + 实阴线跌破EMA5/EMA25 ----
        sd_condition = (
            not_below_both &
            (vol >= 70_000_000) & (vol < 90_000_000) &
            (dataframe['close'] < dataframe['open']) &
            ((dataframe['open'] - dataframe['close']) >= 250) &
            (dataframe['close'] < dataframe['ema5']) &
            (dataframe['close'] < dataframe['ema25'])
        )

        for idx in sd_condition[sd_condition].index:
            pos = dataframe.index.get_loc(idx)
            sig_open_val = max(dataframe.iloc[pos]['ema5'], dataframe.iloc[pos]['ema25'])
            sig_close_val = dataframe.iloc[pos]['close']
            sig_range_val = sig_open_val - sig_close_val
            signal_time = dataframe.iloc[pos]['date']
            if sig_range_val <= 0:
                continue
            if pos + 1 < len(dataframe):
                entry_idx = dataframe.index[pos + 1]
                if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                    dataframe.loc[entry_idx, 'enter_short'] = 1
                    dataframe.loc[entry_idx, 'enter_tag'] = f'vps_sd:{sig_open_val}:{sig_range_val}'
                    entry_time = dataframe.loc[entry_idx, 'date']

        # ---- 做多D：日线不在MA5/MA25之下 + 实阳线突破EMA5/EMA25 ----
        sl_condition = (
            below_both &
            (vol >= 70_000_000) & (vol < 90_000_000) &
            (dataframe['close'] > dataframe['open']) &
            ((dataframe['close'] - dataframe['open']) >= 250) &
            (dataframe['close'] > dataframe['ema5']) &
            (dataframe['close'] > dataframe['ema25'])
        )

        for idx in sl_condition[sl_condition].index:
            pos = dataframe.index.get_loc(idx)
            sig_open_val = min(dataframe.iloc[pos]['ema5'], dataframe.iloc[pos]['ema25'])
            sig_close_val = dataframe.iloc[pos]['close']
            sig_range_val = sig_close_val - sig_open_val
            signal_time = dataframe.iloc[pos]['date']
            if sig_range_val <= 0:
                continue
            if pos + 1 < len(dataframe):
                entry_idx = dataframe.index[pos + 1]
                if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                    dataframe.loc[entry_idx, 'enter_long'] = 1
                    dataframe.loc[entry_idx, 'enter_tag'] = f'vps_sl:{sig_open_val}:{sig_range_val}'
                    entry_time = dataframe.loc[entry_idx, 'date']

        # ================================================================
        # 第三部分：F 分支（弱势突破，立即入场）
        # ================================================================

        # ---- 做空F：日线不在MA之下 + 跌破双均线，high-ema_min>800 ----
        sf_below = (
            (dataframe['close'] < dataframe['ema5']) &
            (dataframe['close'] < dataframe['ema25'])
        )
        sf_range = (dataframe['high_since_cross'] - dataframe['ema_min']) >= 150
        sf_vol = (vol >= 3_000_000) & (vol < 10_000_000)
        sf_weak = sf_vol & ((dataframe['close'] - dataframe['open']).abs() < 100)
        sf_condition = not_below_both & sf_below & sf_range & sf_weak

        for idx in sf_condition[sf_condition].index:
            pos = dataframe.index.get_loc(idx)
            ma_min_val = dataframe.iloc[pos]['ema_min']
            recorded_high = dataframe.iloc[pos]['high_since_cross']
            sig_range_val = recorded_high - ma_min_val
            signal_time = dataframe.iloc[pos]['date']
            if sig_range_val <= 0:
                continue
            if pos + 1 < len(dataframe):
                entry_idx = dataframe.index[pos + 1]
                if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                    dataframe.loc[entry_idx, 'enter_short'] = 1
                    dataframe.loc[entry_idx, 'enter_tag'] = f'vps_sf:{ma_min_val}:{sig_range_val}'
                    entry_time = dataframe.loc[entry_idx, 'date']

        # ---- 做多F：日线不在MA之下 + 升破双均线，ema_max-low>800，原版固定止损 ----
        lf_above = (
            (dataframe['close'] > dataframe['ema5']) &
            (dataframe['close'] > dataframe['ema25'])
        )
        lf_range = (dataframe['ema_max'] - dataframe['low_since_cross']) >= 150
        lf_vol = vol < 6_000_000
        lf_weak = lf_vol & ((dataframe['close'] - dataframe['open']).abs() < 100)
        lf_condition = not_below_both & lf_above & lf_range & lf_weak

        for idx in lf_condition[lf_condition].index:
            pos = dataframe.index.get_loc(idx)
            ma_max_val = dataframe.iloc[pos]['ema_max']
            recorded_low = dataframe.iloc[pos]['low_since_cross']
            sig_range_val = ma_max_val - recorded_low
            signal_time = dataframe.iloc[pos]['date']
            if sig_range_val <= 0:
                continue
            if pos + 1 < len(dataframe):
                entry_idx = dataframe.index[pos + 1]
                if dataframe.loc[entry_idx, 'enter_short'] != 1 and dataframe.loc[entry_idx, 'enter_long'] != 1:
                    dataframe.loc[entry_idx, 'enter_long'] = 1
                    dataframe.loc[entry_idx, 'enter_tag'] = f'vps_lf:{ma_max_val}:{sig_range_val}'
                    entry_time = dataframe.loc[entry_idx, 'date']

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
                tag = trade.enter_tag
                if tag.startswith('vps_sd') or tag.startswith('vps_sl'):
                    # SDOnly：止损在 sig_open（均线突破位）
                    return stoploss_from_absolute(sig_open, current_rate,
                                                  trade.is_short, trade.leverage)
                elif tag.startswith('vps_sf'):
                    # 做空F：止损 sig_open + sig_range
                    stop_rate = sig_open + sig_range
                elif tag.startswith('vps_lf'):
                    # 做多F：止损 sig_open - sig_range
                    stop_rate = sig_open - sig_range
                elif trade.is_short:
                    # VolumeSpike 做空：sig_open + sig_range
                    stop_rate = sig_open + sig_range
                else:
                    # VolumeSpike 做多：sig_open - sig_range
                    stop_rate = sig_open - sig_range
                return stoploss_from_absolute(stop_rate, current_rate,
                                              trade.is_short, trade.leverage)
        return self.stoploss

    # ----- 自定义止盈 ----
    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        if not trade.enter_tag or not trade.enter_tag.startswith('vps_'):
            return None

        parts = trade.enter_tag.split(':')
        if len(parts) < 3:
            return None
        sig_open = float(parts[1])
        sig_range = float(parts[2])
        tag = trade.enter_tag

        if tag.startswith('vps_sd'):
            if current_rate <= sig_open - 2 * sig_range:
                pair_key = pair
                ts = current_time.isoformat()
                is_spike = self._candle_vol_spike.get(pair_key, {}).get(ts, '') == 'bear'
                if is_spike:
                    if current_rate <= sig_open - 3 * sig_range:
                        return 'take_profit'
                    return None
                return 'take_profit'

        elif tag.startswith('vps_sf'):
            # 做空F止盈：原逻辑
            if current_rate <= sig_open - sig_range:
                pair_key = pair
                ts = current_time.isoformat()
                is_spike = self._candle_vol_spike.get(pair_key, {}).get(ts, '') == 'bear'
                if is_spike:
                    if current_rate <= sig_open - 3 * sig_range:
                        return 'take_profit'
                    return None
                return 'take_profit'
        elif tag.startswith('vps_sl'):
            if current_rate >= sig_open + 2 * sig_range:
                pair_key = pair
                ts = current_time.isoformat()
                is_spike = self._candle_vol_spike.get(pair_key, {}).get(ts, '') == 'bull'
                if is_spike:
                    if current_rate >= sig_open + 3 * sig_range:
                        return 'take_profit'
                    return None
                return 'take_profit'

        elif tag.startswith('vps_lb'):
            if current_rate >= sig_open + 2 * sig_range:
                return 'take_profit'

        elif tag.startswith('vps_l'):
            if current_rate >= sig_open + sig_range:
                return 'take_profit'

        elif tag.startswith('vps_lf'):
            if current_rate >= sig_open + sig_range:
                pair_key = pair
                ts = current_time.isoformat()
                is_spike = self._candle_vol_spike.get(pair_key, {}).get(ts, '') == 'bull'
                if is_spike:
                    if current_rate >= sig_open + 3 * sig_range:
                        return 'take_profit'
                    return None
                return 'take_profit'

        elif trade.is_short:
            if current_rate <= sig_open - sig_range:
                pair_key = pair
                ts = current_time.isoformat()
                is_spike = self._candle_vol_spike.get(pair_key, {}).get(ts, '') == 'bear'
                if is_spike:
                    if current_rate <= sig_open - 2 * sig_range:
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
                branch_name = parts[0]
                sig_open = float(parts[1])
                sig_range = float(parts[2])
                dir_cn = "做空" if trade.is_short else "做多"
                open_bt = trade.open_date + timedelta(hours=8)
                close_bt = current_time + timedelta(hours=8)
                profit_pct = ((rate / trade.open_rate) - 1) * (-1 if trade.is_short else 1) * trade.leverage * 100
                print(f"  [成交] {dir_cn} {branch_name} 开仓={open_bt.strftime('%m-%d %H:%M')} "
                      f"平仓={close_bt.strftime('%m-%d %H:%M')} "
                      f"入场={trade.open_rate:.1f} 出场={rate:.1f} "
                      f"盈亏={profit_pct:.2f}% 原因={exit_reason}")
        return True

    # ----- 自定义入场价：maker挂单（免taker手续费）-----
    def custom_entry_price(self, pair: str, trade: Optional["Trade"],
                           current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        if side == "short":
            return proposed_rate + 10
        return proposed_rate - 10

    # ----- 自定义仓位：每单0.01 BTC -----
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float],
                            max_stake: float, leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        target_margin = 0.05 * current_rate / leverage
        if min_stake is not None:
            target_margin = max(target_margin, min_stake)
        target_margin = min(target_margin, max_stake)
        return target_margin
