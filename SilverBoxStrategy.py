# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Dict, Optional, Union

from freqtrade.strategy import (
    IStrategy, Trade, Order,
    merge_informative_pair,
    stoploss_from_absolute,
    timeframe_to_minutes,
)
import talib.abstract as ta


class SilverBoxStrategy(IStrategy):
    """
    白银主连箱体策略（最终版 v2.0）

    交易日划分（北京时间）：
      夜盘 21:15 ~ 次日 02:30
      早盘 09:00 ~ 15:00
      完整交易日：21:15 ~ 次日15:00

    中轴 = 每个交易日第一根15分钟K线的开盘价（夜盘21:15）

    做多逻辑：
      1. 日线不在MA5/MA25之下（close >= lower_ma）
      2. 日内先跌超200点
      3. 首次15分钟K线收盘站上中轴 → 入场做多
      4. 止损：中轴 - range（日内最低oc_min）
      5. 止盈：中轴 + range（收盘价触发）
      6. 放量阳线（vol>3万手+close>open）→ 止盈扩大到1.5倍range

    做空逻辑：
      1. 日线不在MA5/MA25之上（close <= higher_ma）
      2. 日内先涨超200点
      3. 首次15分钟K线收盘跌破中轴 → 入场做空
      4. 止损：中轴 + range（日内最高oc_max）
      5. 止盈：中轴 - range（收盘价触发）
      6. 放量阴线（vol>3万手+close<open）→ 止盈扩大到1.5倍range

    过滤条件：
      - 02:30夜盘收盘K线不产生信号
      - 信号K线收盘价已超过1.5倍止盈位的过滤
      - 周末休市的日线MA用前一日ffill填充
      - 入场价：下一根K线开盘价（市价入场）
      - 出场价：触发止盈/止损时市价（K线收盘价成交）
      - 收盘前（14:45 BJT）未平仓的强制平仓
    """

    INTERFACE_VERSION = 3

    timeframe = '15m'
    can_short: bool = True

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
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {
        "entry": "IOC",
        "exit": "IOC",
    }

    def __init__(self, config: dict, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self._candle_vol_spike: Dict[str, Dict[str, str]] = {}

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return 1.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ---- 日线MA5/MA25（重采样15m→1d，ffill填充周末） ----
        daily = dataframe.resample('1D', on='date').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
        }).dropna().reset_index()
        daily['ma5'] = ta.SMA(daily, timeperiod=5)
        daily['ma25'] = ta.SMA(daily, timeperiod=25)
        # shift(1)滞后一天 + ffill填充周末休市空缺
        daily['date'] = daily['date'] + pd.Timedelta(days=1)
        dataframe = merge_informative_pair(dataframe, daily, self.timeframe, '1d', ffill=True)

        # ---- 按交易日分组（21:15~次日15:00 BJT） ----
        if 'datetime' in dataframe.columns:
            bj_dt = pd.to_datetime(dataframe['datetime'])
        else:
            bj_dt = dataframe['date']
        # 00:00~02:59归前一天夜盘，09:00~14:59归前一天早盘，21:00~23:59归当天
        def get_trade_date(dt):
            if dt.hour >= 21:
                return dt.date()
            elif dt.hour < 3:
                return (dt - pd.Timedelta(days=1)).date()
            elif dt.hour >= 9 and dt.hour < 15:
                return (dt - pd.Timedelta(days=1)).date()
            else:
                return None
        dataframe['_trade_date'] = bj_dt.apply(get_trade_date)
        # 去掉非交易时段
        dataframe = dataframe[dataframe['_trade_date'].notna()].copy()

        # ---- 中轴 = 每个交易日第一根K线开盘价（夜盘21:15） ----
        dataframe['center_axis'] = dataframe.groupby('_trade_date')['open'].transform('first')

        # ---- 日内指标 ----
        dataframe['oc_min'] = dataframe[['open', 'close']].min(axis=1)
        dataframe['intraday_lowest'] = dataframe.groupby('_trade_date')['oc_min'].cummin()
        dataframe['oc_max'] = dataframe[['open', 'close']].max(axis=1)
        dataframe['intraday_highest'] = dataframe.groupby('_trade_date')['oc_max'].cummax()
        dataframe['range_long'] = dataframe['center_axis'] - dataframe['intraday_lowest']
        dataframe['range_short'] = dataframe['intraday_highest'] - dataframe['center_axis']

        # ---- 触发条件 ----
        dataframe['drop_200'] = dataframe['intraday_lowest'] <= dataframe['center_axis'] - 150
        dataframe['drop_triggered'] = dataframe.groupby('_trade_date')['drop_200'].cummax()
        dataframe['rise_200'] = dataframe['intraday_highest'] >= dataframe['center_axis'] + 150
        dataframe['rise_triggered'] = dataframe.groupby('_trade_date')['rise_200'].cummax()

        # 清首根K线
        first_of_day = dataframe.groupby('_trade_date').cumcount() == 0
        dataframe.loc[first_of_day, 'drop_triggered'] = False
        dataframe.loc[first_of_day, 'rise_triggered'] = False

        # ---- 信号过滤标记 ----
        # 02:30夜盘收盘不产生信号
        dataframe['_night_close'] = (dataframe['date'].dt.hour == 2) & (dataframe['date'].dt.minute == 30)
        # 信号K线收盘已超1.5倍扩盈位则过滤
        dataframe['_too_late_long'] = dataframe['close'] >= dataframe['center_axis'] + 1.5 * dataframe['range_long']
        dataframe['_too_late_short'] = dataframe['close'] <= dataframe['center_axis'] - 1.5 * dataframe['range_short']

        # ---- 缓存量能特征用于custom_exit扩盈判断 ----
        pair_key = metadata.get('pair', '')
        if pair_key and self._candle_vol_spike.get(pair_key) is None:
            self._candle_vol_spike[pair_key] = {}
        volume_sma20 = dataframe['volume'].rolling(window=20).mean()
        for _, row in dataframe.iterrows():
            ts = row['date'].isoformat()
            is_bearish = row['close'] < row['open']
            is_high_vol = row['volume'] >= 3 * volume_sma20.loc[_]
            candle_type = ''
            if is_high_vol:
                candle_type = 'bear' if is_bearish else 'bull'
            self._candle_vol_spike[pair_key][ts] = candle_type

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0

        first_of_day = dataframe.groupby('_trade_date').cumcount() == 0

        # ---- 做多信号 ----
        above_axis = dataframe['close'] > dataframe['center_axis']
        was_below = dataframe['close'].shift(1) <= dataframe['center_axis'].shift(1)
        had_drop = dataframe['drop_triggered']

        # 做多range>700过滤
        long_signal = (
            above_axis & was_below & had_drop &
            ~first_of_day & ~dataframe['_too_late_long'] & ~dataframe['_night_close'] &
            (dataframe['center_axis'] - dataframe['intraday_lowest'] <= 700)
        )

        for idx in long_signal[long_signal].index:
            pos = dataframe.index.get_loc(idx)
            center = dataframe.iloc[pos]['center_axis']
            low_oc = dataframe.iloc[pos]['intraday_lowest']
            sig_range = center - low_oc
            if sig_range <= 0:
                continue
            if pos + 1 < len(dataframe):
                entry_idx = dataframe.index[pos + 1]
                if dataframe.loc[entry_idx, 'enter_short'] != 1:
                    dataframe.loc[entry_idx, 'enter_long'] = 1
                    dataframe.loc[entry_idx, 'enter_tag'] = f'silver_l:{center}:{sig_range}'

        # ---- 做空信号 ----
        below_axis = dataframe['close'] < dataframe['center_axis']
        was_above = dataframe['close'].shift(1) >= dataframe['center_axis'].shift(1)
        had_rise = dataframe['rise_triggered']

        short_signal = (
            below_axis & was_above & had_rise &
            ~first_of_day & ~dataframe['_too_late_short'] & ~dataframe['_night_close']
        )

        for idx in short_signal[short_signal].index:
            pos = dataframe.index.get_loc(idx)
            center = dataframe.iloc[pos]['center_axis']
            high_oc = dataframe.iloc[pos]['intraday_highest']
            sig_range = high_oc - center
            if sig_range <= 0:
                continue
            if pos + 1 < len(dataframe):
                entry_idx = dataframe.index[pos + 1]
                if dataframe.loc[entry_idx, 'enter_long'] != 1:
                    dataframe.loc[entry_idx, 'enter_short'] = 1
                    dataframe.loc[entry_idx, 'enter_tag'] = f'silver_s:{center}:{sig_range}'

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:
        if trade.enter_tag and (trade.enter_tag.startswith('silver_l') or trade.enter_tag.startswith('silver_s')):
            parts = trade.enter_tag.split(':')
            if len(parts) >= 3:
                center = float(parts[1])
                sig_range = float(parts[2])
                if trade.enter_tag.startswith('silver_l'):
                    stop_rate = center - sig_range
                else:
                    stop_rate = center + sig_range
                return stoploss_from_absolute(stop_rate, current_rate,
                                              trade.is_short, trade.leverage)
        return self.stoploss

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        if not trade.enter_tag or not (trade.enter_tag.startswith('silver_l') or trade.enter_tag.startswith('silver_s')):
            return None

        # 日内交易：收盘前强制平仓（14:45 BJT后全部离场）
        bj_dt = current_time + timedelta(hours=8)
        if bj_dt.hour >= 14 and bj_dt.minute >= 45:
            return 'exit_end_of_day'

        parts = trade.enter_tag.split(':')
        if len(parts) < 3:
            return None
        center = float(parts[1])
        sig_range = float(parts[2])

        if trade.enter_tag.startswith('silver_l'):
            # 做多止盈（range>700过滤）：收盘价触发，放量阳线扩盈
            if current_rate >= center + sig_range:
                pair_key = pair
                ts = current_time.isoformat()
                is_spike = self._candle_vol_spike.get(pair_key, {}).get(ts, '') == 'bull'
                if is_spike:
                    if current_rate >= center + 1.5 * sig_range:
                        return 'take_profit_ext'
                    return None
                return 'take_profit'
        else:
            # 做空止盈：收盘价触发，放量阴线扩盈
            if current_rate <= center - sig_range:
                pair_key = pair
                ts = current_time.isoformat()
                is_spike = self._candle_vol_spike.get(pair_key, {}).get(ts, '') == 'bear'
                if is_spike:
                    if current_rate <= center - 1.5 * sig_range:
                        return 'take_profit_ext'
                    return None
                return 'take_profit'

        return None

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float],
                            max_stake: float, leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        target_margin = 0.01 * current_rate / leverage
        if min_stake is not None:
            target_margin = max(target_margin, min_stake)
        target_margin = min(target_margin, max_stake)
        return target_margin
