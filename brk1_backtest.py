"""
brk1 突破交易策略回测脚本
白银期货 AG88 15分钟K线
"""
import pandas as pd, numpy as np
from datetime import datetime

# ====== 数据加载 ======
df = pd.read_feather('/root/lianghua/user_data/data/silver_15m.feather')
df['date'] = pd.to_datetime(df['date']).round('1min')
df = df.sort_values('date').reset_index(drop=True)

def tg_original(bjt):
    if bjt.hour>=21: return bjt.date()
    elif bjt.hour<3: return (bjt-pd.Timedelta(days=1)).date()
    elif bjt.hour>=9 and bjt.hour<15: return (bjt-pd.Timedelta(days=1)).date()
    else: return None

df['tg0'] = df['date'].apply(tg_original)
df = df[df['tg0'].notna()].copy()
sorted_tgs = sorted(df['tg0'].unique())
tg_map = {}
prev_tg = None; prev_tg0 = None
for tg in sorted_tgs:
    if prev_tg is None:
        tg_map[tg] = tg; prev_tg = tg; prev_tg0 = tg
    else:
        gap = (tg - prev_tg0).days
        if gap == 2:
            tg_map[tg] = prev_tg; prev_tg0 = tg
        else:
            tg_map[tg] = tg; prev_tg = tg; prev_tg0 = tg
df['tg'] = df['tg0'].map(tg_map)

start='2025-05-29'; end='2026-05-29'
mask = (df['date']>=start)&(df['date']<end)
data = df[mask].copy().reset_index(drop=True)
data['oc_max'] = data[['open','close']].max(axis=1)
data['oc_min'] = data[['open','close']].min(axis=1)
data['body'] = data['close'] - data['open']
data['mid'] = (data['high'] + data['low']) / 2
data['is_bull'] = data['close'] > data['open']
data['is_bear'] = data['close'] < data['open']

daily_max = data.groupby('tg')['oc_max'].max()
daily_min = data.groupby('tg')['oc_min'].min()
data['tg_ocmax_sofar'] = data.groupby('tg')['oc_max'].cummax()
data['tg_ocmin_sofar'] = data.groupby('tg')['oc_min'].cummin()
data['first_of_tg'] = data.groupby('tg').cumcount() == 0
data['prev1_high'] = data['tg'].map(daily_max.shift(1).to_dict())
data['prev1_low'] = data['tg'].map(daily_min.shift(1).to_dict())
data['prev1_ocmin'] = data['tg'].map(daily_min.shift(1).to_dict())
data['prev1_ocmax'] = data['tg'].map(daily_max.shift(1).to_dict())

# ---- 日线MA5/MA25过滤 ----
daily_close = data.groupby('tg')['close'].last()
daily_ma5 = daily_close.rolling(5, min_periods=1).mean()
daily_ma25 = daily_close.rolling(25, min_periods=1).mean()
prev_ma5 = data['tg'].map(daily_ma5.shift(1).to_dict())
prev_ma25 = data['tg'].map(daily_ma25.shift(1).to_dict())
data['prev_higher_ma'] = data[['prev1_ocmax']].max(axis=1)
# 实际上higher_ma = max(ma5, ma25)，但ocmax已经是最高了，我们需要用MA值
data['prev_day_ma5'] = prev_ma5
data['prev_day_ma25'] = prev_ma25
data['prev_higher_ma'] = data[['prev_day_ma5', 'prev_day_ma25']].max(axis=1)
data['prev_lower_ma'] = data[['prev_day_ma5', 'prev_day_ma25']].min(axis=1)


vol_th = 25000
ex230 = ~((data['date'].dt.hour==2)&(data['date'].dt.minute==30)).values

def calc_range(row, is_long, rc, ro):
    rv = row[rc]
    if is_long:
        if row['first_of_tg']: return rv - row[ro]
        d = rv - row['tg_ocmin_sofar']; return d if d>0 else rv - row[ro]
    else:
        if row['first_of_tg']: return row[ro] - rv
        d = row['tg_ocmax_sofar'] - rv; return d if d>0 else row[ro] - rv

def calc_sl(row, is_long, rc, ro):
    if is_long: return row[ro] if row['oc_min']>row[rc] else row['oc_min']
    return row[ro] if row['oc_max']<row[rc] else row['oc_max']

def run_backtest(data, short_gap_wait=True):
    """short_gap_wait: 做空时如果信号K线超过突破点100点，等待下个周期确认"""
    all_trades = []
    
    # ====== 做多 ======
    for sname, il, rc, ro, wait_flag in [('brk1_long', True, 'prev1_high', 'prev1_ocmin', True)]:
        sr = (data['close']>data[rc]) & (data['volume']>vol_th) & data[rc].notna() & ex230 & data['is_bull'] & (data['close']>=data['prev_lower_ma'])
        sig_pool = {}
        for idx in data.index[sr]:
            r = data.loc[idx]; rv = r[rc]
            ng = calc_range(r, il, rc, ro)
            if ng > 0 and ng <= 1000:
                slv = calc_sl(r, il, rc, ro)
                tp = rv + ng
                sig_pool[idx] = {'ref':rv, 'rng':ng, 'sl':slv, 'tp':tp,
                                'sig_dt':r['date'], 'tg':r['tg'],
                                'vol':r['volume'], 'body':r['body'],
                                'o':r['open'], 'c':r['close'],
                                'hi':r['high'], 'lo':r['low'],
                                'mid':r['mid']}
        en = pd.Series(False, index=data.index)
        en_price = pd.Series(0.0, index=data.index)
        sig_idx_map = {}
        used_tg = set()
        for idx in sorted(sig_pool.keys()):
            p = idx; si = sig_pool[idx]
            tg_key = (si['tg'], sname)
            if tg_key in used_tg: continue
            if p+1 >= len(data): continue
            wk = data.iloc[p+1]
            if wk['close'] <= wk['open']: continue
            gap = si['c'] - si['ref']
            if abs(gap) > 100: continue
            if p+1 >= len(data): continue
            ei = p+1
            if en.iloc[ei]: continue
            en.iloc[ei] = True
            en_price.iloc[ei] = data.iloc[ei]['open']
            sig_idx_map[ei] = idx
            used_tg.add(tg_key)
        
        # 持仓跟踪
        trades = []; open_pos = []
        for i in range(len(data)):
            r = data.iloc[i]; bj = r['date']
            # 02:30盈利取利
            if bj.hour == 2 and bj.minute == 30:
                for t in open_pos:
                    pnl = r['close'] - t['entry']
                    if pnl > 0:
                        t['exit']=r['close']; t['pts']=pnl; t['r']='230'; t['xt']=r['date']
                        trades.append(t)
                open_pos = [t for t in open_pos if 'r' not in t]
                continue
            # 14:45强制平仓
            if bj.hour == 14 and bj.minute >= 45:
                for t in open_pos:
                    t['exit']=r['close']; t['pts']=r['close']-t['entry']; t['r']='eod'; t['xt']=r['date']
                    trades.append(t)
                open_pos=[]; continue
            keep=[]
            for t in open_pos:
                if r['low']<=t['sl']:
                    t['exit']=r['close']; t['pts']=r['close']-t['entry']; t['r']='sl'; t['xt']=r['date']
                    trades.append(t); continue
                if r['high']>=t['tp']:
                    pnl=r['close']-t['entry']
                    if r['close'] <= r['mid']:
                        t['r']='tp_fail'
                    else:
                        t['r']='tp' if pnl>0 else 'tp_fail'
                    t['exit']=r['close']; t['pts']=pnl; t['xt']=r['date']
                    trades.append(t); continue
                keep.append(t)
            open_pos=keep
            if len(open_pos)>=2: continue
            if en.iloc[i]:
                si_=sig_pool.get(sig_idx_map.get(i,0),{})
                open_pos.append({'d':'long','entry':en_price.iloc[i],'et':r['date'],
                                 'sl':si_.get('sl',0),'tp':si_.get('tp',0),
                                 'sn':sname,'sig':si_})
        for t in trades: all_trades.append(t)
    
    # ====== 做空 ======
    for sname, il, rc, ro, wait_flag in [('brk1_short', False, 'prev1_low', 'prev1_ocmax', False)]:
        sr = (data['close']<data[rc]) & (data['volume']>vol_th) & data[rc].notna() & ex230 & data['is_bear'] & (data['close']<=data['prev_higher_ma']) & (data['body'].abs()>100)
        sig_pool = {}
        for idx in data.index[sr]:
            r = data.loc[idx]; rv = r[rc]
            ng = calc_range(r, il, rc, ro)
            if ng > 0 and ng <= 1000:
                slv = calc_sl(r, il, rc, ro)
                tp = rv - ng
                sig_pool[idx] = {'ref':rv, 'rng':ng, 'sl':slv, 'tp':tp,
                                'sig_dt':r['date'], 'tg':r['tg'],
                                'vol':r['volume'], 'body':r['body'],
                                'o':r['open'], 'c':r['close'],
                                'hi':r['high'], 'lo':r['low'],
                                'mid':r['mid']}
        en = pd.Series(False, index=data.index)
        en_price = pd.Series(0.0, index=data.index)
        sig_idx_map = {}
        used_tg = set()
        for idx in sorted(sig_pool.keys()):
            p = idx; si = sig_pool[idx]
            tg_key = (si['tg'], sname)
            if tg_key in used_tg: continue
            if p+1 >= len(data): continue
            
            # 新规则: gap>100点需要等待确认
            gap = abs(si['c'] - si['ref'])
            if gap > 100:
                if p+1 >= len(data): continue
                nxt = data.iloc[p+1]
                if nxt['low'] < si['lo'] and nxt['close'] < nxt['open']:
                    if p+2 >= len(data): continue
                    ei = p+2
                else:
                    continue
            else:
                ei = p+1
            
            if en.iloc[ei]: continue
            en.iloc[ei] = True
            en_price.iloc[ei] = data.iloc[ei]['open']
            sig_idx_map[ei] = idx
            used_tg.add(tg_key)
        
        trades = []; open_pos = []
        for i in range(len(data)):
            r = data.iloc[i]; bj = r['date']
            if bj.hour == 2 and bj.minute == 30:
                for t in open_pos:
                    pnl = t['entry'] - r['close']
                    if pnl > 0:
                        t['exit']=r['close']; t['pts']=pnl; t['r']='230'; t['xt']=r['date']
                        trades.append(t)
                open_pos = [t for t in open_pos if 'r' not in t]
                continue
            if bj.hour == 14 and bj.minute >= 45:
                for t in open_pos:
                    t['exit']=r['close']; t['pts']=t['entry']-r['close']; t['r']='eod'; t['xt']=r['date']
                    trades.append(t)
                open_pos=[]; continue
            keep=[]
            for t in open_pos:
                if r['high']>=t['sl']:
                    t['exit']=r['close']; t['pts']=t['entry']-r['close']; t['r']='sl'; t['xt']=r['date']
                    trades.append(t); continue
                if r['low']<=t['tp']:
                    pnl=t['entry']-r['close']
                    if r['close'] >= r['mid']:
                        t['r']='tp_fail'
                    else:
                        t['r']='tp' if pnl>0 else 'tp_fail'
                    t['exit']=r['close']; t['pts']=pnl; t['xt']=r['date']
                    trades.append(t); continue
                keep.append(t)
            open_pos=keep
            if len(open_pos)>=2: continue
            if en.iloc[i]:
                si_=sig_pool.get(sig_idx_map.get(i,0),{})
                open_pos.append({'d':'short','entry':en_price.iloc[i],'et':r['date'],
                                 'sl':si_.get('sl',0),'tp':si_.get('tp',0),
                                 'sn':sname,'sig':si_})
        for t in trades: all_trades.append(t)
    
    return all_trades

# 回测
print("运行回测...")
trades = run_backtest(data)

# 统计
ts_l = [t for t in trades if t['sn']=='brk1_long']
ts_s = [t for t in trades if t['sn']=='brk1_short']
tot_w = sum(1 for t in trades if t['pts']>0)
tot_l = sum(1 for t in trades if t['pts']<0)
tot_p = sum(t['pts'] for t in trades)

print(f"\n{'='*60}")
print(f"brk1 突破策略 回测结果")
print(f"{'='*60}")
print(f"回测期: {start} ~ {end}")
print(f"品种: AG88 15分钟K线")
print(f"\n【做多 brk1_long】")
print(f"  交易: {len(ts_l)}笔")
if ts_l:
    long_w = sum(1 for t in ts_l if t['pts']>0)
    print(f"  胜率: {long_w/len(ts_l)*100:.1f}%")
    print(f"  总盈亏: {sum(t['pts'] for t in ts_l):+}点")
    print(f"  最大亏损: {min(t['pts'] for t in ts_l):+}点")
    print(f"  最大盈利: {max(t['pts'] for t in ts_l):+}点")

print(f"\n【做空 brk1_short】")
print(f"  交易: {len(ts_s)}笔")
if ts_s:
    short_w = sum(1 for t in ts_s if t['pts']>0)
    print(f"  胜率: {short_w/len(ts_s)*100:.1f}%")
    print(f"  总盈亏: {sum(t['pts'] for t in ts_s):+}点")
    print(f"  最大亏损: {min(t['pts'] for t in ts_s):+}点")
    print(f"  最大盈利: {max(t['pts'] for t in ts_s):+}点")

print(f"\n【合计】")
print(f"  交易: {len(trades)}笔")
print(f"  胜率: {tot_w/len(trades)*100:.1f}%")
print(f"  总盈亏: {tot_p:+}点")

# 出场方式分布
print(f"\n{'='*60}")
print(f"出场方式分布")
print(f"{'='*60}")
for sname, ts, label in [('brk1_long', ts_l, '做多'), ('brk1_short', ts_s, '做空')]:
    byr = {}
    for t in ts:
        r=t['r']; byr.setdefault(r,{'n':0,'p':0})
        byr[r]['n']+=1; byr[r]['p']+=t['pts']
    print(f"\n{label} {sname}:")
    rn_map = {'sl':'止损','eod':'收盘','tp':'止盈','tp_fail':'过冲','230':'2:30取利'}
    for r in ['tp','sl','eod','tp_fail','230']:
        if r in byr:
            print(f"  {rn_map[r]}: {byr[r]['n']}笔 {byr[r]['p']:+}点")

# 打印亏损明细
print(f"\n{'='*60}")
print(f"亏损交易明细")
print(f"{'='*60}")

for sname, ts, label in [('brk1_long', ts_l, '做多'), ('brk1_short', ts_s, '做空')]:
    losses = sorted([t for t in ts if t['pts']<0], key=lambda x: x['pts'])
    if not losses: continue
    print(f"\n{label} 亏损{len(losses)}笔 {sum(t['pts'] for t in losses):+}点")
    print(f"{'信号时间':<12} {'信号价':>6} {'入场':<12} {'入场价':>6} {'出场':<12} {'出场价':>6} {'盈亏':>6} {'原因':<6} {'range':>5}")
    print('-'*75)
    for t in losses:
        si=t['sig']
        rt={'sl':'止损','eod':'收盘','tp_fail':'过冲','230':'2:30取利'}.get(t['r'],t['r'])
        print(f"{si['sig_dt']:%m/%d %H:%M} {si['c']:6.0f} {t['et']:%m/%d %H:%M} {t['entry']:6.0f} {t['xt']:%m/%d %H:%M} {t['exit']:6.0f} {t['pts']:+6.0f} {rt:<6} {si['rng']:4.0f}")
