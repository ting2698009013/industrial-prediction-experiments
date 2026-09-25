import numpy as np
from .config import SELECTED_COLS, F63_PAIRS, F64_PAIRS, VISIBLE_LEN, PRED_LEN
from .mode import mode_onehot, time_since_segment_start, is_near_switch

def _safe_div(a,b): return a/(b+1e-8)

def _lag(y,cut,lags=(1,2,3,5,10,20,50,100,500,1000)):
    return [float(y[max(0,cut-l)]) for l in lags]

def _stats(y,cut,windows=(5,10,20,50,100,500,1000)):
    out=[]
    for w in windows:
        s=max(0,cut-w); seg=y[s:cut].astype('float64')
        out += [float(seg.mean()), float(seg.std()), float(seg.min()), float(seg.max())]
    for w in (100,500,1000):
        s=max(0,cut-w); seg=y[s:cut].astype('float64')
        out.append(float((seg[-1]-seg[0])/max(1,len(seg)-1)) if len(seg)>=2 else 0.0)
    return out


def _f63_extra_state(y, cut):
    """f63 专用状态特征。只使用 cut 之前的目标历史，避免泄漏。"""
    out = []
    windows = (100, 300, 500, 1000, 3000, 6000, 10000, 20000)
    means = {}
    for w in windows:
        s = max(0, cut - w)
        seg = y[s:cut].astype("float64")
        if len(seg) == 0:
            m = float(y[max(0, cut - 1)])
            sd = 0.0
        else:
            m = float(seg.mean())
            sd = float(seg.std())
        means[w] = m
        out.extend([m, sd, float(y[cut-1]) - m])
    # 多尺度均值差，用于捕捉慢漂移
    for a, b in ((100, 1000), (500, 3000), (1000, 10000), (3000, 20000)):
        out.append(means[a] - means[b])
    # 斜率
    for w in (300, 1000, 3000, 10000):
        s = max(0, cut - w)
        seg = y[s:cut].astype("float64")
        out.append(float((seg[-1] - seg[0]) / max(1, len(seg)-1)) if len(seg) >= 2 else 0.0)
    return out

def _cross_features(full,t_idx,target_index,mode='core'):
    if mode=='none': return None, []
    if target_index==1:
        pairs=F63_PAIRS
    elif target_index==2:
        pairs=F64_PAIRS if mode=='full' else [(50,51),(50,53),(50,52),(52,53)]
    else:
        return None, []
    parts=[]; names=[]
    for a,b in pairs:
        va=full[t_idx,a].astype(np.float32); vb=full[t_idx,b].astype(np.float32)
        parts += [(va*vb)[:,None], (va-vb)[:,None], _safe_div(va,vb)[:,None]]
        names += [f'f{a}xf{b}',f'f{a}-f{b}',f'f{a}d{b}']
    return np.hstack(parts).astype(np.float32), names

def build_direct_features(full,target_index,cuts,horizons,labels=None,selected_cols=SELECTED_COLS,add_mode=True,add_switch=True,cross='core',f63_state='base'):
    y_col=62+target_index; target=full[:,y_col].astype(np.float32); t_idx=cuts+horizons
    if np.any(t_idx < 0) or np.any(t_idx >= len(full)):
        raise IndexError(f't_idx out of bounds: min={int(t_idx.min())}, max={int(t_idx.max())}, len={len(full)}; horizons should be 0..PRED_LEN-1 for a prediction window')
    Xraw=full[:,selected_cols].astype(np.float32,copy=False); parts=[]; names=[]
    parts.append(Xraw[t_idx]); names += [f'f{c:02d}_t' for c in selected_cols]
    lag=np.maximum(t_idx-1,0); parts.append(Xraw[lag]); names += [f'f{c:02d}_lag1' for c in selected_cols]
    parts.append(Xraw[t_idx]-Xraw[lag]); names += [f'f{c:02d}_diff1' for c in selected_cols]
    h=(horizons.astype(np.float32)+1.0)
    parts.append(np.column_stack([h/PRED_LEN,np.log1p(h)/np.log1p(PRED_LEN),np.sqrt(h)/np.sqrt(PRED_LEN)]).astype(np.float32)); names += ['h_norm','h_log','h_sqrt']
    state_y=(full[:,62].astype(np.float32)-full[:,1].astype(np.float32)) if target_index==0 else target
    rows=[]
    for c in cuts:
        c=int(c); rows.append(_lag(state_y,c)+_stats(state_y,c)+[float(state_y[c-1]),float(np.mean(state_y[max(0,c-3000):c]))])
    parts.append(np.asarray(rows,dtype=np.float32)); names += [f'y_state_{i}' for i in range(len(rows[0]))]
    if target_index == 1 and f63_state in ('strong', 'extra'):
        extra_rows = []
        for c in cuts:
            extra_rows.append(_f63_extra_state(state_y, int(c)))
        parts.append(np.asarray(extra_rows, dtype=np.float32))
        names += [f'f63_extra_state_{i}' for i in range(len(extra_rows[0]))]
    if target_index==0:
        parts.append(full[t_idx,1].astype(np.float32)[:,None]); names.append('f01_future_anchor')
    cx,cxn=_cross_features(full,t_idx,target_index,cross)
    if cx is not None: parts.append(cx); names += cxn
    if labels is not None and add_mode:
        lab=labels[t_idx]; parts.append(mode_onehot(lab)); names += ['mode_0','mode_1','mode_2']
        if add_switch:
            parts.append(np.column_stack([time_since_segment_start(labels)[t_idx]/60000.0,is_near_switch(labels)[t_idx]]).astype(np.float32)); names += ['time_since_mode_start_norm','is_near_switch']
    X=np.hstack(parts).astype(np.float32)
    y=(full[t_idx,62].astype(np.float32)-full[t_idx,1].astype(np.float32)) if target_index==0 else target[t_idx].astype(np.float32)
    return X,y,names

def build_prediction_features_for_batch(batch,target_index,labels_batch=None,selected_cols=SELECTED_COLS,add_mode=True,add_switch=True,cross='core',f63_state='base'):
    cuts=np.full(PRED_LEN,VISIBLE_LEN,dtype=np.int64); horizons=np.arange(0,PRED_LEN,dtype=np.int64)
    X,_,names=build_direct_features(batch,target_index,cuts,horizons,labels=labels_batch,selected_cols=selected_cols,add_mode=add_mode,add_switch=add_switch,cross=cross,f63_state=f63_state)
    return X,names
