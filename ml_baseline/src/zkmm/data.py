from pathlib import Path
import json, numpy as np
from .config import BATCH_LEN,VISIBLE_LEN,N_BATCHES

def load_train(path, mmap=True):
    arr=np.load(path, mmap_mode='r' if mmap else None)
    if arr.ndim!=2 or arr.shape[1]!=65: raise ValueError(f'train data must be (N,65), got {arr.shape}')
    return arr

def load_question(path):
    q=np.load(path)
    if q.shape!=(65,10,60000): raise ValueError(f'question must be (65,10,60000), got {q.shape}')
    return q

def fold_window(fold:int):
    bs=fold*BATCH_LEN; cut=bs+VISIBLE_LEN; return bs,cut,cut,bs+BATCH_LEN

def labels_from_segments(n:int, segments):
    lab=np.full(n,-1,dtype=np.int16)
    for s in segments: lab[int(s['start']):int(s['end'])]=int(s['label'])
    last=0
    for i in range(n):
        if lab[i]<0: lab[i]=last
        else: last=lab[i]
    return lab

def save_json(obj,path):
    def default(o):
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,np.ndarray): return o.tolist()
        return str(o)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=default),encoding='utf-8')

def mse_by_target(pred,true):
    out={}
    for i,n in enumerate(['f62','f63','f64']):
        e=pred[i].astype('float64')-true[i].astype('float64')
        out[f'mse_{n}']=float(np.mean(e*e)); out[f'mae_{n}']=float(np.mean(np.abs(e))); out[f'bias_{n}']=float(np.mean(e))
    out['mse_sum']=out['mse_f62']+out['mse_f63']+out['mse_f64']
    return out
