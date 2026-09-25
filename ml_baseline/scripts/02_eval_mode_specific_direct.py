import argparse,sys,time; from pathlib import Path; import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]/'src'))
from zkmm.data import load_train,fold_window,mse_by_target,save_json
from zkmm.train_direct import train_target_model,predict_batch_with_models
from zkmm.config import VISIBLE_LEN

def folds(s): return list(range(10)) if s=='all' else [int(x) for x in s.split(',')]
p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--labels',required=True); p.add_argument('--folds',default='9'); p.add_argument('--model',default='lgbm'); p.add_argument('--fast',action='store_true'); p.add_argument('--cut-stride',type=int,default=3000); p.add_argument('--row-stride',type=int,default=10); p.add_argument('--cross',default='core',choices=['none','core','full']); p.add_argument('--f63-state',default='base',choices=['base','strong']); p.add_argument('--blend-global-weight',type=float,default=0.15); p.add_argument('--out',default='runs/mode_specific_eval'); a=p.parse_args()
data=load_train(a.data); labels=np.load(a.labels); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); rows=[]
for f in folds(a.folds):
    t=time.time(); bs,cut,vs,ve=fold_window(f); bundle={}
    for ti in range(3):
        gm,gmeta=train_target_model(data,ti,cut,labels,None,a.model,a.fast,a.cut_stride,a.row_stride,a.cross); tb={'global_model':gm,**gmeta,'blend_global_weight':0.0 if ti==0 else a.blend_global_weight}
        if ti in (1,2):
            tb['mode_models']={}
            for mo in (0,1,2):
                try:
                    mm,meta=train_target_model(data,ti,cut,labels,mo,a.model,a.fast,a.cut_stride,a.row_stride,a.cross); tb['mode_models'][str(mo)]=mm; print(f'  fold={f} target={ti} mode={mo} samples={meta["n_samples"]}')
                except Exception as e: print(f'  [fallback] fold={f} target={ti} mode={mo}: {e}')
        bundle[f'target_{ti}']=tb
    batch=np.array(data[bs:ve],dtype=np.float32,copy=True); batch[VISIBLE_LEN:,[62,63,64]]=0
    pred=predict_batch_with_models(batch,bundle,labels[bs:ve]); true=np.stack([data[vs:ve,62],data[vs:ve,63],data[vs:ve,64]]).astype(np.float32); m=mse_by_target(pred,true); m.update({'fold':f,'seconds':round(time.time()-t,1)}); rows.append(m); print(f"[fold {f}] sum={m['mse_sum']:.6f} f62={m['mse_f62']:.6f} f63={m['mse_f63']:.6f} f64={m['mse_f64']:.6f}")
df=pd.DataFrame(rows); df.to_csv(out/'per_fold.csv',index=False); save_json({'folds':folds(a.folds),'mean_mse_f62':float(df.mse_f62.mean()),'mean_mse_f63':float(df.mse_f63.mean()),'mean_mse_f64':float(df.mse_f64.mean()),'mean_mse_sum':float(df.mse_sum.mean()),'rows':rows},out/'summary.json'); print('[OK]',out/'summary.json')
