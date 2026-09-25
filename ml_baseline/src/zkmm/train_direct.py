import time, numpy as np
from .config import BATCH_LEN,VISIBLE_LEN,PRED_LEN,N_BATCHES
from .features_direct import build_direct_features, build_prediction_features_for_batch
from .models import make_regressor

def make_training_cuts(train_end,min_cut=2000,cut_stride=3000,horizon_max=PRED_LEN):
    last=train_end-horizon_max
    if last<=min_cut: return np.array([],dtype=np.int64)
    cuts=np.arange(min_cut,last+1,cut_stride,dtype=np.int64)
    official=[]
    for b in range(N_BATCHES):
        c=b*BATCH_LEN+VISIBLE_LEN
        if c+horizon_max<=train_end: official.append(c)
    return np.unique(np.concatenate([cuts,np.asarray(official,dtype=np.int64)])) if official else cuts

def expand_cuts_horizons(cuts,row_stride=10,horizon_max=PRED_LEN):
    hs=np.arange(0,horizon_max,row_stride,dtype=np.int64)
    return np.repeat(cuts,len(hs)), np.tile(hs,len(cuts))

def train_target_model(data,target_index,train_end,labels,mode_filter=None,model_name='lgbm',fast=False,cut_stride=3000,row_stride=10,cross='core',add_switch=True,f63_state='base'):
    cuts=make_training_cuts(train_end,cut_stride=cut_stride); cc,hh=expand_cuts_horizons(cuts,row_stride=row_stride)
    if labels is not None and mode_filter is not None:
        mask=labels[cc+hh]==mode_filter; cc,hh=cc[mask],hh[mask]
    if len(cc)<100: raise RuntimeError(f'too few samples target={target_index} mode={mode_filter}: {len(cc)}')
    X,y,names=build_direct_features(data,target_index,cc,hh,labels=labels,cross=cross,add_switch=add_switch,f63_state=f63_state)
    model=make_regressor(model_name,fast=fast,random_state=42+target_index+(0 if mode_filter is None else 10*mode_filter)); t0=time.time(); model.fit(X,y)
    meta={'target_index':target_index,'mode_filter':mode_filter,'n_samples':int(len(y)),'n_features':int(X.shape[1]),'fit_seconds':round(time.time()-t0,2),'feature_names':names,'cross':cross,'add_switch':add_switch,'f63_state':f63_state,'row_stride':row_stride,'cut_stride':cut_stride}
    if target_index==0:
        res=data[:train_end,62].astype(np.float32)-data[:train_end,1].astype(np.float32)
        meta['f62_res_q001']=float(np.quantile(res,0.001)); meta['f62_res_q999']=float(np.quantile(res,0.999))
    return model,meta

def predict_batch_with_models(batch,bundle,labels_batch=None):
    pred=np.zeros((3,PRED_LEN),dtype=np.float32)
    for ti in range(3):
        tb=bundle[f'target_{ti}']; X,_=build_prediction_features_for_batch(batch,ti,labels_batch=labels_batch,cross=tb.get('cross','core'),add_switch=tb.get('add_switch',True),f63_state=tb.get('f63_state','base'))
        pg=tb['global_model'].predict(X).astype(np.float32) if 'global_model' in tb else None; p=None
        if labels_batch is not None and 'mode_models' in tb:
            p=np.zeros(PRED_LEN,dtype=np.float32); fl=labels_batch[VISIBLE_LEN:VISIBLE_LEN+PRED_LEN].astype(int)
            for m in (0,1,2):
                idx=np.where(fl==m)[0]
                if len(idx)==0: continue
                mm=tb['mode_models'].get(str(m))
                p[idx]=(mm.predict(X[idx]).astype(np.float32) if mm is not None else pg[idx])
            wg=float(tb.get('blend_global_weight',0.0));
            if pg is not None and wg>0: p=(1-wg)*p+wg*pg
        if p is None: p=pg
        if ti==0:
            if 'f62_res_q001' in tb: p=np.clip(p,tb['f62_res_q001'],tb['f62_res_q999'])
            pred[0]=batch[VISIBLE_LEN:VISIBLE_LEN+PRED_LEN,1].astype(np.float32)+p
        else: pred[ti]=p
    return pred
