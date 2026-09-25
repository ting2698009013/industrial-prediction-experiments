import argparse,sys; from pathlib import Path; import numpy as np, joblib
sys.path.append(str(Path(__file__).resolve().parents[1]/'src'))
from zkmm.data import load_train,save_json
from zkmm.train_direct import train_target_model
p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--labels',required=True); p.add_argument('--model-dir',default='runs/full_mode_specific_model'); p.add_argument('--model',default='lgbm'); p.add_argument('--fast',action='store_true'); p.add_argument('--cut-stride',type=int,default=3000); p.add_argument('--row-stride',type=int,default=5); p.add_argument('--cross',default='core',choices=['none','core','full']); p.add_argument('--f63-state',default='base',choices=['base','strong']); p.add_argument('--blend-global-weight',type=float,default=0.15); a=p.parse_args()
data=load_train(a.data); labels=np.load(a.labels); out=Path(a.model_dir); out.mkdir(parents=True,exist_ok=True); bundle={'meta':{'kind':'direct_block_mode_specific'},'targets':{}}
for ti in range(3):
    gm,gmeta=train_target_model(data,ti,len(data),labels,None,a.model,a.fast,a.cut_stride,a.row_stride,a.cross); tb={'global_model':gm,**gmeta,'blend_global_weight':0.0 if ti==0 else a.blend_global_weight}
    if ti in (1,2):
        tb['mode_models']={}
        for mo in (0,1,2): mm,meta=train_target_model(data,ti,len(data),labels,mo,a.model,a.fast,a.cut_stride,a.row_stride,a.cross); tb['mode_models'][str(mo)]=mm; tb[f'mode_{mo}_meta']=meta
    bundle['targets'][f'target_{ti}']=tb
joblib.dump(bundle,out/'model_bundle.joblib'); save_json({'model_dir':str(out)},out/'summary.json'); print('[OK]',out/'model_bundle.joblib')
