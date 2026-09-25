import argparse,sys; from pathlib import Path; import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]/'src'))
from zkmm.data import load_train,save_json
from zkmm.config import BATCH_LEN,VISIBLE_LEN
p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--submit',required=True); p.add_argument('--out',default='runs/offline_score'); a=p.parse_args(); data=load_train(a.data); sub=np.load(a.submit); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); rows=[]
for b in range(10):
    s=b*BATCH_LEN+VISIBLE_LEN; e=(b+1)*BATCH_LEN; true=np.stack([data[s:e,62],data[s:e,63],data[s:e,64]]).astype(np.float32); pred=sub[:,b,:].astype(np.float32)
    for ti,name in enumerate(['f62','f63','f64']):
        err=pred[ti].astype(float)-true[ti].astype(float); rows.append({'batch':b,'target':name,'mse':float(np.mean(err*err)),'mae':float(np.mean(np.abs(err))),'bias':float(np.mean(err)),'pred_mean':float(pred[ti].mean()),'true_mean':float(true[ti].mean()),'pred_std':float(pred[ti].std()),'true_std':float(true[ti].std())})
df=pd.DataFrame(rows); df.to_csv(out/'per_target_batch.csv',index=False); summ={t:float(df[df.target==t].mse.mean()) for t in ['f62','f63','f64']}; summ['mse_sum']=summ['f62']+summ['f63']+summ['f64']; save_json(summ,out/'summary.json'); print(summ)
