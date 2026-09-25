import argparse,sys; from pathlib import Path; import numpy as np, joblib
sys.path.append(str(Path(__file__).resolve().parents[1]/'src'))
from zkmm.data import load_question
from zkmm.train_direct import predict_batch_with_models
p=argparse.ArgumentParser(); p.add_argument('--question',required=True); p.add_argument('--model-dir',required=True); p.add_argument('--question-labels',required=True,help='npy shape (10,60000)'); p.add_argument('--out',default='submit.npy'); a=p.parse_args()
q=load_question(a.question); bundle=joblib.load(Path(a.model_dir)/'model_bundle.joblib')['targets']; qlab=np.load(a.question_labels); submit=np.zeros((3,10,12000),dtype=np.float32)
for b in range(10):
    batch=q[:,b,:].T.astype(np.float32,copy=True); submit[:,b,:]=predict_batch_with_models(batch,bundle,qlab[b].astype(np.int16)); print('[batch]',b,submit[:,b,:].mean(axis=1))
np.save(a.out,submit.astype(np.float32)); print('[OK]',a.out,submit.shape,submit.dtype)
