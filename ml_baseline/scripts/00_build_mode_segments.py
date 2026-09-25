import argparse,sys,json; from pathlib import Path; import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]/'src'))
from zkmm.data import load_train,save_json
from zkmm.mode import labels_from_known_segments,labels_to_segments
p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--out',default='runs/modes_known'); a=p.parse_args()
data=load_train(a.data); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); labels=labels_from_known_segments(len(data)); np.save(out/'mode_labels.npy',labels); save_json({'segments':labels_to_segments(labels)},out/'mode_segments.json'); print('[OK]',out/'mode_labels.npy'); print(json.dumps(labels_to_segments(labels),ensure_ascii=False,indent=2))
