import argparse,subprocess,sys; from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--labels',required=True); p.add_argument('--folds',default='9'); p.add_argument('--out',default='runs/f64_cross_ablation'); p.add_argument('--fast',action='store_true'); a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
for cross in ['none','core','full']:
    cmd=[sys.executable,'scripts/02_eval_mode_specific_direct.py','--data',a.data,'--labels',a.labels,'--folds',a.folds,'--cross',cross,'--out',str(out/f'cross_{cross}'),'--row-stride','10','--cut-stride','3000']
    if a.fast: cmd.append('--fast')
    print('[RUN]',' '.join(cmd)); subprocess.run(cmd,check=True)
