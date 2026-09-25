import numpy as np
from .config import KNOWN_SEGMENTS
from .data import labels_from_segments

def labels_from_known_segments(n=600000): return labels_from_segments(n, KNOWN_SEGMENTS)

def labels_to_segments(labels):
    labels=np.asarray(labels).astype(int); seg=[]; s=0; cur=int(labels[0])
    for i in range(1,len(labels)):
        if int(labels[i])!=cur:
            seg.append({'start':s,'end':i,'label':cur,'length':i-s}); s=i; cur=int(labels[i])
    seg.append({'start':s,'end':len(labels),'label':cur,'length':len(labels)-s}); return seg

def mode_onehot(labels):
    labels=np.asarray(labels).astype(int); out=np.zeros((len(labels),3),dtype=np.float32)
    for m in (0,1,2): out[:,m]=(labels==m).astype(np.float32)
    return out

def time_since_segment_start(labels):
    labels=np.asarray(labels).astype(int); out=np.zeros(len(labels),dtype=np.float32); s=0
    for i in range(1,len(labels)):
        if labels[i]!=labels[i-1]: out[s:i]=np.arange(i-s,dtype=np.float32); s=i
    out[s:]=np.arange(len(labels)-s,dtype=np.float32); return out

def is_near_switch(labels,radius=3000):
    labels=np.asarray(labels).astype(int); out=np.zeros(len(labels),dtype=np.float32)
    for c in np.where(labels[1:]!=labels[:-1])[0]+1: out[max(0,c-radius):min(len(labels),c+radius)]=1.0
    return out
