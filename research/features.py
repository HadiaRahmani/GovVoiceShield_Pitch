"""Versioned 90-dimensional MFCC/delta/spectral front end. Browser twin: dsp.mjs."""
import sys
from pathlib import Path
from config import WORK
sys.path.insert(0,str(WORK/'pydeps'))
import numpy as np
SR=16000; NFFT=512; FRAME=400; HOP=160; MELS=26; CEP=13
def banks():
    hz=lambda m:700*(10**(m/2595)-1)
    mel=lambda f:2595*np.log10(1+f/700)
    points=hz(np.linspace(mel(20),mel(8000),MELS+2));freq=np.arange(257)*SR/NFFT
    b=np.array([np.maximum(0,np.minimum((freq-points[i])/(points[i+1]-points[i]),(points[i+2]-freq)/(points[i+2]-points[i+1]))) for i in range(MELS)])
    d=np.array([[np.cos(np.pi*k*(j+.5)/MELS)*np.sqrt((1 if k==0 else 2)/MELS) for j in range(MELS)] for k in range(CEP)])
    return b,d
B,D=banks()
def delta(x):
    p=np.pad(x,((2,2),(0,0)),mode='edge')
    return (p[3:-1]-p[1:-3]+2*(p[4:]-p[:-4]))/10
def features(wave):
    x=np.asarray(wave,dtype=np.float64)
    if x.ndim!=1 or len(x)<400 or not np.isfinite(x).all(): raise ValueError('Invalid mono waveform')
    x=x-x.mean(); peak=np.max(np.abs(x)); x=x/max(peak,1e-8)
    frames=np.lib.stride_tricks.sliding_window_view(x,FRAME)[::HOP]
    p=np.abs(np.fft.rfft(frames*np.hamming(FRAME),n=NFFT))**2/NFFT
    mfcc=np.log(np.maximum(p@B.T,1e-12))@D.T
    d=delta(mfcc);dd=delta(d)
    freq=np.arange(257)*SR/NFFT; tot=p.sum(axis=1)+1e-12
    cent=(p@freq)/tot
    width=np.sqrt((p*(freq[None,:]-cent[:,None])**2).sum(axis=1)/tot)
    roll=np.argmax(np.cumsum(p,axis=1)>=.85*tot[:,None],axis=1)*SR/NFFT
    flat=np.exp(np.log(p+1e-12).mean(axis=1))/(p.mean(axis=1)+1e-12)
    zcr=np.mean((frames[:,1:]>=0)!=(frames[:,:-1]>=0),axis=1)
    rms=np.sqrt((frames**2).mean(axis=1))
    extra=np.stack([zcr,cent/8000,width/8000,roll/8000,flat,rms],axis=1)
    return np.concatenate([v for a in [mfcc,d,dd,extra] for v in [a.mean(axis=0),a.std(axis=0)]])
