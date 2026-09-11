"""Development-only fitting. Run evaluate_v2.py only after this writes model-lock.json."""
from features import *
from config import ROOT,WORK
import json,hashlib,time
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DEST=ROOT/'research/v2';DATA=WORK/'v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(r):
    x,sr=sf.read(DATA/r['file']);x=x.mean(axis=1) if x.ndim==2 else x
    if sr!=16000:x=resample_poly(x,16000,sr)
    return x.astype(np.float32)
def domain(r):return 'paired' if r['id'].startswith(('libri-','edge-')) else 'public'
def noise(x,id):
    rng=np.random.default_rng(int(hashlib.sha256(id.encode()).hexdigest()[:8],16));n=rng.standard_normal(len(x));return (x+n*np.sqrt(np.mean(x*x)/100)/max(np.sqrt(np.mean(n*n)),1e-10)).astype(np.float32)
def weights(rows):
    cells=[(domain(r),r['label']) for r in rows];return np.array([len(rows)/(4*cells.count(c)) for c in cells])
def featurize(rows,augment=False):
    X=[];rr=[]
    for i,r in enumerate(rows):
        x=load(r)
        for cond in (['clean','noise'] if augment else ['clean']):
            p=DATA/(r['id']+'-'+cond+'.npy')
            if p.exists():f=np.load(p)
            else:f=features(x if cond=='clean' else noise(x,r['id']));np.save(p,f)
            X.append(f);rr.append(r)
        if i%250==0:print('Features',i,'/',len(rows),flush=True)
    return np.array(X),rr
def main():
    if (DEST/'model-lock.json').exists():raise RuntimeError('Frozen model exists; preserve this experiment.')
    lock=json.loads((DEST/'completed-lock.json').read_text());assert sha(DEST/'completed-manifest.json')==lock['completed_manifest_sha256']
    rows=json.loads((DEST/'completed-manifest.json').read_text());parts={s:[r for r in rows if r['split']==s] for s in ['train','tune','calibration']}
    X,tr=featurize(parts['train'],True);T,tu=featurize(parts['tune']);K,ca=featurize(parts['calibration']);y=np.array([r['label'] for r in tr]);yt=np.array([r['label'] for r in tu]);yc=np.array([r['label'] for r in ca])
    scaler=StandardScaler().fit(X);Z=scaler.transform(X);ZT=scaler.transform(T);ZK=scaler.transform(K);candidates=[];best=None
    for c in [1.,10.,100.]:
        for g in [.003,.01,.03]:
            m=SVC(C=c,gamma=g).fit(Z,y,sample_weight=weights(tr));p=m.decision_function(ZT);scores={d:float(roc_auc_score(yt[[domain(r)==d for r in tu]],p[[domain(r)==d for r in tu]])) for d in ['paired','public']};score=float(np.mean(list(scores.values())));candidates.append(dict(C=c,gamma=g,domain_auc=scores,mean_auc=score));print(candidates[-1],flush=True)
            if best is None or score>best[0]:best=(score,m)
    m=best[1];pl=LogisticRegression(C=1000).fit(m.decision_function(ZK).reshape(-1,1),yc,sample_weight=weights(ca));p=pl.predict_proba(m.decision_function(ZK).reshape(-1,1))[:,1];real=-1.;fake=2.
    for t in np.linspace(0,.2,101):
        ix=p<=t
        if ix.sum()>=10 and np.average(yc[ix]==0,weights=weights(ca)[ix])>=.9:real=float(t)
    for t in np.linspace(.8,1,101):
        ix=p>=t
        if ix.sum()>=10 and np.average(yc[ix]==1,weights=weights(ca)[ix])>=.9:fake=float(t);break
    model=dict(version='govvoice-svm-v2',feature_version='mfcc90-v1',sample_rate=16000,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),support=m.support_vectors_.tolist(),dual=m.dual_coef_[0].tolist(),intercept=float(m.intercept_[0]),gamma=float(m._gamma),calibration=dict(slope=float(pl.coef_[0,0]),intercept=float(pl.intercept_[0]),prior='Balanced domain/class study prior'),thresholds=dict(real=real,fake=fake,policy='Separate calibration, weighted precision >=90%, minimum 10 predictions, conservative 0.2/0.8 caps; not an operational guarantee.'),ood_limit=float(np.quantile(np.mean(Z**2,axis=1),.995)*3))
    (ROOT/'dist/model.json').write_text(json.dumps(model,separators=(',',':')));(DEST/'selection.json').write_text(json.dumps(dict(candidates=candidates,selected=dict(C=m.C,gamma=m._gamma),splits={s:len(r) for s,r in parts.items()},training_augmented=len(tr),thresholds=model['thresholds']),indent=2))
    (DEST/'model-lock.json').write_text(json.dumps(dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),model_sha256=sha(ROOT/'dist/model.json'),manifest_sha256=sha(DEST/'completed-manifest.json'),training_code_sha256=sha(Path(__file__)),protocol_sha256=sha(DEST/'PROTOCOL.md'),test_access='Not accessed for features or predictions before this lock'),indent=2));print('MODEL FROZEN',flush=True)
if __name__=='__main__':main()

