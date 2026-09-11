"""Train, select on development speakers, calibrate separately, then evaluate once.
All exported numbers are computed here, not copied from a paper.
"""
from features import *
import json,hashlib,time,subprocess,io
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,brier_score_loss,roc_curve
from sklearn.model_selection import GroupShuffleSplit
from config import ROOT,CACHE
def load(r):
    x,sr=sf.read(CACHE/r['file']);x=x.mean(axis=1) if x.ndim==2 else x
    if sr!=16000:x=resample_poly(x,16000,sr)
    return x.astype(np.float32)
def metrics(y,p):
    pred=p>=.5; n=len(y);acc=float(accuracy_score(y,pred)); z=1.96
    center=(acc+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(acc*(1-acc)/n+z*z/(4*n*n))/(1+z*z/n)
    fpr,tpr,th=roc_curve(y,p);eer=float((fpr[np.argmin(np.abs(fpr-(1-tpr)))]+(1-tpr)[np.argmin(np.abs(fpr-(1-tpr)))])/2)
    return dict(n=n,real=int(sum(y==0)),fake=int(sum(y==1)),accuracy=acc,accuracy_ci95=[center-half,center+half],precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),roc_auc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)),eer=eer,confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())
def thresholds(y,p):
    # Choose maximum calibration coverage with >=90% observed precision, >=20 predictions per side.
    real=-1.;fake=2.
    for t in np.linspace(0,.49,100):
        ix=p<=t
        if ix.sum()>=20 and (y[ix]==0).mean()>=.90:real=float(t)
    for t in np.linspace(.51,1,100):
        ix=p>=t
        if ix.sum()>=20 and (y[ix]==1).mean()>=.90:fake=float(t);break
    return dict(real=real,fake=fake,policy='Maximum coverage on calibration speakers with at least 20 predictions and observed class precision >=90%; not a guarantee on new audio. -1/2 disables an unsupported decision.')
def main():
    data=json.loads((ROOT/'research/dataset-manifest.json').read_text());rows=data['records']
    X=[];valid=[];excluded=[]
    for i,r in enumerate(rows):
        x=load(r)
        if len(x)<32000 or len(x)>960000 or np.sqrt(np.mean(x*x))<1e-4:
            excluded.append(dict(id=r['utt_id'],reason='Outside supported duration or silent'));continue
        fp=CACHE/(r['utt_id']+'.npy')
        if fp.exists():f=np.load(fp)
        else:f=features(x);np.save(fp,f)
        X.append(f);valid.append(r)
        if i%300==0:print('Features',i,flush=True)
    X=np.array(X);y=np.array([r['label'] for r in valid]);sp=np.array([r['speaker'] for r in valid]);parts=np.array([r['split'] for r in valid]);train=np.where(parts=='train')[0];dev=np.where(parts=='dev')[0]
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=42).split(X[dev],y[dev],sp[dev]));tune=dev[a];cal=dev[b]
    seen=set(sp[train])|set(sp[dev]);test=np.where((parts=='eval') & ~np.isin(sp,list(seen)))[0]
    assert not (set(sp[train])&set(sp[dev]));assert not(set(sp[tune])&set(sp[cal]));assert not(set(sp[test])&seen)
    groups=[train,tune,cal,test]
    assert all(len(np.unique(y[g]))==2 for g in groups)
    for i,g in enumerate(groups):
        for h in groups[i+1:]:assert not(set(valid[k]['sha256'] for k in g)&set(valid[k]['sha256'] for k in h))
    scaler=StandardScaler().fit(X[train]);Z=scaler.transform(X);candidates=[];best=None
    for c in [1.,10.,100.]:
        for gamma in [.003,.01,.03]:
            model=SVC(C=c,gamma=gamma,class_weight='balanced').fit(Z[train],y[train]);auc=roc_auc_score(y[tune],model.decision_function(Z[tune]));candidates.append(dict(C=c,gamma=gamma,tuning_auc=float(auc)))
            if best is None or auc>best[0]:best=(auc,model)
    model=best[1];platt=LogisticRegression(C=1e3).fit(model.decision_function(Z[cal]).reshape(-1,1),y[cal])
    prob=lambda z:platt.predict_proba(model.decision_function(z).reshape(-1,1))[:,1]
    limits=thresholds(y[cal],prob(Z[cal]));p=prob(Z[test]);clean=metrics(y[test],p)
    ood=float(np.quantile(np.mean(Z[train]**2,axis=1),.995)*3)
    out=dict(version='govvoice-svm-v1',feature_version='mfcc90-v1',sample_rate=16000,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),support=model.support_vectors_.tolist(),dual=model.dual_coef_[0].tolist(),intercept=float(model.intercept_[0]),gamma=float(model._gamma),calibration=dict(slope=float(platt.coef_[0,0]),intercept=float(platt.intercept_[0])),thresholds=limits,ood_limit=ood)
    (ROOT/'dist/model.json').write_text(json.dumps(out,separators=(',',':')))
    print('Selected',model.C,model._gamma,'clean',clean,flush=True)
    rng=np.random.default_rng(42);noisy=[];compressed=[]
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for i,k in enumerate(test):
        x=load(valid[k]);noise=rng.standard_normal(len(x));noise*=np.sqrt(np.mean(x*x)/100)/max(np.sqrt(np.mean(noise*noise)),1e-10)
        noisy.append(features((x+noise).astype(np.float32)))
        args=[ffmpeg,'-hide_banner','-loglevel','error','-f','f32le','-ar','16000','-ac','1','-i','pipe:0','-f','mp3','-b:a','32k','pipe:1']
        encoded=subprocess.run(args,input=x.tobytes(),capture_output=True,check=True,timeout=15).stdout
        decoded=subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-i','pipe:0','-f','f32le','-ar','16000','-ac','1','pipe:1'],input=encoded,capture_output=True,check=True,timeout=15).stdout
        compressed.append(features(np.frombuffer(decoded,dtype=np.float32)))
        if i%100==0:print('Robustness',i,'/',len(test),flush=True)
    # Quality and OOD abstention match browser behavior.
    accepted=(p<=limits['real'])|(p>=limits['fake']);accepted &= np.mean(Z[test]**2,axis=1)<=ood
    for j,k in enumerate(test):
        xx=load(valid[k]);accepted[j] &= len(xx)>=48000 and np.mean(np.abs(xx)>=.999)<=.1
    selective=dict(coverage=float(accepted.mean()),accepted=int(accepted.sum()),uncertain=int((~accepted).sum()),accuracy=float(np.mean((p[accepted]>=.5)==y[test][accepted])) if accepted.any() else None)
    splits={name:dict(n=len(g),real=int(sum(y[g]==0)),fake=int(sum(y[g]==1)),speakers=sorted(set(sp[g])),ids=[valid[k]['utt_id'] for k in g]) for name,g in zip(['train','tune','calibration','test'],groups)}
    evals=dict(model='Standardized MFCC/delta/spectral features + RBF SVM + sigmoid calibration',selection=candidates,selected=dict(C=model.C,gamma=model._gamma,support_vectors=len(model.support_)),splits=splits,conditions=dict(clean=clean,noise_20db=metrics(y[test],prob(scaler.transform(noisy))),mp3_32kbps=metrics(y[test],prob(scaler.transform(compressed)))),selective=selective,thresholds=limits,excluded=excluded,excluded_seen_eval_speakers=int(sum(parts=='eval')-len(test)),dataset=data['repository'],revision=data['revision'],limitations=['Deterministic convenience subset, not the full benchmark; English only.','ASVspoof 2019 synthesis methods are dated. No operational or Arabic validation.','Independent speakers and exact-byte duplicate checks do not establish independence of every script or source utterance.','Segment scores are uncalibrated and localization accuracy has not been established.','Abstention and calibration do not guarantee detection of unfamiliar synthesis.','No pretrained neural comparison was run; compact SVM selected to support local browser inference.'])
    # Blind reel: first four eligible test clips per class by ID; never select by success.
    audio=ROOT/'dist/audio';audio.mkdir(exist_ok=True)
    selected=[]
    for label in [0,1]:selected.append(sorted([k for k in test if y[k]==label],key=lambda k:valid[k]['utt_id'])[:4])
    reel=[]
    for j,k in enumerate([v for pair in zip(*selected) for v in pair]):
        id=f'clip-{j+1:02d}';sf.write(audio/(id+'.wav'),load(valid[k]),16000,subtype='PCM_16')
        reel.append(dict(id=id,file='audio/'+id+'.wav',label='fake' if y[k] else 'real',source_id=valid[k]['utt_id'],source='ASVspoof 2019 LA / ANC-Spoof Original',license='ASVspoof ODC-By; ANC-Spoof packaging CC-BY-NC-4.0',kind='benchmark'))
    team=json.loads((ROOT/'research/generated-speech.json').read_text(encoding='utf-8-sig'));team_results=[]
    for item in team:
        x,sr=sf.read(audio/(item['id']+'.wav'));pp=float(prob(scaler.transform([features(x)]))[0]);team_results.append(dict(id=item['id'],probability=pp,label='fake',engine=item['voice']))
        reel.append(dict(id=item['id'],file='audio/'+item['id']+'.wav',label='fake',source='Generated for this project using Microsoft Edge '+item['voice'],kind='team-generated',transcript=item['text'],note='Stock neural text-to-speech, no voice cloning. Not used for training, tuning, or calibration.'))
    evals['team_generated_unseen_engine']=team_results
    (ROOT/'dist/reel.json').write_text(json.dumps(reel,indent=2));(ROOT/'dist/evaluation.json').write_text(json.dumps(evals,indent=2))
    (ROOT/'research/test-predictions.json').write_text(json.dumps([dict(id=valid[k]['utt_id'],label=int(y[k]),probability=float(p[j]),accepted=bool(accepted[j])) for j,k in enumerate(test)],indent=2))
    # Cross-language DSP parity fixture and actual held-out waveform.
    fixture=load(valid[test[0]])[:48000]
    (ROOT/'tests/parity.json').write_text(json.dumps(dict(wave=fixture.tolist(),features=features(fixture).tolist(),probability=float(prob(scaler.transform([features(fixture)]))[0]))))
    print('DONE',dict(splits={k:v['n'] for k,v in splits.items()},clean=clean,selective=selective,team=team_results),flush=True)
if __name__=='__main__':main()
