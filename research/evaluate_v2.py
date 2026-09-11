"""One locked test evaluation; outputs retain each prediction, including failures."""
from train_v2 import *
from train import metrics
import subprocess,imageio_ffmpeg
def score(X,m):
    z=(np.asarray(X)-m['mean'])/m['scale'];sv=np.asarray(m['support']);raw=np.array([m['intercept']+np.dot(m['dual'],np.exp(-m['gamma']*np.sum((sv-a)**2,axis=1))) for a in z]);p=1/(1+np.exp(-np.clip(m['calibration']['slope']*raw+m['calibration']['intercept'],-50,50)));return p,np.mean(z*z,axis=1)
def compress(x):
    ff=imageio_ffmpeg.get_ffmpeg_exe();b=subprocess.run([ff,'-v','error','-f','f32le','-ar','16000','-ac','1','-i','pipe:0','-f','mp3','-b:a','32k','pipe:1'],input=x.tobytes(),capture_output=True,check=True,timeout=20).stdout
    b=subprocess.run([ff,'-v','error','-i','pipe:0','-f','f32le','-ar','16000','-ac','1','pipe:1'],input=b,capture_output=True,check=True,timeout=20).stdout;return np.frombuffer(b,dtype=np.float32)
def accepted(x,p,d,m):return bool((p<=m['thresholds']['real'] or p>=m['thresholds']['fake']) and d<=m['ood_limit'] and len(x)>=48000 and np.mean(np.abs(x)>=.999)<=.1)
def main():
    if (DEST/'final-predictions.json').exists():raise RuntimeError('Final evaluation exists; do not rerun for tuning.')
    lock=json.loads((DEST/'model-lock.json').read_text());assert sha(ROOT/'dist/model.json')==lock['model_sha256'];assert sha(DEST/'completed-manifest.json')==lock['manifest_sha256']
    m=json.loads((ROOT/'dist/model.json').read_text());rows=json.loads((DEST/'completed-manifest.json').read_text());test=[r for r in rows if r['split']=='test'];primary=[r for r in test if domain(r)=='paired'];records=[]
    for i,r in enumerate(test):
        x=load(r)
        for condition,xx in [('clean',x),('noise_20db',noise(x,r['id'])),('mp3_32kbps',compress(x))]:
            p,d=score([features(xx)],m);records.append(dict(id=r['id'],label=r['label'],domain=domain(r),platform=r['platform'],condition=condition,probability=float(p[0]),accepted=accepted(xx,p[0],d[0],m)))
        if i%75==0:print('Final evaluation',i,'/',len(test),flush=True)
    (DEST/'final-predictions.json').write_text(json.dumps(records,indent=2))
    def measure(rr):return metrics(np.array([r['label'] for r in rr]),np.array([r['probability'] for r in rr]))
    conditions={c:measure([r for r in records if r['domain']=='paired' and r['condition']==c]) for c in ['clean','noise_20db','mp3_32kbps']};public={c:measure([r for r in records if r['domain']=='public' and r['condition']==c]) for c in conditions};clean=[r for r in records if r['domain']=='paired' and r['condition']=='clean'];ac=[r for r in clean if r['accepted']];selective=dict(coverage=len(ac)/len(clean),accepted=len(ac),uncertain=len(clean)-len(ac),accuracy=sum((r['probability']>=.5)==r['label'] for r in ac)/len(ac) if ac else None)
    platforms={}
    for p in sorted(set(r['platform'] for r in records if r['domain']=='public')):
        rr=[r for r in records if r['domain']=='public' and r['condition']=='clean' and r['platform']==p];platforms[p]=dict(n=len(rr),binary_accuracy=sum((r['probability']>=.5)==r['label'] for r in rr)/len(rr),decision_coverage=sum(r['accepted'] for r in rr)/len(rr))
    old=json.loads((ROOT/'model-baseline.json').read_text());X=[features(load(r)) for r in primary];p,_=score(X,old);baseline=metrics(np.array([r['label'] for r in primary]),p)
    splits={s:dict(n=sum(r['split']==s for r in rows),speakers=sorted(set(r['source_group'] for r in rows if r['split']==s))) for s in ['train','tune','calibration','test']}
    reel=[];a=sorted([r for r in primary if r['label']==0],key=lambda r:r['id'])[:4];b=sorted([r for r in primary if r['label']==1],key=lambda r:r['id'])[:4]
    for j,r in enumerate([v for pair in zip(a,b) for v in pair]):
        id=f'clip-{j+1:02d}';sf.write(ROOT/'dist/audio'/f'{id}.wav',load(r),16000,subtype='PCM_16');reel.append(dict(id=id,file=f'audio/{id}.wav',label='fake' if r['label'] else 'real',source_id=r['id'],source=r['dataset'],kind='locked-test',note='Selected by ID, not model success. Read speech; not an impersonation of an official.'))
    probes=[]
    for r in json.loads((ROOT/'research/generated-speech.json').read_text(encoding='utf-8-sig')):
        x,sr=sf.read(ROOT/'dist/audio'/f"{r['id']}.wav");p,d=score([features(x)],m);accept=accepted(x,p[0],d[0],m);probes.append(dict(id=r['id'],probability=float(p[0]),accepted=accept,label='fake'));reel.append(dict(id=r['id'],file=f"audio/{r['id']}.wav",label='fake',source='Project-generated Microsoft Edge '+r['voice'],kind='team-generated',transcript=r['text'],note='Fictional announcement. Previously inspected regression probe, excluded from final-test metrics.'))
    ev=dict(model='90 acoustic features, RBF SVM trained on contemporary speech, separately sigmoid-calibrated',protocol='Paired English read speech; held-out human speakers, text IDs and Edge stock voices. Edge engine represented in training. Public collection is a separate diagnostic.',conditions=conditions,selective=selective,splits=splits,thresholds=m['thresholds'],public_diagnostic=public,public_platforms=platforms,prior_svm_on_fresh_primary=baseline,regression_probes=probes,limitations=['Small English read-speech test; not Arabic, telephone, government-domain, video or identity verification.','Clip-level confidence interval is descriptive; paired scripts and shared synthesis voices are dependent.','Balanced-study calibrated estimates are not operational probabilities.','Public diagnostic has only a few source groups, unverified speaker identities and potentially shared synthetic scripts.','Window scores are uncalibrated review hints, not validated splice localization.','AASIST v1 and the two original announcements were inspected previously; they are not fresh validation.'])
    (ROOT/'dist/evaluation.json').write_text(json.dumps(ev,indent=2));(ROOT/'dist/reel.json').write_text(json.dumps(reel,indent=2));fixture=load(primary[0])[:48000];p,_=score([features(fixture)],m);(ROOT/'tests/v2-parity.json').write_text(json.dumps(dict(wave=fixture.tolist(),features=features(fixture).tolist(),probability=float(p[0]))));print(json.dumps(dict(primary=conditions,selective=selective,public=public,platforms=platforms,probes=probes),indent=2),flush=True)
if __name__=='__main__':main()
