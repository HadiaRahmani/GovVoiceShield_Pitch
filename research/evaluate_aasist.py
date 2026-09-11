"""Fixed second candidate after baseline failure; no test-driven parameter search.
Frozen public AASIST ONNX. Mean negative bona-fide logit across 64600-sample
non-overlapping blocks, repeat-pad final block. Fit sigmoid on calibration only.
"""
from train import *
import onnxruntime as ort,shutil,urllib.request
ROOT=Path(__file__).resolve().parents[1]
opts=ort.SessionOptions();opts.intra_op_num_threads=2
session=ort.InferenceSession(str(WORK/'aasist.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
def scores(waves):
    chunks=[];owners=[]
    for i,x in enumerate(waves):
        for start in range(0,len(x),64600):
            a=x[start:start+64600];chunks.append(np.resize(a,64600));owners.append(i)
    result=np.zeros(len(waves));counts=np.zeros(len(waves))
    for start in range(0,len(chunks),8):
        logits=session.run(None,{'wav':np.array(chunks[start:start+8],dtype=np.float32)})[0]
        for j,v in enumerate(logits):result[owners[start+j]]-=float(v[1]);counts[owners[start+j]]+=1
    return result/counts
def cached(rows,condition='clean'):
    result=[];rng=np.random.default_rng(42);import imageio_ffmpeg
    for start in range(0,len(rows),12):
        batch=rows[start:start+12];waves=[];missing=[]
        for i,r in enumerate(batch):
            path=CACHE/(r['utt_id']+'-aasist-'+condition+'.json')
            if not path.exists():
                x=load(r)
                if condition=='noise_20db':
                    # ID-derived seed keeps transformation stable regardless of caching/batching.
                    nr=np.random.default_rng(int(hashlib.sha256(r['utt_id'].encode()).hexdigest()[:8],16));noise=nr.standard_normal(len(x));noise*=np.sqrt(np.mean(x*x)/100)/max(np.sqrt(np.mean(noise*noise)),1e-10);x=(x+noise).astype(np.float32)
                if condition=='mp3_32kbps':
                    ff=imageio_ffmpeg.get_ffmpeg_exe();enc=subprocess.run([ff,'-hide_banner','-loglevel','error','-f','f32le','-ar','16000','-ac','1','-i','pipe:0','-f','mp3','-b:a','32k','pipe:1'],input=x.tobytes(),capture_output=True,check=True,timeout=15).stdout
                    dec=subprocess.run([ff,'-hide_banner','-loglevel','error','-i','pipe:0','-f','f32le','-ar','16000','-ac','1','pipe:1'],input=enc,capture_output=True,check=True,timeout=15).stdout;x=np.frombuffer(dec,dtype=np.float32)
                waves.append(x);missing.append(path)
        if waves:
            vals=scores(waves)
            for path,val in zip(missing,vals):path.write_text(json.dumps(float(val)))
        result += [json.loads((CACHE/(r['utt_id']+'-aasist-'+condition+'.json')).read_text()) for r in batch]
        if start%120==0:print(condition,start,'/',len(rows),flush=True)
    return np.array(result)
def main():
    e=json.loads((ROOT/'research/baseline-evaluation.json').read_text());allrows=json.loads((ROOT/'research/dataset-manifest.json').read_text())['records'];byid={r['utt_id']:r for r in allrows}
    groups={name:[byid[id] for id in e['splits'][name]['ids']] for name in ['tune','calibration','test']}
    yt=np.array([r['label'] for r in groups['tune']]);yc=np.array([r['label'] for r in groups['calibration']]);y=np.array([r['label'] for r in groups['test']])
    tuning=cached(groups['tune']);auc=float(roc_auc_score(yt,tuning));print('Tuning AUC',auc,flush=True)
    cal=cached(groups['calibration']);platt=LogisticRegression(C=1e3).fit(cal.reshape(-1,1),yc);prob=lambda x:platt.predict_proba(np.array(x).reshape(-1,1))[:,1];limits=thresholds(yc,prob(cal))
    raw=cached(groups['test']);p=prob(raw);conditions={'clean':metrics(y,p)};print('AASIST clean',conditions['clean'],flush=True)
    for condition in ['noise_20db','mp3_32kbps']:conditions[condition]=metrics(y,prob(cached(groups['test'],condition)))
    baseline=json.loads((ROOT/'model-baseline.json').read_text());oldpred={r['id']:r for r in json.loads((ROOT/'research/test-predictions.json').read_text())};accepted=(p<=limits['real'])|(p>=limits['fake']);disagreements=0
    # Two-model disagreement routes to human review, a predeclared conservative rule.
    for j,r in enumerate(groups['test']):
        x=load(r);f=features(x);z=(f-np.array(baseline['mean']))/np.array(baseline['scale']);disagree=(oldpred[r['utt_id']]['probability']>=.5)!=(p[j]>=.5)
        disagreements+=int(disagree);accepted[j]&=not disagree and len(x)>=48000 and np.mean(np.abs(x)>=.999)<=.1 and np.mean(z*z)<=baseline['ood_limit']
    selective=dict(coverage=float(accepted.mean()),accepted=int(accepted.sum()),uncertain=int((~accepted).sum()),accuracy=float(np.mean((p[accepted]>=.5)==y[accepted])) if accepted.any() else None,model_disagreements=disagreements)
    team=[]
    for item in json.loads((ROOT/'research/generated-speech.json').read_text()):
        x,sr=sf.read(ROOT/'dist/audio'/(item['id']+'.wav'));s=float(scores([x.astype(np.float32)])[0]);team.append(dict(id=item['id'],probability=float(prob([s])[0]),raw_score=s,label='fake',engine=item['voice']))
    baseline['version']='govvoice-aasist-svm-v1';baseline['neural']=dict(file='aasist.onnx',input='wav',output='logits',samples=64600,calibration=dict(slope=float(platt.coef_[0,0]),intercept=float(platt.intercept_[0])),thresholds=limits,aggregation='mean negative bona-fide logit across all consecutive 64600-sample blocks; repeat-pad last block')
    (ROOT/'dist/model.json').write_text(json.dumps(baseline,separators=(',',':')));shutil.copyfile(WORK/'aasist.onnx',ROOT/'dist/aasist.onnx');shutil.copyfile(WORK/'LICENSE',ROOT/'dist/vendor/AASIST-LICENSE.txt')
    e['baseline_conditions']=e['conditions'];e['conditions']=conditions;e['selective']=selective;e['thresholds']=limits;e['model']='Frozen AASIST neural detector + separately fitted sigmoid calibration; trained MFCC-SVM disagreement check';e['neural_tuning_auc']=auc;e['team_generated_unseen_engine']=team
    e['limitations']=[x for x in e['limitations'] if 'No pretrained' not in x]+['AASIST backbone is pretrained, not retrained here. Its calibration layer is fitted here.','Second candidate was introduced after observing baseline failures. This is an exploratory comparison; obtain a fresh locked test set for confirmatory claims.','The two-model disagreement rule is conservative but does not guarantee uncertainty on every unseen generator.','Neural whole-recording score aggregates consecutive 4.04-second blocks; timeline remains the separately trained SVM on 2-second windows.']
    e['neural_source']=dict(url='https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST',sha256=hashlib.sha256((ROOT/'dist/aasist.onnx').read_bytes()).hexdigest(),upstream='https://github.com/clovaai/aasist',license='MIT')
    (ROOT/'dist/evaluation.json').write_text(json.dumps(e,indent=2));(ROOT/'research/aasist-test-predictions.json').write_text(json.dumps([dict(id=r['utt_id'],label=int(y[j]),probability=float(p[j]),raw_score=float(raw[j]),accepted=bool(accepted[j])) for j,r in enumerate(groups['test'])],indent=2))
    x=load(groups['test'][0])[:48000];(ROOT/'tests/neural-parity.json').write_text(json.dumps(dict(wave=x.tolist(),raw_score=float(scores([x])[0]))))
    print('DONE',dict(clean=conditions['clean'],selective=selective,team=team),flush=True)
if __name__=='__main__':main()
