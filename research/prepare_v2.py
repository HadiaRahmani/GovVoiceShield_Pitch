"""Create and seal new source-group partitions BEFORE extracting model features.
Hume is reserved as an unseen platform. Same numeric synthetic source IDs are
grouped across platforms conservatively; real clips are grouped by source video.
LibriSpeech adds independent speakers and text-matched stock Edge synthesis.
"""
from features import *
from config import ROOT,WORK
import urllib.request,json,hashlib,time,io,re
from concurrent.futures import ThreadPoolExecutor,as_completed
import soundfile as sf,pyarrow.parquet as pq
V2=WORK/'v2';V2.mkdir(exist_ok=True)
def get(url):
    for a in range(6):
        try:
            with urllib.request.urlopen(url,timeout=90) as r:return r.read()
        except Exception:
            if a==5:raise
            time.sleep(min(30,2**a))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gs_split(name):
    prefix,index=name.split('_')[:2];i=int(index)
    if prefix=='yt':
        if i in [9,3,5]:return 'test'
        if i in [1,12]:return 'calibration'
        if i in [4,6]:return 'tune'
        return 'train'
    if prefix=='hu':return 'test'
    return {4:'tune',5:'calibration',6:'test'}.get(i%10,'train')
def main():
    dest=ROOT/'research/v2';dest.mkdir(exist_ok=True)
    if (dest/'data-lock.json').exists():raise RuntimeError('Data already sealed; refusing to overwrite.')
    repo='garystafford/deepfake-audio-detection';rev=json.loads(get('https://huggingface.co/api/datasets/'+repo))['sha'];items=[]
    (dest/'CONTEMPORARY_DATASET_CARD.md').write_bytes(get('https://huggingface.co/datasets/'+repo+'/raw/'+rev+'/README.md'))
    for cat in ['real','fake']:
        listing=json.loads(get('https://huggingface.co/api/datasets/'+repo+'/tree/'+rev+'/'+cat+'?limit=1000'))
        for item in listing:
            if item['type']!='file' or not item['path'].endswith('.flac'):continue
            name=Path(item['path']).name;prefix,index=name.split('_')[:2]
            items.append(dict(id='gs-'+name[:-5],path=item['path'],file='gs-'+name,split=gs_split(name),label=int(cat=='fake'),source_group=prefix+'_'+index,text_group='gs-script-'+index if cat=='fake' else prefix+'_'+index,platform=prefix if cat=='fake' else 'human',dataset=repo,revision=rev,license='CC-BY-4.0 (dataset author declaration)',label_basis='Publisher real/fake directory'))
    # Lock intended assignment independently of any model output.
    (dest/'planned-groups.json').write_text(json.dumps(items,indent=2))
    def download(r):
        p=V2/r['file']
        if not p.exists():p.write_bytes(get('https://huggingface.co/datasets/'+repo+'/resolve/'+rev+'/'+r['path']))
        r['sha256']=digest(p);info=sf.info(p);r['duration']=info.duration
        return r
    records=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,f in enumerate(as_completed([pool.submit(download,r) for r in items])):
            records.append(f.result())
            if (i+1)%300==0:print('Contemporary clips',i+1,flush=True)
    libri='openslr/librispeech_asr';lrev=json.loads(get('https://huggingface.co/api/datasets/'+libri))['sha']
    (dest/'LIBRISPEECH_CARD.md').write_bytes(get('https://huggingface.co/datasets/'+libri+'/raw/'+lrev+'/README.md'))
    voice_map={'train':['en-US-AriaNeural','en-US-ChristopherNeural','en-GB-SoniaNeural','en-GB-RyanNeural'],'tune':['en-US-EricNeural','en-GB-LibbyNeural'],'calibration':['en-US-MichelleNeural','en-US-SteffanNeural'],'test':['en-US-GuyNeural','en-US-JennyNeural']}
    for partition in ['validation.clean','test.clean']:
        p=V2/(partition+'.parquet')
        if not p.exists():p.write_bytes(get('https://huggingface.co/datasets/'+libri+'/resolve/'+lrev+'/all/'+partition+'/0000.parquet'))
        selected={}
        for batch in pq.ParquetFile(p).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                words=len(row['text'].split());speaker=str(row['speaker_id']);raw=row['audio']['bytes'];info=sf.info(io.BytesIO(raw))
                if not(8<=words<=28 and 3<=info.duration<=12):continue
                selected.setdefault(speaker,[])
                if len(selected[speaker])<2:selected[speaker].append(row)
        speakers=sorted([s for s,a in selected.items() if len(a)>=2],key=lambda s:hashlib.sha256(('govvoice-v2-'+s).encode()).hexdigest())
        assert len(speakers)>=30
        for j,speaker in enumerate(speakers[:40]):
            split='test' if partition=='test.clean' else 'train' if j<20 else 'tune' if j<30 else 'calibration'
            chosen=selected[speaker][:1 if split=='test' else 2]
            for row in chosen:
                id='libri-'+row['id'];file=id+'.flac';(V2/file).write_bytes(row['audio']['bytes'])
                rec=dict(id=id,file=file,split=split,label=0,source_group='libri-speaker-'+speaker,text_group='libri-text-'+row['id'],platform='human',dataset=libri,revision=lrev,license='CC-BY-4.0',label_basis='LibriSpeech human-read corpus',text=row['text'],sha256=digest(V2/file),duration=sf.info(V2/file).duration)
                records.append(rec)
                voice=voice_map[split][len([r for r in records if r.get('platform')=='edge' and r['split']==split])%len(voice_map[split])]
                records.append(dict(rec,id='edge-'+row['id'],file='edge-'+row['id']+'.wav',label=1,source_group='edge-voice-'+voice,platform='edge',dataset='Project-generated Edge stock neural speech',label_basis='Generated by research/generate_v2.py',voice=voice,sha256=None,duration=None,license='Stock speech synthesis output; source text CC-BY-4.0'))
    records.sort(key=lambda r:r['id'])
    # No exact text or source group may cross development and final-test boundaries.
    for key in ['source_group','text_group']:
        buckets={s:{r[key] for r in records if r['split']==s} for s in ['train','tune','calibration','test']}
        # Hume is held out entirely; its numeric IDs can coincide with other platforms.
        if key=='text_group':
            buckets={s:{r[key] for r in records if r['split']==s and r['platform']!='hu'} for s in buckets}
        for a in buckets:
            for b in buckets:
                if a<b:assert not buckets[a]&buckets[b],(key,a,b)
    (dest/'manifest.json').write_text(json.dumps(records,indent=2))
    lock=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),manifest_sha256=digest(dest/'manifest.json'),test_ids=[r['id'] for r in records if r['split']=='test'],policy='No final-test features or predictions before model/threshold freeze. Hume entirely held out. Source video groups disjoint. Synthetic numeric IDs conservatively grouped across other platforms; shared Hume text cannot be excluded because transcripts are unavailable. Edge text/speakers disjoint. Legacy samples never reused as new final test.',test_access='Metadata and decoding for integrity only until freeze; no test-based selection or tuning.')
    (dest/'data-lock.json').write_text(json.dumps(lock,indent=2))
    from collections import Counter
    print('SEALED',Counter((r['split'],r['label']) for r in records),flush=True)
if __name__=='__main__':main()
