"""Download a deterministic, evenly spread ORIGINAL-only ASVspoof19 LA subset.
Source: abdulahh35/ANC-Spoof, revision resolved and recorded at acquisition.
No generated/codec-derived copies cross the official partitions.
"""
import sys, os, json, time, hashlib, urllib.request, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import ROOT,CACHE
CACHE.mkdir(parents=True,exist_ok=True)
REPO='abdulahh35/ANC-Spoof'
def get(url):
    for attempt in range(7):
        try:
            with urllib.request.urlopen(url,timeout=55) as r: return r.read()
        except Exception:
            if attempt==6: raise
            time.sleep(min(60,5*2**attempt))
def page(task):
    split,offset=task
    q=urllib.parse.urlencode(dict(dataset=REPO,config='ASVspoof2019',split=split,offset=offset,length=50))
    cache=CACHE/f'metadata-{split}-{offset}.json'
    d=None
    if cache.exists():
        saved=json.loads(cache.read_text())
        src=saved['rows'][0]['row']['audio'][0]['src']
        expiry=float(urllib.parse.parse_qs(urllib.parse.urlparse(src).query).get('Expires',['0'])[0])
        if expiry>time.time()+900:d=saved
    if d is None:
        time.sleep(2)
        d=json.loads(get('https://datasets-server.huggingface.co/rows?'+q))
        cache.write_text(json.dumps(d))
    return [dict(x['row'],row_index=x['row_idx']) for x in d['rows']]
def download(row):
    p=CACHE/(row['utt_id']+'.flac')
    if not p.exists(): p.write_bytes(get(row['audio'][0]['src']))
    return {k:v for k,v in row.items() if k!='audio'}|dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
    revision=json.loads(get('https://huggingface.co/api/datasets/'+REPO))['sha']
    tasks=[]
    for split,real,total,pages in [('train',2580,25380,12),('dev',2548,24844,6),('eval',7355,71237,6)]:
        for lo,hi in [(0,real),(real,total)]:
            for j in range(pages): tasks.append((split,int(lo+(hi-lo-50)*j/(pages-1))))
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(page,t) for t in tasks]): rows+=f.result()
    assert all(r['codec']=='Original' for r in rows)
    assert len({r['utt_id'] for r in rows})==len(rows)
    print('Metadata ready:',len(rows),flush=True)
    records=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,f in enumerate(as_completed([pool.submit(download,r) for r in rows])):
            records.append(f.result())
            if (i+1)%200==0: print('Downloaded',i+1,flush=True)
    records.sort(key=lambda r:(r['split'],r['row_index']))
    (ROOT/'research'/'dataset-manifest.json').write_text(json.dumps(dict(repository=REPO,revision=revision,selection='50 consecutive rows at evenly spaced offsets per class in official original partitions; deterministic convenience subset, not a full benchmark',records=records),indent=2))
    print('Complete',len(records),flush=True)
if __name__=='__main__': main()
