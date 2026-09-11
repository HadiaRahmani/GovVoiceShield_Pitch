"""Generate stock-voice speech from the locked, text-matched corpus specification."""
from features import *
from config import ROOT,WORK
import asyncio,json,hashlib,subprocess
import edge_tts,imageio_ffmpeg,soundfile as sf
V2=WORK/'v2';D=ROOT/'research/v2'
async def main():
    planned=json.loads((D/'manifest.json').read_text());lock=json.loads((D/'data-lock.json').read_text())
    assert hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest()==lock['manifest_sha256']
    sem=asyncio.Semaphore(2);ff=imageio_ffmpeg.get_ffmpeg_exe()
    async def run(r):
        if r['platform']!='edge':return r
        target=V2/r['file']
        async with sem:
            if not target.exists():
                mp3=V2/(r['id']+'.mp3')
                for attempt in range(4):
                    try:
                        await edge_tts.Communicate(r['text'].capitalize(),r['voice']).save(str(mp3));break
                    except Exception:
                        if attempt==3:raise
                        await asyncio.sleep(2**attempt)
                subprocess.run([ff,'-hide_banner','-loglevel','error','-y','-i',str(mp3),'-ar','16000','-ac','1',str(target)],check=True,timeout=30)
            return dict(r,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),duration=sf.info(target).duration)
    results=[]
    for task in asyncio.as_completed([run(r) for r in planned]):
        results.append(await task)
        if len(results)%40==0:print('Prepared',len(results),flush=True)
    results.sort(key=lambda r:r['id'])
    # IDs/splits are immutable. Only generated file hashes and durations were filled.
    assert [(r['id'],r['split']) for r in results]==[(r['id'],r['split']) for r in planned]
    file=D/'completed-manifest.json';file.write_text(json.dumps(results,indent=2))
    hashes={}
    for r in results:
        if r['sha256'] in hashes:assert hashes[r['sha256']]==r['split'],'Cross-split duplicate'
        hashes[r['sha256']]=r['split']
    (D/'completed-lock.json').write_text(json.dumps(dict(parent_manifest_sha256=lock['manifest_sha256'],completed_manifest_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),test_ids=lock['test_ids'],cross_split_exact_duplicates=0),indent=2))
    print('Generation and integrity checks complete',flush=True)
if __name__=='__main__':asyncio.run(main())
