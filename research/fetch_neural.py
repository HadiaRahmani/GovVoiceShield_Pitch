import urllib.request,json,hashlib
from config import ROOT,WORK
source=json.loads((ROOT/'research/neural-source.json').read_text())
for file in ['aasist.onnx','LICENSE']:
    url='https://huggingface.co/'+source['repository']+'/resolve/'+source['revision']+'/'+file
    b=urllib.request.urlopen(url,timeout=60).read()
    if file=='aasist.onnx':assert hashlib.sha256(b).hexdigest()==source['sha256'],'Model hash mismatch'
    (WORK/file).write_bytes(b)
print('Pinned model and licence downloaded and verified.')
