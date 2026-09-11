from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,white
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'govvoice-shield';e=json.loads((P/'dist/evaluation.json').read_text());t=json.loads((P/'tests/v2-results.json').read_text());m=e['conditions']['clean'];pc=lambda x:f'{100*x:.1f}%'
W,H=960,540;navy=HexColor('#0c2335');teal=HexColor('#087f7c');ink=HexColor('#173549');muted=HexColor('#536779');pale=HexColor('#e8f4f2');line=HexColor('#dce6eb')
c=canvas.Canvas(str(ROOT/'GovVoice_Shield_Competition.pdf'),pagesize=(W,H));c.setTitle('GovVoice Shield | Measured audio screening prototype');c.setAuthor('GovVoice Shield project team')
def para(txt,x,y,w,size=17,color=ink,bold=False):
    s=ParagraphStyle('x',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,leading=size*1.35,textColor=color);p=Paragraph(txt,s);_,h=p.wrap(w,1000);p.drawOn(c,x,H-y-h);return y+h
def rect(x,y,w,h,color):c.setFillColor(color);c.roundRect(x,H-y-h,w,h,12,stroke=0,fill=1)
def start(n,k,title,subtitle):
    c.setFillColor(white);c.rect(0,0,W,H,fill=1,stroke=0);c.setFillColor(teal);c.rect(0,H-9,W,9,fill=1,stroke=0)
    para('GOVVOICE SHIELD  /  '+k,40,26,850,11,teal,True);para(title,40,53,880,30,ink,True);para(subtitle,40,101,880,15,muted)
    c.setStrokeColor(line);c.line(40,34,920,34);para('Research prototype • English audio • Screening supports official-channel verification',40,513,820,9,muted);para(f'{n} / 5',881,513,50,9,muted)
def source(label,url):
    para(label,40,489,880,9,muted);c.linkURL(url,(40,37,920,52),relative=0)
start(1,'TITLE','Would you trust this voice?','Deepfake & Synthetic Media Detection for Government Communications')
rect(40,153,880,245,navy);para('Listen first. Verify next.',66,179,820,35,white,True);para('A simple audio-screening prototype that lets an audience listen, make a call, then compare its judgment with a trained detector.',66,239,770,22,white);para('Human  /  Synthetic  /  Uncertain',66,332,790,19,HexColor('#75ded0'),True)
para('DELIVERY',40,426,270,11,teal,True);para('Upload + ten-sample reel',40,447,300,18,ink,True);para('EVIDENCE',355,426,250,11,teal,True);para('Locked test + source labels',355,447,290,18,ink,True);para('PRINCIPLE',675,426,230,11,teal,True);para('Show mistakes honestly',675,447,240,18,ink,True)
c.showPage()
start(2,'PROJECT OBJECTIVE','Help reviewers decide what needs verification','Target user: a government communications or security reviewer receiving a purported official voice message.')
rect(40,151,420,291,pale);para('The problem',62,173,370,22,teal,True);para('A recognizable voice can still be synthetic. Listening alone gives limited evidence, while uploading sensitive recordings to a third party adds friction.',62,215,370,19);para('The prototype adds a local screening step to manual review. It does not confirm identity, truthfulness or authorization.',62,331,370,17)
para('The review flow',501,168,410,22,ink,True)
for y,a,b in [(217,'01  Listen','Choose a reel sample or WAV/MP3.'),(278,'02  Make a call','Human, synthetic or not sure.'),(339,'03  Reveal and review','See the estimate, source label and timestamps.'),(411,'04  Verify','Use a known official channel for consequential claims.')]:
    para(a,501,y,390,17,teal,True);para(b,501,y+25,390,15)
source('Scope follows the supplied competition brief. No operational time saving or customer adoption has been measured.','https://www.openslr.org/12')
c.showPage()
start(3,'PROPOSED SOLUTION','Compact model. Local processing. Clear outcomes.','The final detector is trained here; it does not call a remote AI detection service.')
steps=[('INPUT','WAV / MP3\n2–60 seconds'),('FEATURES','16 kHz mono\n90 statistics'),('MODEL','RBF SVM\ntrained classifier'),('OUTPUT','Calibrated estimate\n+ uncertainty')]
for i,(a,b) in enumerate(steps):
    x=40+i*225;rect(x,159,205,101,pale);para(a,x+16,174,175,12,teal,True);para(b.replace('\n','<br/>'),x+16,197,175,16)
para('Why this model',40,287,420,21,ink,True);para('MFCCs, deltas and spectral statistics feed a small classifier. Nine configurations were compared on tuning data. Better contemporary examples replaced the slow, unreliable two-model combination.',40,326,414,17)
para('Useful innovation',501,287,419,21,ink,True);para('Blind audience guess → model reveal → source label. Two-second score windows offer places to listen again. Uncertain results stay visible; the demo reports errors instead of hiding them.',501,326,410,17)
para('Boundary: timestamps are review hints, not validated edit localization. No ENF or speaker-authentication claim.',40,446,875,14,muted)
source('Method reference: scikit-learn SVM documentation. Full parameters, sources and licences accompany the code.','https://scikit-learn.org/stable/modules/svm.html')
c.showPage()
start(4,'SOLUTION VALIDATION','Fix the model before opening the final test','Public data + project-generated stock speech; source assignments, hashes and model freeze are retained.')
for i,(a,b) in enumerate([('1,093','Training'),('247','Tuning'),('281','Calibration'),('485','Held out')]):
    x=40+i*225;rect(x,153,205,78,pale);para(a,x+17,163,175,26,teal,True);para(b,x+17,198,175,13)
para('80-clip primary test',40,253,420,20,ink,True);para('40 human + 40 text-matched synthetic recordings. Human speakers, text IDs and stock voices are disjoint from development. The Edge synthesis engine is represented in training.',40,290,420,16)
para('405-clip diagnostic',40,389,420,20,ink,True);para('Public source groups held out; Hume unseen during training. Unknown speaker identities/shared scripts limit independence.',40,423,420,15)
para('Engineering checks',501,253,419,20,ink,True);para(f"{t['passed']} v2 checks passed. Python/JavaScript parity, ten-file worker inference, error recovery and repeat runs. A 60-second recording took {t['seconds60']:.2f} s on this machine.",501,290,410,16)
para('Privacy + limits',501,389,419,20,ink,True);para('Local inference, bounded inputs, worker timeout and safe text rendering. Browser playback/accessibility and competition-device rehearsal remain unverified.',501,423,410,15)
source('Data: LibriSpeech + garystafford/deepfake-audio-detection; generated stock Edge speech. Full provenance in source package.','https://huggingface.co/datasets/garystafford/deepfake-audio-detection')
c.showPage()
start(5,'RESULTS & CONCLUSIONS','Measured progress, with a clear validation boundary','Primary test only • synthetic is positive • table uses forced 0.5 decisions, including uncertain files')
xs=[52,240,390,540,690,827];labels=['Condition','Accuracy','Precision','Recall','AUC','N'];rect(40,153,880,38,navy)
for x,s in zip(xs,labels):para(s,x,163,145,13,white,True)
for i,(key,label) in enumerate([('clean','Clean'),('noise_20db','Noise 20 dB'),('mp3_32kbps','MP3 32 kbps')]):
    r=e['conditions'][key];y=197+i*34
    for x,s in zip(xs,[label,pc(r['accuracy']),pc(r['precision']),pc(r['recall']),f"{r['roc_auc']:.3f}",str(r['n'])]):para(s,x,y,145,15)
para(f"Clean confusion matrix: {m['confusion_matrix']}  |  F1 {m['f1']:.3f}  |  Brier {m['brier']:.3f}",40,311,880,14)
para(f"Decision coverage {pc(e['selective']['coverage'])}: {e['selective']['uncertain']} uncertain; accepted-only accuracy {pc(e['selective']['accuracy']) if e['selective']['accuracy'] is not None else 'N/A'}.",40,340,880,15,teal,True)
para(f"Separate public diagnostic: {pc(e['public_diagnostic']['clean']['accuracy'])} accuracy. Hume detection: {pc(e['public_platforms']['hu']['binary_accuracy'])} of {e['public_platforms']['hu']['n']} synthetic clips. These are not pooled into the primary result.",40,373,880,15)
para('Conclusion: a working, inspectable research demo. Next evidence needed: independent government-domain/Arabic audio, unseen engines and device rehearsal. No production-readiness or perfect-score claim.',40,426,880,16,ink,True)
source('All numbers computed by research/evaluate_v2.py. Small paired study; calibration and clip intervals do not establish operational reliability.','https://scikit-learn.org/stable/modules/calibration.html')
c.save();print(ROOT/'GovVoice_Shield_Competition.pdf')
