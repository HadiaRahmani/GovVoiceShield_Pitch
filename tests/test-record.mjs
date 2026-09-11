import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createVoiceRecorder,pcmWav} from '../dist/record.mjs';
import {inspectFile,readPCM} from '../dist/input.mjs';
let passed=0;const test=async(n,f)=>{await f();passed++;console.log('PASS',n);};
let current,clock=0,tick,stops=0,blobs=[],errors=[];const track={stop(){stops++;}},stream={getTracks:()=>[track],getAudioTracks:()=>[track]};
class Recorder{static isTypeSupported(){return true;}constructor(){current=this;this.state='inactive';this.mimeType='audio/webm';}start(){this.state='recording';}stop(){this.state='inactive';this.ondataavailable?.({data:new Blob(['audio'])});this.onstop?.();}}
const make=(extra={})=>createVoiceRecorder({devices:{getUserMedia:async()=>stream},Recorder,secure:true,now:()=>clock,every:fn=>{tick=fn;return 1;},clear:()=>{},onComplete:b=>blobs.push(b),onError:e=>errors.push(e),...extra});
await test('Recording requires explicit start and stops tracks',async()=>{const r=make();assert.equal(r.state,'idle');await r.start();assert.equal(r.state,'recording');r.stop();assert.equal(r.state,'idle');assert.equal(blobs.length,1);assert(stops>0);});
await test('Automatic stop at 15 seconds',async()=>{clock=0;const r=make();await r.start();clock=15000;tick();assert.equal(r.state,'idle');});
await test('Cancel discards audio',async()=>{const n=blobs.length,r=make();await r.start();r.cancel();assert.equal(blobs.length,n);assert.equal(r.state,'idle');});
await test('Late permission grant after cancellation releases microphone',async()=>{let resolve;const n=stops,r=make({devices:{getUserMedia:()=>new Promise(ok=>resolve=ok)}});const pending=r.start();r.cancel();resolve(stream);await pending;assert.equal(r.state,'idle');assert.equal(stops,n+1);});
await test('Permission denial is recoverable',async()=>{const r=make({devices:{getUserMedia:async()=>{throw Object.assign(Error(),{name:'NotAllowedError'});}}});await r.start();assert.match(errors.at(-1),/denied/);assert.equal(r.state,'idle');});
await test('Insecure contexts give an actionable error',async()=>{await make({secure:false}).start();assert.match(errors.at(-1),/HTTPS/);});
await test('Recorder error releases tracks',async()=>{const r=make();await r.start();const n=stops;current.onerror();assert(stops>n);assert.equal(r.state,'idle');});
await test('Oversized recording is discarded',async()=>{const r=make(),n=blobs.length;await r.start();current.ondataavailable({data:new Blob([new Uint8Array(5*1024*1024+1)])});assert.equal(r.state,'idle');assert.equal(blobs.length,n);assert.match(errors.at(-1),/size/);});
await test('Captured PCM round trips through existing upload parser',async()=>{const x=Float32Array.from({length:48000},(_,i)=>.2*Math.sin(i*.1)),a=await pcmWav(x).arrayBuffer(),info=inspectFile(a),y=readPCM(a,info);assert.equal(info.rate,16000);assert.equal(y.length,x.length);assert(Math.max(...y.map((v,i)=>Math.abs(v-x[i])))<.00005);});
await test('Too-short and nonfinite captures rejected',()=>{assert.throws(()=>pcmWav(new Float32Array(100)));assert.throws(()=>pcmWav(Float32Array.from({length:48000},()=>NaN)));});
fs.writeFileSync(new URL('record-results.json',import.meta.url),JSON.stringify({passed,failed:0,scope:'Mocked microphone lifecycle and real PCM parsing. Physical microphone/browser permission flow not exercised.'},null,2));
