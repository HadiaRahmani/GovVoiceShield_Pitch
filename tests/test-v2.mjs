import fs from 'node:fs';
import assert from 'node:assert/strict';
import {features,predict,analyze} from '../dist/dsp.mjs';
import {inspectFile,readPCM} from '../dist/input.mjs';
const root=new URL('../',import.meta.url),read=p=>JSON.parse(fs.readFileSync(new URL(p,root),'utf8'));
const model=read('dist/model.json'),fixture=read('tests/v2-parity.json'),checks=[];
function test(name,fn){fn();checks.push(name);console.log('PASS',name);}
test('Current v2 model loaded',()=>assert.equal(model.version,'govvoice-svm-v2'));
test('Python/JS features agree',()=>assert(Math.max(...features(fixture.wave).map((v,i)=>Math.abs(v-fixture.features[i])))<1e-8));
test('Python/JS calibrated probability agrees',()=>assert(Math.abs(predict(fixture.wave,model).probability-fixture.probability)<1e-8));
let message;globalThis.self={postMessage:m=>message=m};await import('../dist/worker.mjs');
const results=[];
for(const clip of read('dist/reel.json')){const b=fs.readFileSync(new URL('dist/'+clip.file,root)),ab=b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),wave=readPCM(ab,inspectFile(ab)),start=performance.now();await self.onmessage({data:{id:clip.id,wave:wave.buffer,model}});assert(!message.error,message.error);assert(['real','fake','uncertain'].includes(message.result.label));results.push({id:clip.id,truth:clip.label,label:message.result.label,probability:message.result.probability,seconds:(performance.now()-start)/1000});}
checks.push('All ten reel files decode and run through actual worker handler');
const failFile=fs.readFileSync(new URL('dist/audio/known-error.wav',root)),fab=failFile.buffer.slice(failFile.byteOffset,failFile.byteOffset+failFile.byteLength);
test('Known human false alarm is reproduced and honestly labelled',()=>{assert.equal(read('dist/failure.json').label,'real');assert.equal(analyze(readPCM(fab,inspectFile(fab)),model).label,'fake');});
const mp3=fs.readFileSync(new URL('tests/mp3-fixture.mp3',root));test('MP3 fixture header is accepted',()=>assert.equal(inspectFile(mp3.buffer.slice(mp3.byteOffset,mp3.byteOffset+mp3.byteLength)).type,'mp3'));
const wave=Float32Array.from(fixture.wave);test('Repeat inference is deterministic',()=>assert.deepEqual(analyze(wave,model),analyze(wave,model)));
for(const [name,x] of [['silence',new Float32Array(48000)],['short',new Float32Array(100)],['overlong',new Float32Array(976000)],['nonfinite',Float32Array.from({length:48000},()=>NaN)]]){await self.onmessage({data:{id:1,wave:x.buffer,model}});assert(message.error);checks.push('Worker returns recoverable '+name+' error');}
await self.onmessage({data:{id:2,wave:wave.buffer,model}});test('Valid input succeeds after errors',()=>assert(!message.error));
const long=Float32Array.from({length:960000},(_,i)=>wave[i%wave.length]),start=performance.now();const r=analyze(long,model);const seconds60=(performance.now()-start)/1000;test('60-second input completes before UI timeout',()=>{assert(seconds60<45);assert.equal(r.windows.length,59);});
const output={passed:checks.length,failed:0,checks,reel:results,seconds60,browserVisualPlaybackTested:false,notice:'Worker handler tested under Node; not a browser interaction test.'};fs.writeFileSync(new URL('tests/v2-results.json',root),JSON.stringify(output,null,2));console.log(JSON.stringify(output,null,2));
