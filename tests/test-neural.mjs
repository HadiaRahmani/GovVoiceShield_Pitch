import assert from 'node:assert/strict';import fs from 'node:fs';
import {createDetector,neuralScore,screen} from '../dist/neural.mjs';
import {inspectFile,readPCM} from '../dist/input.mjs';
const root=new URL('../',import.meta.url),read=p=>JSON.parse(fs.readFileSync(new URL(p,root))),model=read('research/archive/v1/model.json'),fixture=read('tests/neural-parity.json'),reel=read('dist/reel.json');
const bytes=p=>{const b=fs.readFileSync(new URL(p,root));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
let passed=0;const record=name=>{passed++;console.log('PASS',name);};
const session=await createDetector(new Uint8Array(bytes('dist/aasist.onnx')));
const wave=Float32Array.from(fixture.wave),score=await neuralScore(wave,session);const error=Math.abs(score.score-fixture.raw_score);assert(error<5e-4,'Native/WASM drift '+error);record('Native ONNX to WASM neural parity');
const first=await screen(wave,model,session),second=await screen(wave,model,session);assert.deepEqual(first,second);record('Neural repeatability');
await assert.rejects(()=>screen(new Float32Array(48000),model,session),/No usable/);record('Silence rejected before neural inference');
const demo=[];for(const clip of reel){const b=bytes('dist/'+clip.file),x=readPCM(b,inspectFile(b)),start=performance.now(),r=await screen(x,model,session);assert(Number.isFinite(r.probability)&&r.probability>=0&&r.probability<=1);assert(['real','fake','uncertain'].includes(r.label));assert(r.windows.length>0);demo.push({id:clip.id,verifiedLabel:clip.label,assessment:r.label,probability:r.probability,seconds:(performance.now()-start)/1000});record('Actual inference on '+clip.id);}
const mp3=bytes('tests/mp3-fixture.mp3');assert.equal(inspectFile(mp3).type,'mp3');record('Generated MP3 header and duration');
const long=Float32Array.from({length:960000},(_,i)=>wave[i%wave.length]),start=performance.now(),longResult=await screen(long,model,session),elapsed=(performance.now()-start)/1000;assert.equal(longResult.neuralBlocks,15);assert(elapsed<45,'60 second audio exceeds worker budget on this machine: '+elapsed);record('Full 60-second recording processed within worker timeout');
fs.writeFileSync(new URL('tests/neural-recheck-results.json',root),JSON.stringify({passed,failed:0,nativeWasmScoreError:error,sixtySecondAudioRuntime:elapsed,demo,browserVisualAndPlaybackQA:'Not performed; tests run actual shared inference code with ONNX Runtime WASM in Node. Browser event and audio playback behavior require separate interactive verification.',webmcpValidation:'No supported WebMCP validation context was available; optional tools are unverified.'},null,2));
await session.release();

