import assert from 'node:assert/strict';
import fs from 'node:fs';
import {applySourcePolicy,REPLAY_WARNING} from '../dist/policy.mjs';
import {applyNeuralSafetyPolicy} from '../dist/neural.mjs';

const base={label:'real',probability:0.01,warnings:[],windows:[{start:0,end:2,score:-1}]};
const replay=applySourcePolicy(base,'record');
assert.equal(replay.label,'uncertain');
assert.equal(replay.baseLabel,'real');
assert.equal(replay.policyReason,'microphone-replay-risk');
assert.equal(replay.probability,0.01);
assert(replay.warnings.includes(REPLAY_WARNING));
assert.equal(base.label,'real');
assert.deepEqual(base.warnings,[]);

const detected=applySourcePolicy({...base,label:'fake',probability:0.97},'record');
assert.equal(detected.label,'fake');
assert.equal(detected.replayRisk,true);

const upload=applySourcePolicy(base,'upload');
assert.equal(upload.label,'real');
assert.equal(upload.replayRisk,false);

const uncertain=applySourcePolicy({...base,label:'uncertain'},'record');
assert.equal(uncertain.label,'uncertain');
assert.equal(uncertain.policyReason,undefined);

const neural=JSON.parse(fs.readFileSync(new URL('../dist/neural-config.json',import.meta.url),'utf8'));
assert.equal(neural.file,'aasist.onnx');
assert(Number.isFinite(neural.calibration.slope));
const thresholds=neural.thresholds;
const challenged=applyNeuralSafetyPolicy({...base,warnings:[]},0.9,thresholds);
assert.equal(challenged.label,'uncertain');
assert.equal(challenged.policyReason,'neural-safety-challenge');
const preserved=applyNeuralSafetyPolicy({...base,label:'fake',probability:0.97,warnings:[]},0.05,thresholds);
assert.equal(preserved.label,'fake');
assert.equal(preserved.probability,0.97);
const cleared=applyNeuralSafetyPolicy({...base,warnings:[]},0.05,thresholds);
assert.equal(cleared.label,'real');
const priorUncertain=applyNeuralSafetyPolicy({...base,label:'uncertain',warnings:[]},0.05,thresholds);
assert.equal(priorUncertain.label,'uncertain');
const worker=fs.readFileSync(new URL('../dist/worker.mjs',import.meta.url),'utf8');
assert(worker.includes("e.data.sourceMode==='upload'"));
const app=fs.readFileSync(new URL('../dist/app.mjs',import.meta.url),'utf8');
assert(app.includes('sourceMode:mode'));
assert(app.includes("fetch('neural-config.json')"));

fs.writeFileSync(new URL('policy-results.json',import.meta.url),JSON.stringify({
  passed:11,
  failed:0,
  checks:[
    'recorded human-leaning output abstains and preserves its base result',
    'recorded synthetic output remains synthetic',
    'uploaded human output remains human',
    'existing uncertainty remains uncertainty',
    'AASIST upload configuration is valid',
    'AASIST can challenge a primary human result',
    'AASIST cannot clear a primary synthetic warning',
    'AASIST preserves the primary calibrated estimate',
    'existing primary uncertainty remains uncertainty after AASIST',
    'worker routes the neural second opinion only for uploads',
    'application passes its source mode and loads neural configuration'
  ]
},null,2)+'\n');
console.log('PASS 11 source-policy checks');
