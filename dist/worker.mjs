import {analyze} from './dsp.mjs';
let session;
self.onmessage=async e=>{try{
 const wave=new Float32Array(e.data.wave),m=e.data.model;
 let result=analyze(wave,m);
 if(m.neural&&e.data.sourceMode==='upload'){const {createDetector,screen}=await import('./neural.mjs');session??=await createDetector();result=await screen(wave,m,session);}
 self.postMessage({id:e.data.id,result});
 }catch(err){self.postMessage({id:e.data.id,error:err.message||'Analysis failed.'});}};
