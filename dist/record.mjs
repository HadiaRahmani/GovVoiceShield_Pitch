// Microphone access occurs only after an explicit Start click. No network calls.
export function createVoiceRecorder({onState=()=>{},onComplete=()=>{},onError=()=>{},devices=globalThis.navigator?.mediaDevices,Recorder=globalThis.MediaRecorder,secure=globalThis.isSecureContext,now=()=>Date.now(),every=setInterval,clear=clearInterval}={}){
 let generation=0,stream=null,recorder=null,timer=null,state='idle',started=0,chunks=[],bytes=0;
 const emit=(s,seconds=0)=>{state=s;onState(s,seconds);};
 const release=()=>{if(timer!==null)clear(timer);timer=null;stream?.getTracks().forEach(t=>t.stop());stream=null;};
 function cancel(){generation++;release();if(recorder){recorder.ondataavailable=null;recorder.onstop=null;recorder.onerror=null;if(recorder.state!=='inactive')recorder.stop();}recorder=null;chunks=[];emit('idle');}
 function stop(){if(state!=='recording')return;emit('stopping');recorder.stop();release();}
 async function start(){cancel();if(!secure||!devices?.getUserMedia||!Recorder){onError('Microphone recording needs HTTPS or localhost and a supported browser. Try Chrome, or upload a WAV/MP3.');return;}const token=generation;emit('requesting');try{
  const granted=await devices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});
  if(token!==generation){granted.getTracks().forEach(t=>t.stop());return;}stream=granted;
  const mime=['audio/webm;codecs=opus','audio/mp4','audio/webm'].find(t=>Recorder.isTypeSupported(t));recorder=mime?new Recorder(stream,{mimeType:mime}):new Recorder(stream);chunks=[];bytes=0;
  recorder.ondataavailable=e=>{if(token!==generation)return;if(e.data.size){bytes+=e.data.size;if(bytes>5*1024*1024){cancel();onError('Recording exceeded the size limit. Try again.');return;}chunks.push(e.data);}};
  recorder.onerror=()=>{if(token===generation){cancel();onError('Microphone recording failed. Check the device and try again.');}};
  recorder.onstop=()=>{if(token!==generation)return;const blob=new Blob(chunks,{type:recorder.mimeType});release();recorder=null;chunks=[];emit('idle');if(!blob.size){onError('No audio was captured. Check your microphone.');return;}onComplete(blob);};
  stream.getAudioTracks().forEach(t=>t.onended=()=>{if(token===generation&&state==='recording')stop();});
  recorder.start(250);started=now();emit('recording',0);timer=every(()=>{const seconds=(now()-started)/1000;onState('recording',seconds);if(seconds>=15)stop();},200);
 }catch(e){if(token!==generation)return;cancel();onError(e.name==='NotAllowedError'?'Microphone permission was denied. Allow it in your browser’s site settings, or upload a file.':e.name==='NotFoundError'?'No microphone was found. Connect one or upload a file.':'Could not open the microphone. Close other recording apps and try again.');}}
 return {start,stop,cancel,get state(){return state;}};
}

export function pcmWav(wave,rate=16000){
 if(wave.length<rate*2||wave.length>rate*16)throw Error('Record at least 2 seconds of speech, up to 15 seconds.');
 const a=new ArrayBuffer(44+wave.length*2),v=new DataView(a),str=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
 str(0,'RIFF');v.setUint32(4,a.byteLength-8,true);str(8,'WAVE');str(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);str(36,'data');v.setUint32(40,wave.length*2,true);
 for(let i=0;i<wave.length;i++){if(!Number.isFinite(wave[i]))throw Error('Recording contains invalid samples.');const x=Math.max(-1,Math.min(1,wave[i]));v.setInt16(44+i*2,Math.round(x*(x<0?32768:32767)),true);}return new Blob([a],{type:'audio/wav'});
}
