const REPLAY_WARNING='Microphone playback and room acoustics can conceal synthetic-speech artifacts. Use the original digital file when possible and verify consequential messages through an official channel.';

export function applySourcePolicy(input,sourceMode){
 const result={...input,warnings:Array.isArray(input?.warnings)?[...input.warnings]:[],baseLabel:input?.label,sourceMode,replayRisk:sourceMode==='record'};
 if(result.replayRisk&&result.label==='real'){
  result.label='uncertain';
  result.policyReason='microphone-replay-risk';
  if(!result.warnings.includes(REPLAY_WARNING))result.warnings.push(REPLAY_WARNING);
 }
 return result;
}

export {REPLAY_WARNING};
