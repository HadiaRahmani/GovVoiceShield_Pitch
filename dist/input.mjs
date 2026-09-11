// Header-level limits run BEFORE browser audio decoding.
export function inspectFile(buffer){
 if(!(buffer instanceof ArrayBuffer)||buffer.byteLength<12)throw Error('This file is empty or is not a supported audio recording.');
 if(buffer.byteLength>20*1024*1024)throw Error('Choose a file no larger than 20 MB.');
 const v=new DataView(buffer),u=new Uint8Array(buffer),str=(p,n)=>String.fromCharCode(...u.subarray(p,p+n));
 if(str(0,4)==='RIFF'&&str(8,4)==='WAVE'){
  let pos=12,fmt=null,data=null;
  while(pos+8<=v.byteLength){const name=str(pos,4),size=v.getUint32(pos+4,true),start=pos+8;if(size>v.byteLength-start)throw Error('The WAV file is truncated. Export a complete recording.');if(name==='fmt '){if(size<16)throw Error('Invalid WAV format.');fmt={format:v.getUint16(start,true),channels:v.getUint16(start+2,true),rate:v.getUint32(start+4,true),align:v.getUint16(start+12,true),bits:v.getUint16(start+14,true)};}if(name==='data')data={start,size};pos=start+size+(size%2);}
  if(!fmt||!data||![1,3].includes(fmt.format)||![8,16,24,32].includes(fmt.bits)||fmt.channels<1||fmt.channels>8||fmt.rate<8000||fmt.rate>192000||fmt.align!==fmt.channels*fmt.bits/8||(fmt.format===3&&fmt.bits!==32))throw Error('Use a standard PCM WAV or MP3 recording.');
  const duration=data.size/fmt.align/fmt.rate;if(duration<2||duration>60)throw Error('Choose 2–60 seconds of audio.');return {type:'wav',duration,...fmt,data};
 }
 let pos=0;if(str(0,3)==='ID3'){if(u[6]&128||u[7]&128||u[8]&128||u[9]&128)throw Error('Invalid MP3 metadata.');pos=10+((u[6]<<21)|(u[7]<<14)|(u[8]<<7)|u[9]);if(u[5]&16)pos+=10;}
 let frames=0,duration=0,skipped=0,rate=null,channels=null;
 while(pos+4<u.length){let h=v.getUint32(pos,false);if((h>>>21)!==2047){pos++;if(++skipped>4096)break;continue;}const ver=(h>>>19)&3,layer=(h>>>17)&3,bi=(h>>>12)&15,si=(h>>>10)&3,pad=(h>>>9)&1;if(ver===1||layer!==1||bi===0||bi===15||si===3){pos++;continue;}let sr=[44100,48000,32000][si]/(ver===3?1:ver===2?2:4);let kb=(ver===3?[0,32,40,48,56,64,80,96,112,128,160,192,224,256,320]:[0,8,16,24,32,40,48,56,64,80,96,112,128,144,160])[bi];let size=Math.floor((ver===3?144000:72000)*kb/sr)+pad;if(pos+size>u.length)break;if(rate===null){rate=sr;channels=((h>>>6)&3)===3?1:2;}frames++;duration+=(ver===3?1152:576)/sr;if(duration>60.3)throw Error('Choose a recording no longer than 60 seconds.');pos+=size;skipped=0;}
 if(u.length-pos>4096)throw Error('The MP3 contains unrecognized trailing data. Export a standard MP3 or WAV.');if(frames<3)throw Error('Unsupported or damaged file. Choose a WAV or MP3 recording.');if(duration<2)throw Error('Use at least 2 seconds of speech.');return {type:'mp3',duration,rate,channels};
}
export function readPCM(buffer,info){const v=new DataView(buffer),count=Math.floor(info.data.size/info.align),out=new Float32Array(count);for(let i=0;i<count;i++){let sum=0;for(let c=0;c<info.channels;c++){let p=info.data.start+i*info.align+c*info.bits/8,val;if(info.format===3)val=v.getFloat32(p,true);else if(info.bits===8)val=(v.getUint8(p)-128)/128;else if(info.bits===16)val=v.getInt16(p,true)/32768;else if(info.bits===24){let k=v.getUint8(p)|(v.getUint8(p+1)<<8)|(v.getUint8(p+2)<<16);val=((k<<8)>>8)/8388608;}else val=v.getInt32(p,true)/2147483648;if(!Number.isFinite(val))throw Error('Audio contains invalid samples.');sum+=val;}out[i]=sum/info.channels;}return out;}

