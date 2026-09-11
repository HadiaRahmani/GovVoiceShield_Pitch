from features import *
import asyncio,json,subprocess
import edge_tts,imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1]
items=[dict(id='team-01',voice='en-US-GuyNeural',text='This is a fictional demonstration announcement. The community service centre will open at nine tomorrow morning. Please confirm opening times using the official directory before travelling.'),dict(id='team-02',voice='en-US-JennyNeural',text='This is a fictional demonstration announcement. Scheduled maintenance will affect the public information portal tonight. For urgent assistance, use the contact details already published on the official website.')]
async def main():
    audio=ROOT/'dist/audio';audio.mkdir(exist_ok=True)
    for item in items:
        mp3=WORK/(item['id']+'.mp3')
        await edge_tts.Communicate(item['text'],item['voice']).save(str(mp3))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-hide_banner','-loglevel','error','-i',str(mp3),'-ar','16000','-ac','1',str(audio/(item['id']+'.wav'))],check=True,timeout=30)
        print('Generated',item['id'],flush=True)
    (ROOT/'research/generated-speech.json').write_text(json.dumps(items,indent=2))
if __name__=='__main__':asyncio.run(main())
