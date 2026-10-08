"""ChatGPT 訂閱繁中 → 日文實測；只送測試檔，不開啟麥克風。

  .venv/Scripts/python.exe scripts/verify/verify_chatgpt_live.py
結果存放忽略的 out/chatgpt-ja/；不記錄 token 或帳戶識別資訊。
"""
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vlt.session.base import SessionConfig, create_session


async def main():
    out = ROOT / 'out' / 'chatgpt-ja'
    out.mkdir(parents=True, exist_ok=True)
    texts, chunks = [], []
    input_started = None
    session = create_session(SessionConfig(provider='chatgpt', source_lang='zh', target_lang='ja', output_audio=True))
    session.on_event = lambda name, data: print(name, data, flush=True)

    def on_text(delta):
        texts.append({'at': round(time.perf_counter() - input_started, 3) if input_started else None,
                      'source': delta.source, 'text': delta.confirmed, 'final': delta.is_final})
        if delta.is_final:
            print('原文：', delta.source, '\n日文：', delta.confirmed, flush=True)

    result = {'input': 'testdata/zh_test_16k.pcm', 'target': 'ja', 'errors': []}
    try:
        started = time.perf_counter()
        await session.start(on_text, chunks.append)
        result['connect_seconds'] = round(time.perf_counter() - started, 3)
        print('訂閱語音已連線，開始傳送中文測試錄音。', flush=True)
        pcm = (ROOT / 'testdata' / 'zh_test_16k.pcm').read_bytes()
        input_started = time.perf_counter()
        for offset in range(0, len(pcm) + 32000 * 20, 3200):
            if not session.is_alive:
                raise RuntimeError(session.fail_reason)
            block = pcm[offset:offset + 3200] if offset < len(pcm) else bytes(3200)
            samples = np.frombuffer(block, dtype='<i2')
            if max(abs(int(samples.min())), abs(int(samples.max()))) >= 250:
                session.note_voice()
            await session.send_audio(block)
            session.tick()
            await asyncio.sleep(max(0, input_started + (offset + 3200) / 32000 - time.perf_counter()))
        session.tick()
    except Exception as exc:
        result['errors'].append(str(exc))
        result['diagnostics'] = session.diagnostics()
    finally:
        await session.close()
    audio = b''.join(chunks)
    with wave.open(str(out / 'translation.wav'), 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000); wav.writeframes(audio)
    samples = np.frombuffer(audio, dtype='<i2')
    peak = max(abs(int(samples.min())), abs(int(samples.max()))) if samples.size else 0
    result.update({'text_events': texts, 'audio_bytes': len(audio), 'audio_peak': peak,
                   'first_text_seconds': next((entry['at'] for entry in texts if entry['text']), None),
                   'has_final': any(entry['final'] for entry in texts)})
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'text_events'}, ensure_ascii=False, indent=2))
    return 0 if not result['errors'] and texts and peak > 250 and result['has_final'] else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
