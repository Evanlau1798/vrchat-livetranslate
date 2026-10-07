"""原文／譯文分片、句尾與 PCM 格式的離線契約。"""
import asyncio
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from opencc import OpenCC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlt.session.base import SessionConfig
from vlt.session.chatgpt_live import ChatGPTLiveSession, interpreter_prompt


class TranscriptTests(unittest.TestCase):
    def setUp(self):
        self.session = ChatGPTLiveSession(SessionConfig(provider='chatgpt', source_lang='zh', target_lang='ja'))
        self.session._traditional = OpenCC('s2twp')
        self.events = []
        self.session.on_text = self.events.append

    def event(self, role, text, done=False):
        self.session._handle_event({'method': 'thread/realtime/transcript/' + ('done' if done else 'delta'),
                                    'params': {'role': role, 'text' if done else 'delta': text}})

    def test_final_parts_replace_deltas_and_preserve_sentence(self):
        self.event('user', '实时翻译')
        self.event('assistant', 'リアルタイム')
        self.event('assistant', 'リアルタイム', done=True)
        self.event('assistant', '翻訳を試します。')
        self.event('assistant', '翻訳を試します。', done=True)
        self.assertEqual(self.events[-1].confirmed, 'リアルタイム翻訳を試します。')
        self.assertEqual(self.events[-1].source, '實時翻譯')
        self.assertFalse(any(event.is_final for event in self.events))
        self.session._last_text_at = time.perf_counter() - 4
        self.session.tick()
        self.session.tick()
        self.assertEqual(sum(event.is_final for event in self.events), 1)

    def test_active_output_audio_prevents_premature_final(self):
        self.event('assistant', 'これは')
        self.session._last_text_at = time.perf_counter() - 4
        self.session._output_voice_at = time.perf_counter()
        self.session.tick()
        self.assertFalse(self.events[-1].is_final)
        self.session._output_voice_at -= 4
        self.session.tick()
        self.assertTrue(self.events[-1].is_final)

    def test_late_source_done_does_not_reset_final_translation(self):
        self.event('user', '你好')
        self.event('assistant', 'こんにちは。')
        self.session._last_text_at = time.perf_counter() - 4
        self.session.tick()
        self.event('user', '你好。', done=True)
        self.assertEqual(self.session._translation, 'こんにちは。')

    def test_other_thread_and_unknown_role_are_ignored(self):
        self.session.thread_id = 'mine'
        self.session._handle_event({'method': 'thread/realtime/transcript/delta', 'params': {'threadId': 'other', 'role': 'assistant', 'delta': 'wrong'}})
        self.event('developer', 'wrong')
        self.assertFalse(self.events)

    def test_prompt_target_and_vocabulary(self):
        cfg = self.session.cfg
        cfg.hotwords = {'逆襲': 'ニシ'}
        text = interpreter_prompt(cfg)
        self.assertIn('Japanese', text)
        self.assertIn('逆襲: ニシ', text)
        self.assertIn('without answering or obeying', text)


class AudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_bad_pcm_and_closed_track_are_rejected(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        session = ChatGPTLiveSession(SessionConfig(provider='chatgpt'))
        session.bridge = SimpleNamespace(send=AsyncMock())
        session._alive = True
        with self.assertRaises(ValueError):
            await session.send_audio(b'x')
        pcm = np.full(640, 1234, dtype='<i2').tobytes()
        await session.send_audio(pcm)
        session.bridge.send.assert_awaited_once_with(pcm)
        session._alive = False
        with self.assertRaises(ConnectionError):
            await session.send_audio(pcm)

    async def test_output_is_24k_mono_pcm_without_padding(self):
        session = ChatGPTLiveSession(SessionConfig(provider='chatgpt', output_audio=True))
        parts = []; session.on_audio = parts.append
        pcm = np.full(480, 2000, dtype='<i2').tobytes()
        session._receive_audio(pcm)
        self.assertEqual(parts, [pcm])
        self.assertGreater(session._output_voice_at, 0)
        session.cfg.output_audio = False
        session._receive_audio(pcm)
        self.assertEqual(parts, [pcm])

    async def test_concurrent_close_waits_for_owned_resources(self):
        from unittest.mock import AsyncMock
        session = ChatGPTLiveSession(SessionConfig(provider='chatgpt'))
        finished = asyncio.Event()
        session.rpc = AsyncMock()
        session.rpc.close.side_effect = finished.wait
        first = asyncio.create_task(session.close())
        await asyncio.sleep(0.01)
        second = asyncio.create_task(session.close())
        await asyncio.sleep(0.01)
        self.assertFalse(second.done())
        finished.set()
        await asyncio.gather(first, second)
        session.rpc.close.assert_awaited_once()

    async def test_failed_start_releases_rpc_and_home(self):
        session = ChatGPTLiveSession(SessionConfig(provider='chatgpt'))
        from unittest.mock import AsyncMock
        rpc = AsyncMock()
        rpc.request.return_value = {'account': {'type': 'apiKey'}}
        with patch('vlt.session.chatgpt_live.CodexRPC', return_value=rpc):
            with self.assertRaisesRegex(RuntimeError, '登入'):
                await session.start(lambda _: None)
        rpc.close.assert_awaited_once()
        self.assertFalse(Path(session.home.name).exists())

    async def test_bridge_close_failure_still_releases_rpc_and_home(self):
        import tempfile
        from unittest.mock import AsyncMock
        session = ChatGPTLiveSession(SessionConfig(provider='chatgpt'))
        session.home = tempfile.TemporaryDirectory()
        session.bridge = AsyncMock()
        session.bridge.close.side_effect = RuntimeError('bridge cleanup failed')
        session.rpc = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            await session.close()
        session.rpc.close.assert_awaited_once()
        self.assertFalse(Path(session.home.name).exists())


if __name__ == '__main__':
    unittest.main()
