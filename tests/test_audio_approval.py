"""Offline audio approval at actual Windows/Linux consumers; no devices or services."""
import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlt.session.chatgpt_output import scope
from vlt.output.virtualmic import VirtualMic, PwCatVirtualMic, resample_24k_mono_to_48k_stereo
from vlt.output.micproxy import MicProxy, MODE_TRANSLATED, _TranslatedBuffer
from vlt.output.micproxy_linux import LinuxMicProxy

BLOCK = 3840
PCM = b'\x11\x22' * (BLOCK // 2)


class Approval:
    def __init__(self, direction='mine'):
        self.valid = True
        self.direction = direction

    def __call__(self, direction=None):
        return self.valid and (direction is None or direction == self.direction)


class AudioApprovalTests(unittest.TestCase):
    def setUp(self):
        no_device = patch('sounddevice.RawOutputStream', side_effect=AssertionError('device opened'))
        no_device.start()
        self.addCleanup(no_device.stop)

    def consumers(self):
        vm = VirtualMic(device_index=0, device_name='unused', buffer_ms=1)
        def drain_vm():
            out = bytearray(BLOCK)
            vm._audio_callback(out, BLOCK // 4, None, None)
            return bytes(out)
        yield 'VirtualMic', vm, drain_vm
        for cls in (MicProxy, LinuxMicProxy):
            proxy = cls(audio_cfg={})
            proxy._mode = MODE_TRANSLATED
            proxy._translated = _TranslatedBuffer(device_index=0, device_name='unused', buffer_ms=1)
            def drain(proxy=proxy):
                out = bytearray(BLOCK)
                proxy._out_callback(out, BLOCK // 4, None, None)
                return bytes(out)
            yield cls.__name__, proxy._translated, drain

    def test_each_consumer_discards_revoked_remaining_audio(self):
        for name, vm, drain in self.consumers():
            with self.subTest(consumer=name):
                approval = Approval()
                with scope(approval):
                    vm.push(PCM * 2)
                    vm.end_sentence()
                self.assertEqual(drain(), PCM)
                approval.valid = False
                self.assertEqual(drain(), bytes(BLOCK))
                self.assertEqual(vm._buf_bytes, 0)
                self.assertFalse(vm._head_started)

    def test_invalid_head_does_not_discard_fresh_approved_tail(self):
        for name, vm, drain in self.consumers():
            with self.subTest(consumer=name):
                stale, fresh = Approval(), Approval()
                with scope(stale):
                    vm.push(PCM)
                    vm.end_sentence()
                with scope(fresh):
                    vm.push(b'\x33\x44' * (BLOCK // 2))
                    vm.end_sentence()
                stale.valid = False
                self.assertEqual(drain(), b'\x33\x44' * (BLOCK // 2))
                self.assertEqual(drain(), bytes(BLOCK))

    def test_other_direction_never_enters_virtual_mic(self):
        for name, vm, drain in self.consumers():
            with self.subTest(consumer=name):
                with scope(Approval('theirs')):
                    vm.push(PCM)
                self.assertEqual(vm._buf_bytes, 0)
                self.assertEqual(drain(), bytes(BLOCK))

    def test_pipe_writer_discards_before_actual_fake_pipe_write(self):
        vm = PwCatVirtualMic(target='offline-test', buffer_ms=1)
        approval = Approval()
        with scope(approval):
            vm.push(PCM)
            vm.end_sentence()
        approval.valid = False
        writes = []
        def write(data):
            writes.append(data)
            vm._stop.set()
            return len(data)
        vm._proc = SimpleNamespace(stdin=SimpleNamespace(write=write, flush=Mock()))
        vm._writer_loop()
        self.assertEqual(writes, [bytes(vm._chunk)])
        self.assertEqual(vm._buf_bytes, 0)

    def test_close_releases_started_sentence_state(self):
        for name, vm, drain in self.consumers():
            with self.subTest(consumer=name):
                vm.push(PCM * 2)
                self.assertEqual(drain(), PCM)
                self.assertTrue(vm._head_started)
                vm.close()
                self.assertFalse(vm._head_started)
                self.assertEqual(vm._buf_bytes, 0)

    def test_engine_rejects_audio_approval_for_another_direction(self):
        from vlt.engine import Engine, EngineEvents
        from test_repeat_guard import _cfg

        cfg = _cfg()
        cfg.directions['theirs'] = cfg.directions['mine']
        engine = Engine(cfg=cfg, direction='theirs', source='loopback', sinks=set(), events=EngineEvents())
        engine._virtualmic = Mock()
        with scope(Approval('mine')):
            engine._on_audio(PCM)
        engine._virtualmic.push.assert_not_called()
        self.assertEqual(engine._audio_chunks, 0)

    def test_direct_pipe_close_releases_started_sentence_state(self):
        vm = PwCatVirtualMic(target='offline-test', buffer_ms=1)
        vm.push(PCM * 2)
        vm.end_sentence()
        out = bytearray(BLOCK)
        vm._audio_callback(out, BLOCK // 4, None, None)
        self.assertEqual(bytes(out), PCM)
        self.assertTrue(vm._head_started)
        self.assertEqual(vm._buf_bytes, BLOCK)
        vm.close()
        self.assertFalse(vm._head_started)
        self.assertEqual(vm._buf_bytes, 0)
        self.assertFalse(vm._primed)
        vm.close()
        self.assertFalse(vm._head_started)


class CoordinatorAudioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from vlt.session.base import SessionConfig
        from vlt.session.chatgpt_pipeline import Coordinator

        verdict = dict(approved=True, language='ja', meaning=True, numbers=True,
                       negation=True, complete=True, extra_content=False)
        self.pcm = b'\x11\x22' * 1000
        services = SimpleNamespace(source_asr=AsyncMock(return_value='source'),
            translate=AsyncMock(return_value=dict(translation='translation', language='ja')),
            validate_text=AsyncMock(return_value=verdict),
            synthesize=AsyncMock(return_value=('candidate', self.pcm)),
            audio_asr=AsyncMock(return_value='translation'),
            validate_audio=AsyncMock(return_value=verdict))
        self.pipeline = Coordinator({'mine': SessionConfig(provider='chatgpt', source_lang='zh',
            target_lang='ja', output_audio=True)}, services, Mock(), lambda: 1)
        self.addAsyncCleanup(self.pipeline.close)
        self.vm = VirtualMic(device_index=0, device_name='unused', buffer_ms=1)
        self.assertTrue(self.pipeline.submit('mine', bytes(320)))
        await asyncio.gather(*tuple(self.pipeline._owned))

    def audio(self, pcm):
        self.vm.push(resample_24k_mono_to_48k_stereo(pcm))

    def drain(self):
        out = bytearray(BLOCK)
        self.vm._audio_callback(out, BLOCK // 4, None, None)
        return bytes(out)

    async def test_actual_coordinator_closure_revokes_buffered_pcm(self):
        self.assertTrue(self.pipeline.consume('mine', Mock(), self.audio))
        self.assertEqual(self.vm._buf_bytes, len(self.pcm) * 4)
        self.pipeline.invalidate()
        self.assertEqual(self.drain(), bytes(BLOCK))
        self.assertEqual(self.vm._buf_bytes, 0)

    async def test_approved_pcm_is_consumed_once_without_losing_short_tail(self):
        text = Mock()
        self.assertTrue(self.pipeline.consume('mine', text, self.audio))
        self.vm.end_sentence()
        rendered = resample_24k_mono_to_48k_stereo(self.pcm)
        blocks = (len(rendered) + BLOCK - 1) // BLOCK
        actual = b''.join(self.drain() for _ in range(blocks))
        self.assertEqual(actual, rendered + bytes(blocks * BLOCK - len(rendered)))
        self.assertFalse(self.pipeline.consume('mine', text, self.audio))
        text.assert_called_once()
        self.assertEqual(self.drain(), bytes(BLOCK))

    async def test_text_callback_invalidation_prevents_audio_enqueue(self):
        self.pipeline.consume('mine', lambda _: self.pipeline.invalidate(), self.audio)
        self.assertEqual(self.vm._buf_bytes, 0)
        self.assertEqual(self.drain(), bytes(BLOCK))



if __name__ == "__main__":
    unittest.main()
