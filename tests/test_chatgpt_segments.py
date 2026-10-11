"""合成 PCM 的分段邊界；不使用音檔、ASR、音訊裝置或網路。"""
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlt.session.chatgpt_segments import Segmenter


def pcm(value, samples):
    return struct.pack('<h', value) * samples


def feed(segmenter, data):
    result = []
    for start in range(0, len(data), 3200):
        result.extend(segmenter.feed(data[start:start + 3200]))
    return result


class SegmentsTests(unittest.TestCase):
    def test_silence_never_becomes_an_utterance(self):
        self.assertEqual(feed(Segmenter(), pcm(219, 40000)), [])

    def test_threshold_exact_silence_and_preroll_are_sample_bound(self):
        s = Segmenter()
        self.assertEqual(feed(s, pcm(0, 32000)), [])
        self.assertEqual(feed(s, pcm(220, 160)), [])
        self.assertEqual(feed(s, pcm(0, 7999)), [])
        results = s.feed(pcm(0, 1))
        self.assertEqual(len(results), 1, '500 ms silence must close the voiced section')
        self.assertEqual(results[0].pcm, pcm(0, 16000) + pcm(220, 160) + pcm(0, 8000))

    def test_closed_samples_are_never_reused_as_preroll(self):
        results = feed(Segmenter(), pcm(400, 160) + pcm(0, 8000) +
                       pcm(600, 160) + pcm(0, 8000))
        self.assertEqual(len(results), 2, 'two actual utterances must remain separate')
        self.assertEqual(results[1].pcm, pcm(600, 160) + pcm(0, 8000))

    def test_limit_rejects_whole_section_until_next_silence(self):
        s = Segmenter()
        results = feed(s, pcm(500, 480001))
        self.assertEqual([(r.pcm, r.reason) for r in results], [(b'', 'segment-size')])
        self.assertEqual(feed(s, pcm(500, 16000) + pcm(0, 7999)), [])
        s.feed(pcm(0, 1))
        recovered = feed(s, pcm(700, 100) + pcm(0, 8000))
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].reason, '')
        self.assertNotIn(pcm(500, 1), recovered[0].pcm)

    def test_reset_discards_partial_section(self):
        s = Segmenter()
        feed(s, pcm(500, 100))
        s.reset()
        self.assertEqual(feed(s, pcm(0, 8000)), [])

    def test_size_limit_at_silence_boundary_does_not_discard_the_next_utterance(self):
        s = Segmenter()
        rejected = feed(s, pcm(500, 472001) + pcm(0, 8000))
        self.assertEqual([(r.pcm, r.reason) for r in rejected], [(b'', 'segment-size')])
        normal = feed(s, pcm(700, 100) + pcm(0, 8000))
        self.assertEqual(len(normal), 1, 'completed rejection boundary must allow the next full utterance')
        self.assertEqual(normal[0].pcm, pcm(700, 100) + pcm(0, 8000))

    def test_bad_pcm_is_rejected_without_retaining_partial_speech(self):
        s = Segmenter()
        feed(s, pcm(500, 100))
        with self.assertRaisesRegex(RuntimeError, '^pcm-invalid$'):
            s.feed(b'x')
        self.assertEqual(feed(s, pcm(0, 8000)), [])


if __name__ == '__main__':
    unittest.main()
