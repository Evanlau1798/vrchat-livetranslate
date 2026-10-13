"""ChatGPT 專用 16 kHz mono s16le 分段；不建立音訊或服務連線。"""
from dataclasses import dataclass
from array import array
from collections import deque
import sys

_FRAME_SAMPLES = 160  # 10 ms: isolated peaks must not erase real speech pauses.


@dataclass(frozen=True)
class PCMSection:
    pcm: bytes
    reason: str = ''


class Segmenter:
    def __init__(self):
        self.reset()

    def feed(self, pcm):
        if not isinstance(pcm, bytes) or len(pcm) % 2 or len(pcm) > 32000:
            self.reset()
            raise RuntimeError('pcm-invalid')
        samples = array('h')
        samples.frombytes(pcm)
        if sys.byteorder != 'little':
            samples.byteswap()
        samples = self._pending + samples
        complete = len(samples) // _FRAME_SAMPLES * _FRAME_SAMPLES
        self._pending = samples[complete:]
        results = []
        for start in range(0, complete, _FRAME_SAMPLES):
            frame = samples[start:start + _FRAME_SAMPLES]
            voiced = sum(value * value for value in frame) >= 220 ** 2 * _FRAME_SAMPLES
            if self._section is None and not self._discarding:
                self._onset_frames = self._onset_frames + 1 if voiced else 0
            for value in frame:
                if self._discarding:
                    self._quiet = 0 if voiced else self._quiet + 1
                    if self._quiet >= 8000:
                        self._reset_section()
                    continue
                if self._section is None:
                    if self._onset_frames < 3:
                        self._preroll.append(value)
                        continue
                    self._section = array('h', self._preroll)
                    self._preroll.clear()
                self._section.append(value)
                self._quiet = 0 if voiced else self._quiet + 1
                if len(self._section) > 480000:
                    self._section = None
                    self._discarding = True
                    results.append(PCMSection(b'', 'segment-size'))
                    if self._quiet >= 8000:
                        self._reset_section()
                elif self._quiet >= 8000:
                    if sys.byteorder != 'little':
                        self._section.byteswap()
                    results.append(PCMSection(self._section.tobytes()))
                    self._reset_section()
        if self._section is not None and len(self._section) + len(self._pending) > 480000:
            self._section = None
            self._discarding = True
            results.append(PCMSection(b'', 'segment-size'))
        return results

    def reset(self):
        self._pending = array('h')
        self._reset_section()

    def _reset_section(self):
        self._preroll = deque(maxlen=16000)
        self._section = None
        self._quiet = 0
        self._discarding = False
        self._onset_frames = 0
