"""ChatGPT 專用 16 kHz mono s16le 分段；不建立音訊或服務連線。"""
from dataclasses import dataclass
from array import array
from collections import deque
import sys


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
        results = []
        for value in samples:
            voiced = abs(value) >= 220
            if self._discarding:
                self._quiet = 0 if voiced else self._quiet + 1
                if self._quiet >= 8000:
                    self.reset()
                continue
            if self._section is None:
                if not voiced:
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
                    self.reset()
            elif self._quiet >= 8000:
                if sys.byteorder != 'little':
                    self._section.byteswap()
                results.append(PCMSection(self._section.tobytes()))
                self.reset()
        return results

    def reset(self):
        self._preroll = deque(maxlen=16000)
        self._section = None
        self._quiet = 0
        self._discarding = False
