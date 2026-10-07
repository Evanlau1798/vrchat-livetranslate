"""本機橋接信任邊界與中斷協商清理，不使用外部帳戶或瀏覽器。"""
import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidStatus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlt.session.chatgpt_browser import BrowserBridge


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.audio, self.errors = [], []
        self.bridge = BrowserBridge(AsyncMock(return_value='v=0\r\nanswer'), self.audio.append, self.errors.append)
        self.bridge.server = await serve(self.bridge._handle, '127.0.0.1', 0,
                                         process_request=self.bridge._http, close_timeout=0.1)
        port = self.bridge.server.sockets[0].getsockname()[1]
        self.bridge.origin = f'http://127.0.0.1:{port}'
        self.url = f'ws://127.0.0.1:{port}/{self.bridge.token}/ws'

    async def asyncTearDown(self):
        await self.bridge.close()

    async def test_origin_and_secret_path_are_required(self):
        for url, origin, status in ((self.url, 'https://evil.example', 403),
                                    (self.url, None, 403),
                                    (self.url.replace(self.bridge.token, 'wrong'), self.bridge.origin, 404)):
            with self.assertRaises(InvalidStatus) as caught:
                async with connect(url, origin=origin):
                    self.fail('untrusted socket admitted')
            self.assertEqual(caught.exception.response.status_code, status)
        reader, writer = await asyncio.open_connection('127.0.0.1', self.bridge.server.sockets[0].getsockname()[1])
        writer.write(f'GET /{self.bridge.token} HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n'.encode())
        await writer.drain()
        self.assertIn(b'403', await reader.readline())
        writer.close()
        await writer.wait_closed()

    async def test_offer_ready_audio_and_single_owner(self):
        async with connect(self.url, origin=self.bridge.origin) as socket:
            await socket.send(json.dumps({'type': 'offer', 'sdp': 'v=0\r\noffer'}))
            self.assertEqual(json.loads(await socket.recv())['sdp'], 'v=0\r\nanswer')
            await socket.send('{"type":"ready"}')
            await asyncio.wait_for(self.bridge.ready.wait(), 1)
            with self.assertRaises(InvalidStatus):
                async with connect(self.url, origin=self.bridge.origin):
                    self.fail('second owner admitted')
            await self.bridge.send(b'\x01\x00')
            self.assertEqual(await socket.recv(), b'\x01\x00')
            await socket.send(b'\x02\x00')
            await socket.send('{"type":"offer","sdp":"v=0"}')
            await asyncio.wait_for(socket.wait_closed(), 1)
        self.assertEqual(self.audio, [b'\x02\x00'])
        self.bridge.on_offer.assert_awaited_once()
        self.assertTrue(self.errors)

    async def test_invalid_pcm_is_rejected(self):
        async with connect(self.url, origin=self.bridge.origin) as socket:
            await socket.send(b'x')
            await asyncio.wait_for(socket.wait_closed(), 1)
        self.assertFalse(self.audio)
        self.assertTrue(self.errors)

    async def test_close_cancels_pending_negotiation(self):
        started = asyncio.Event()

        async def pending(_):
            started.set()
            await asyncio.Event().wait()

        self.bridge.on_offer = pending
        async with connect(self.url, origin=self.bridge.origin) as socket:
            await socket.send('{"type":"offer","sdp":"v=0"}')
            await asyncio.wait_for(started.wait(), 1)
            await asyncio.wait_for(self.bridge.close(), 1)
        self.assertTrue(self.bridge.handler.done())
        self.assertFalse(self.errors)


if __name__ == '__main__':
    unittest.main()
