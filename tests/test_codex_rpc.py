"""JSON-RPC 的有界等待、工具拒絕及隱私邊界。"""
import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlt.session.codex_rpc import CodexRPC


class RPCTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_names_the_failed_method_and_clears_pending(self):
        rpc = CodexRPC(lambda _: None)
        rpc._send = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, 'account/read'):
            await rpc.request('account/read', timeout=0.01)
        self.assertFalse(rpc.pending)

    async def test_server_tools_are_rejected_and_errors_do_not_expose_tokens(self):
        rpc = CodexRPC(lambda _: None)
        reader = asyncio.StreamReader()
        rpc.process = SimpleNamespace(stdout=reader)
        rpc._send = AsyncMock()
        future = asyncio.get_running_loop().create_future()
        rpc.pending[2] = future
        messages = [
            {'id': 1, 'method': 'item/tool/call', 'params': {'command': 'dangerous'}},
            {'id': 2, 'error': {'code': -32000, 'message': 'private token should not appear'}},
        ]
        for message in messages:
            reader.feed_data((json.dumps(message) + '\n').encode())
        reader.feed_eof()
        await rpc._read()
        sent = rpc._send.call_args.args[0]
        self.assertEqual(sent['error']['code'], -32601)
        with self.assertRaisesRegex(RuntimeError, '-32000') as caught:
            await future
        self.assertNotIn('private token', str(caught.exception))

    async def test_eof_rejects_inflight_requests(self):
        events = []
        rpc = CodexRPC(events.append)
        reader = asyncio.StreamReader(); reader.feed_eof()
        rpc.process = SimpleNamespace(stdout=reader)
        future = asyncio.get_running_loop().create_future()
        rpc.pending[1] = future
        await rpc._read()
        with self.assertRaises(ConnectionError):
            await future
        self.assertEqual(events[-1]['method'], 'transport/closed')


if __name__ == '__main__':
    unittest.main()
