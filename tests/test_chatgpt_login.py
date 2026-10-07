"""登入途中關窗的離線清理契約，不執行真實登入。"""
import asyncio
import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlt import gui_chatgpt, gui as gui_mod
from vlt.session.codex_rpc import login_chatgpt


class GuiLoginTests(unittest.TestCase):
    def test_repeated_close_does_not_interrupt_login_cleanup_again(self):
        loop, task = Mock(), Mock()
        gui = SimpleNamespace(_chatgpt_login_cancel=threading.Event(), _chatgpt_login_task=(loop, task))
        gui_chatgpt.cancel_login(gui)
        gui_chatgpt.cancel_login(gui)
        loop.call_soon_threadsafe.assert_called_once_with(task.cancel)

    def test_window_close_cancels_owned_login(self):
        started, closed = threading.Event(), threading.Event()
        owner = []

        async def login():
            owner.append((asyncio.get_running_loop(), asyncio.current_task()))
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        gui = SimpleNamespace(_provider=lambda: 'chatgpt', _set_status=Mock(), _q=queue.Queue(),
                              _close_proxy=Mock(), _sync_engine_ctx=Mock(), _engine_ctx=None)
        with patch('vlt.session.codex_rpc.login_chatgpt', side_effect=login), \
             patch.object(gui_mod.gui_engine, 'on_close'):
            try:
                gui_chatgpt.open_signup(gui)
                self.assertTrue(started.wait(2))
                gui_mod.TranslationGUI._on_close(gui)
                self.assertTrue(closed.wait(2), 'window close left login running')
            finally:
                if owner and not closed.is_set():
                    loop, task = owner[0]
                    loop.call_soon_threadsafe(task.cancel)
                closed.wait(2)
                worker = getattr(gui, '_chatgpt_login_thread', None)
                if worker:
                    worker.join(2)
        if worker:
            self.assertFalse(worker.is_alive())
            self.assertFalse(worker.daemon)


class LoginProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_unresponsive_login_is_killed_during_cancel(self):
        started, killed = asyncio.Event(), asyncio.Event()
        process = SimpleNamespace(returncode=None, terminate=Mock(), kill=Mock(side_effect=killed.set))

        async def wait():
            started.set()
            await killed.wait()
            process.returncode = 0
            return 0

        process.wait = wait
        with patch('vlt.session.codex_rpc.codex_command', return_value=['fake-codex']), \
             patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=process)):
            task = asyncio.create_task(login_chatgpt())
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 4)
        process.kill.assert_called_once()

    async def test_cancel_terminates_and_reaps_login_process(self):
        started, stopped = asyncio.Event(), asyncio.Event()
        process = SimpleNamespace(returncode=None, terminate=Mock(side_effect=stopped.set))

        async def wait():
            started.set()
            await stopped.wait()
            process.returncode = 0
            return 0

        process.wait = wait
        with patch('vlt.session.codex_rpc.codex_command', return_value=['fake-codex']), \
             patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=process)):
            task = asyncio.create_task(login_chatgpt())
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        process.terminate.assert_called_once()
        self.assertEqual(process.returncode, 0)


if __name__ == '__main__':
    unittest.main()
