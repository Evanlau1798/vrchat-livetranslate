"""Codex app-server JSON-RPC；沿用 CLI 登入，不讀取或複製 token。"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
from pathlib import Path


def codex_command() -> list[str]:
    executable = shutil.which('codex.exe') or shutil.which('codex')
    if not executable:
        raise RuntimeError('找不到 Codex CLI，請先安裝 @openai/codex 並以 ChatGPT 登入。')
    path = Path(executable)
    if os.name == 'nt' and path.suffix.lower() != '.exe':
        arch = 'aarch64' if platform.machine().lower() in ('arm64', 'aarch64') else 'x86_64'
        package_arch = 'arm64' if arch == 'aarch64' else 'x64'
        root = path.parent / 'node_modules' / '@openai' / 'codex'
        vendor = Path('vendor') / f'{arch}-pc-windows-msvc' / 'bin' / 'codex.exe'
        candidates = [root / 'node_modules' / '@openai' / f'codex-win32-{package_arch}' / vendor, root / vendor]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise RuntimeError('找不到 Codex 原生執行檔，請重新安裝 @openai/codex。')
    return [str(path)]


class CodexRPC:
    def __init__(self, on_event):
        self.on_event = on_event
        self.process = None
        self.reader = None
        self.pending = {}
        self.next_id = 0
        self.lock = asyncio.Lock()
        self.closing = False

    async def start(self, cwd: str):
        self.process = await asyncio.create_subprocess_exec(
            *codex_command(), '--enable', 'realtime_conversation', 'app-server', '--stdio',
            cwd=cwd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=4 * 1024 * 1024,
            **({'creationflags': 0x08000000} if os.name == 'nt' else {}),
        )
        self.reader = asyncio.create_task(self._read())
        await self.request('initialize', {
            'clientInfo': {'name': 'vrchat_livetranslate', 'version': '1'},
            'capabilities': {'experimentalApi': True},
        })
        await self._send({'method': 'initialized'})

    async def _send(self, message):
        if self.process is None or self.process.returncode is not None:
            raise ConnectionError('Codex app-server 已停止。')
        async with self.lock:
            self.process.stdin.write((json.dumps(message, ensure_ascii=False) + '\n').encode('utf-8'))
            await self.process.stdin.drain()

    async def request(self, method, params=None, timeout=20):
        self.next_id += 1
        request_id = self.next_id
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self._send({'id': request_id, 'method': method, 'params': params or {}})
            try:
                return await asyncio.wait_for(future, timeout)
            except asyncio.TimeoutError as exc:
                raise RuntimeError(f'Codex {method} 請求逾時。') from exc
        finally:
            self.pending.pop(request_id, None)

    async def _read(self):
        try:
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                request_id = message.get('id')
                if 'method' in message:
                    if request_id is not None:
                        # 語音內容不能要求執行工具、批准指令或擴大權限。
                        await self._send({'id': request_id, 'error': {'code': -32601, 'message': 'Audio translation only; tools are unavailable.'}})
                    else:
                        self.on_event(message)
                elif request_id in self.pending:
                    future = self.pending[request_id]
                    if future.done():
                        continue
                    if 'error' in message:
                        # 不回傳未過濾的服務端內容，避免 token／帳戶資料進入 GUI log。
                        future.set_exception(RuntimeError(f'Codex 請求失敗（{message["error"].get("code", "unknown")}）。'))
                    else:
                        future.set_result(message.get('result', {}))
        except (ValueError, OSError, asyncio.LimitOverrunError):
            pass
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(ConnectionError('Codex app-server 連線中斷。'))
            if not self.closing:
                self.on_event({'method': 'transport/closed', 'params': {}})

    async def close(self):
        self.closing = True
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 3)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)


async def login_chatgpt():
    """使用官方 CLI 的瀏覽器登入，不在應用程式內儲存憑證。"""
    process = await asyncio.create_subprocess_exec(
        *codex_command(), 'login', stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        **({'creationflags': 0x08000000} if os.name == 'nt' else {}),
    )
    try:
        return await asyncio.wait_for(process.wait(), 180) == 0
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 3)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
