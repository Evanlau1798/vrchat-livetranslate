"""固定的訂閱目的地；無 redirect、proxy、重送或一般 API credit 路線。"""
from __future__ import annotations

import http.client
import re
import socket
import threading
import time
from concurrent.futures import Future, TimeoutError

# 協定來源：openai/codex e72da2b；Wangnov/codex-asr 479f6a7。
_OPERATIONS = {
    'refresh': ('auth.openai.com', '/oauth/token', 65536, 65536),
    'transcribe': ('chatgpt.com', '/backend-api/transcribe', 65536, 4_500_000),
    'responses': ('chatgpt.com', '/backend-api/codex/responses?client_version=0.162.0', 256000, 65536),
    'live': ('chatgpt.com', '/backend-api/codex/realtime/calls?intent=quicksilver&architecture=avas', 256000, 400000),
}
_HEADERS = {'authorization', 'chatgpt-account-id', 'content-type', 'accept', 'openai-beta', 'originator', 'user-agent'}


class _HTTPFailure(RuntimeError):
    pass


_TOKEN = rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+"
_QUOTED = rb'"(?:[\t \x21\x23-\x5b\x5d-\x7e\x80-\xff]|\\[\t \x21-\x7e\x80-\xff])*"'
_CHUNK = re.compile(rb'([0-9A-Fa-f]+)(?:[ \t]*;[ \t]*' + _TOKEN +
                    rb'(?:[ \t]*=[ \t]*(?:' + _TOKEN + rb'|' + _QUOTED + rb'))?)*\r\n')
_TRAILER = re.compile(rb'(' + _TOKEN + rb'):[\t \x21-\x7e\x80-\xff]*\r\n')


class _Response(http.client.HTTPResponse):
    # Python 3.11 的 chunked decoder 不核對分隔 CRLF，並容忍缺少 trailer 終止空行。
    def _get_chunk_left(self):
        if self.chunk_left == 0:
            if self._safe_read(2) != b'\r\n':
                raise _HTTPFailure('http-uncertain')
            self.chunk_left = None
        return super()._get_chunk_left()

    def _read_next_chunk_size(self):
        line = self.fp.readline(8193)
        matched = _CHUNK.fullmatch(line) if len(line) <= 8192 else None
        if matched is None:
            raise _HTTPFailure('http-uncertain')
        return int(matched[1], 16)

    def _read_and_discard_trailer(self):
        total = 0
        for _ in range(101):
            line = self.fp.readline(8193)
            total += len(line)
            if len(line) > 8192 or total > 65536:
                raise _HTTPFailure('http-size')
            if line == b'\r\n':
                return
            matched = _TRAILER.fullmatch(line)
            if matched is None or matched[1].lower() in (b'content-length', b'transfer-encoding', b'trailer', b'host'):
                raise _HTTPFailure('http-uncertain')
        raise _HTTPFailure('http-size')


_DNS = {}
_DNS_LOCK = threading.Lock()


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _HTTPFailure('http-timeout')
    return remaining


def _resolve(host, deadline):
    # getaddrinfo 不可取消；每個固定 host 最多留一個只做 DNS 的 worker，不容它送出請求。
    with _DNS_LOCK:
        job = _DNS.get(host)
        if job is None or job.done():
            job = Future()
            _DNS[host] = job

            def lookup():
                try:
                    job.set_result(socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM))
                except Exception:
                    job.set_exception(_HTTPFailure('http-uncertain'))

            threading.Thread(target=lookup, daemon=True, name='vlt-subscription-dns').start()
    try:
        return job.result(timeout=_remaining(deadline))
    except TimeoutError:
        raise _HTTPFailure('http-timeout') from None


def _connect(host, deadline):
    for family, kind, protocol, _, address in _resolve(host, deadline):
        sock = socket.socket(family, kind, protocol)
        try:
            sock.settimeout(_remaining(deadline))
            sock.connect(address)
            sock.settimeout(_remaining(deadline))
            return sock
        except Exception:
            sock.close()
            _remaining(deadline)
    raise _HTTPFailure('http-uncertain')


def request(operation, body, headers):
    if (not isinstance(operation, str) or operation not in _OPERATIONS or
            not isinstance(body, bytes) or not isinstance(headers, dict)):
        raise RuntimeError('request-invalid')
    if any(not isinstance(k, str) or k.lower() not in _HEADERS or
           not isinstance(v, str) or '\r' in v or '\n' in v for k, v in headers.items()):
        raise RuntimeError('request-invalid')
    if any(k.lower() == 'user-agent' and
           (operation != 'transcribe' or v != 'Codex Desktop/26.429.30905') for k, v in headers.items()):
        raise RuntimeError('request-invalid')
    host, path, limit, body_limit = _OPERATIONS[operation]
    if len(body) > body_limit:
        raise RuntimeError('request-invalid')
    connection = http.client.HTTPSConnection(host, timeout=35)
    connection.response_class = _Response
    deadline = time.monotonic() + 35
    connection._create_connection = lambda *args, **kwargs: _connect(host, deadline)
    transport_ref = [None]

    def abort():
        transport = transport_ref[0] or connection.sock
        if transport:
            try:
                transport.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
        try:
            connection.close()
        except Exception:
            pass

    timer = threading.Timer(35, abort)
    timer.daemon = True
    timer.start()
    response = None
    failed = False
    try:
        connection.timeout = _remaining(deadline)
        connection.connect()
        transport = connection.sock
        transport_ref[0] = transport
        transport.settimeout(_remaining(deadline))
        connection.request('POST', path, body=body, headers=headers)
        transport.settimeout(_remaining(deadline))
        response = connection.getresponse()
        if response.status != 200:
            raise _HTTPFailure('http-rejected')
        if (response.headers.defects or response.headers.get_unixfrom() or response.headers.get_payload() or any(
                _TRAILER.fullmatch((k + ':' + v + '\r\n').encode('latin-1')) is None
                for k, v in response.headers.items())):
            raise _HTTPFailure('http-uncertain')
        lengths = response.headers.get_all('Content-Length', [])
        encodings = response.headers.get_all('Transfer-Encoding', [])
        if (len(lengths) > 1 or len(encodings) > 1 or lengths and encodings or
                encodings and (encodings[0].strip().lower() != 'chunked' or response.chunked is not True)):
            raise _HTTPFailure('http-uncertain')
        if lengths:
            value = lengths[0].strip()
            if (not value.isascii() or not value.isdecimal() or
                    type(response.length) is not int or response.length < 0):
                raise _HTTPFailure('http-uncertain')
        elif not encodings and not response.will_close:
            raise _HTTPFailure('http-uncertain')
        raw = bytearray()
        while True:
            transport.settimeout(_remaining(deadline))
            block = response.read1(min(8192, limit + 1 - len(raw)))
            _remaining(deadline)
            if not block:
                if response.length not in (None, 0):
                    raise _HTTPFailure('http-uncertain')
                return bytes(raw)
            raw.extend(block)
            if len(raw) > limit:
                raise _HTTPFailure('http-size')
    except _HTTPFailure:
        failed = True
        raise
    except Exception:
        failed = True
        raise RuntimeError('http-uncertain') from None
    finally:
        timer.cancel()
        cleanup_failed = False
        for resource in (response, connection):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    cleanup_failed = True
        if cleanup_failed and not failed:
            raise RuntimeError('http-uncertain') from None
