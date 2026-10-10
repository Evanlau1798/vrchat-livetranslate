"""固定的訂閱目的地；無 redirect、proxy、重送或一般 API credit 路線。"""
from __future__ import annotations

import http.client
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
_HEADERS = {'authorization', 'chatgpt-account-id', 'content-type', 'accept', 'openai-beta', 'originator'}


class _HTTPFailure(RuntimeError):
    pass


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
    host, path, limit, body_limit = _OPERATIONS[operation]
    if len(body) > body_limit:
        raise RuntimeError('request-invalid')
    connection = http.client.HTTPSConnection(host, timeout=35)
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
        raw = bytearray()
        while True:
            transport.settimeout(_remaining(deadline))
            block = response.read1(min(8192, limit + 1 - len(raw)))
            if not block:
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
