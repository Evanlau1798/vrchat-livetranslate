"""固定目的地與錯誤邊界的離線測試，不建立連線。"""
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlt.session.chatgpt_http import request
from vlt.session import chatgpt_http as http


class HTTPTests(unittest.TestCase):
    def test_timer_callback_does_not_leak_cleanup_exceptions(self):
        connection = Mock()
        connection.getresponse.return_value = Mock(status=302)
        connection.close.side_effect = OSError('private-timer-close')
        with patch('http.client.HTTPSConnection', return_value=connection), \
             patch('threading.Timer') as timer:
            with self.assertRaisesRegex(RuntimeError, '^http-rejected$'):
                request('refresh', b'data', {})
            callback = timer.call_args.args[1]
            callback()
        self.assertEqual(connection.close.call_count, 2)

    def test_cleanup_errors_are_sanitized_and_preserve_primary_failure(self):
        for status, reason in ((200, 'http-uncertain'), (302, 'http-rejected')):
            for failing in ('response', 'connection'):
                with self.subTest(status=status, failing=failing):
                    response = Mock(status=status, read1=Mock(return_value=b''))
                    connection = Mock()
                    connection.getresponse.return_value = response
                    target = response if failing == 'response' else connection
                    target.close.side_effect = OSError('private-close')
                    with patch('http.client.HTTPSConnection', return_value=connection):
                        with self.assertRaisesRegex(RuntimeError, '^' + reason + '$'):
                            request('refresh', b'data', {})
                    response.close.assert_called_once()
                    connection.close.assert_called_once()

    def test_detached_response_is_closed_on_reject_oversize_and_read_error(self):
        for status, read, reason in ((302, [b''], 'http-rejected'),
                                     (200, [b'x' * 65537], 'http-size'),
                                     (200, OSError('private-details'), 'http-uncertain')):
            with self.subTest(reason=reason):
                connection, response = Mock(), Mock(status=status)
                response.read1.side_effect = read

                def detach():
                    connection.sock = None
                    return response

                connection.getresponse.side_effect = detach
                with patch('http.client.HTTPSConnection', return_value=connection):
                    with self.assertRaisesRegex(RuntimeError, '^' + reason + '$'):
                        request('refresh', b'data', {})
                response.close.assert_called_once()
                connection.close.assert_called_once()

    def test_connection_close_body_keeps_original_socket_budget(self):
        connection, transport, response = Mock(), Mock(), Mock(status=200)
        response.read1.side_effect = [b'ok', b'']
        connection.sock = transport

        def detach():
            connection.sock = None
            return response

        connection.getresponse.side_effect = detach
        with patch('http.client.HTTPSConnection', return_value=connection):
            self.assertEqual(request('refresh', b'data', {}), b'ok')
        self.assertGreaterEqual(transport.settimeout.call_count, 4)

    def test_pending_dns_is_bounded_and_never_spawns_more_resolvers(self):
        entered, release = threading.Event(), threading.Event()

        def dns(*args, **kwargs):
            entered.set()
            release.wait(2)
            return []

        try:
            with patch('socket.getaddrinfo', side_effect=dns) as lookup:
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError, '^http-timeout$'):
                        http._resolve('offline-dns-test', time.monotonic() + .02)
                self.assertTrue(entered.is_set())
                self.assertEqual(lookup.call_count, 1)
        finally:
            release.set()
            http._DNS['offline-dns-test'].result(timeout=2)
            http._DNS.pop('offline-dns-test')
    def test_slow_send_consumes_header_budget(self):
        connection = Mock()
        connection.getresponse.return_value = Mock(status=200, read1=Mock(return_value=b''))
        with patch('http.client.HTTPSConnection', return_value=connection), \
             patch('time.monotonic', side_effect=[0, 0, 10, 36, 36]):
            with self.assertRaisesRegex(RuntimeError, '^http-timeout$'):
                request('refresh', b'data', {})
        self.assertTrue(any(c.args[0] <= 25 for c in connection.sock.settimeout.call_args_list),
                        'headers must use time remaining after request send')

    def test_fixed_destination_no_environment_proxy_or_redirect(self):
        connection = Mock()
        connection.getresponse.return_value = Mock(status=302)
        with patch('http.client.HTTPSConnection', return_value=connection) as connect:
            with self.assertRaisesRegex(RuntimeError, '^http-rejected$'):
                request('transcribe', b'data', {'Authorization': 'Bearer fake'})
        self.assertEqual(connect.call_args.args, ('chatgpt.com',))
        self.assertEqual(connection.request.call_count, 1)
        connection.close.assert_called_once()

    def test_host_override_is_not_a_supported_header(self):
        with patch('http.client.HTTPSConnection') as connect:
            with self.assertRaisesRegex(RuntimeError, '^request-invalid$'):
                request('transcribe', b'data', {'Host': 'other.example'})
        connect.assert_not_called()

    def test_transport_exception_never_exposes_headers_or_body(self):
        connection = Mock()
        connection.request.side_effect = RuntimeError('private-secret')
        with patch('http.client.HTTPSConnection', return_value=connection):
            with self.assertRaisesRegex(RuntimeError, '^http-uncertain$'):
                request('transcribe', b'data', {})
        self.assertEqual(connection.request.call_count, 1)
        connection.close.assert_called_once()

    def test_response_is_bounded_and_connection_closed(self):
        response = Mock(status=200)
        response.read1.return_value = b'x' * 8192
        connection = Mock()
        connection.getresponse.return_value = response
        with patch('http.client.HTTPSConnection', return_value=connection):
            with self.assertRaisesRegex(RuntimeError, '^http-size$'):
                request('transcribe', b'data', {})
        connection.close.assert_called_once()

    def test_valid_body_is_read_once_without_retry(self):
        response = Mock(status=200)
        response.read1.side_effect = [b'{"text":"test"}', b'']
        connection = Mock()
        connection.getresponse.return_value = response
        with patch('http.client.HTTPSConnection', return_value=connection):
            self.assertEqual(request('transcribe', b'data', {}), b'{"text":"test"}')
        self.assertEqual(connection.request.call_count, 1)


if __name__ == '__main__':
    unittest.main()
