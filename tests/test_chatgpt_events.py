"""手工協定 fixtures；不含服務記錄、錄音或帳戶資訊。"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlt.session.chatgpt_events import LiveEvents, response_text


def wire(value):
    return json.dumps(value).encode()


def sse(*events):
    return b''.join(b'data: ' + wire(event) + b'\n\n' for event in events)


def text_response(text='translation'):
    item = {'type': 'message', 'id': 'msg_1', 'role': 'assistant', 'status': 'completed',
            'content': [{'type': 'output_text', 'text': text}]}
    return sse({'type': 'response.created', 'response': {'id': 'resp_1'}},
               {'type': 'response.output_item.added', 'response_id': 'resp_1', 'item': {
                   **item, 'status': 'in_progress', 'content': []}},
               {'type': 'response.output_item.done', 'response_id': 'resp_1', 'item': item},
               {'type': 'response.completed', 'response': {'id': 'resp_1', 'status': 'completed', 'output': [item]}})


class EventsTests(unittest.TestCase):
    def test_live_preserves_turn_role_times_without_claiming_audio_complete(self):
        event = LiveEvents().consume(wire({'type': 'turn.done', 'turn': {
            'id': 'turn_1', 'role': 'assistant', 'transcript': 'test', 'start_ms': 20, 'end_ms': 900}}))
        self.assertIsNotNone(event, 'raw adapter must retain a complete transcript event')
        self.assertEqual((event.turn_id, event.role, event.text), ('turn_1', 'assistant', 'test'))
        self.assertEqual((event.start_ms, event.end_ms), (20, 900))
        self.assertEqual(event.kind, 'transcript-done')

    def test_live_unknown_tool_missing_id_and_role_fail_closed(self):
        for event in ({'type': 'delegation.created', 'item': {'target': 'client'}},
                      {'type': 'tool.call'}, {'type': 'new.event'},
                      {'type': 'output_audio.delta', 'audio': 'private-data'},
                      {'type': 'turn.done', 'turn': {'role': 'assistant', 'transcript': 'test'}},
                      {'type': 'turn.done', 'turn': {'id': 'turn_1', 'role': 'developer', 'transcript': 'test'}}):
            with self.assertRaises(RuntimeError):
                LiveEvents().consume(wire(event))

    def test_live_duplicate_terminal_and_invalid_json_never_resume_candidate(self):
        parser = LiveEvents()
        raw = wire({'type': 'turn.done', 'turn': {'id': 'turn_1', 'role': 'assistant', 'transcript': 'test'}})
        parser.consume(raw)
        with self.assertRaises(RuntimeError):
            parser.consume(raw)
        with self.assertRaises(RuntimeError):
            parser.consume(wire({'type': 'session.updated', 'session': {'id': 'session_1'}}))

    def test_transcript_role_cannot_contradict_the_event_direction(self):
        for kind, role in (('input_transcript.added', 'assistant'), ('output_transcript.added', 'user')):
            with self.assertRaisesRegex(RuntimeError, '^event-invalid$'):
                LiveEvents().consume(wire({'type': kind, 'item': {'id': 'turn_1', 'role': role, 'text': 'test'}}))

    def test_responses_complete_valid_message_and_multiline_sse(self):
        raw = text_response('合法譯文')
        raw = raw.replace(b'data: {', b': keepalive\r\ndata: {').replace(b'\n\n', b'\r\n\r\n')
        self.assertEqual(response_text(raw), '合法譯文')

    def test_empty_terminal_output_uses_only_matched_completed_items(self):
        events = [json.loads(block[6:]) for block in text_response('合法譯文').split(b'\n\n') if block]
        events[-1]['response'].update(output=[], model='gpt-6-sol')
        self.assertEqual(response_text(sse(*events), expected_model='gpt-6-sol'), '合法譯文')

    def test_empty_terminal_never_substitutes_deltas_or_incomplete_items(self):
        events = [json.loads(block[6:]) for block in text_response().split(b'\n\n') if block]
        events[-1]['response'].update(output=[], model='gpt-6-sol')
        incomplete = json.loads(json.dumps(events))
        incomplete[2]['item']['status'] = 'in_progress'
        missing = json.loads(json.dumps(events))
        del missing[-1]['response']['output']
        wrong_id = json.loads(json.dumps(events))
        wrong_id[-1]['response']['id'] = 'resp_other'
        wrong_model = json.loads(json.dumps(events))
        wrong_model[-1]['response']['model'] = 'other'
        for values in (events[:2] + events[3:], incomplete, missing, wrong_id, wrong_model, events[:-1]):
            with self.subTest(), self.assertRaisesRegex(RuntimeError, '^event-invalid$'):
                response_text(sse(*values), expected_model='gpt-6-sol')

    def test_done_reasoning_rejects_explicit_incomplete_status_in_both_terminal_forms(self):
        for status in ('completed', None, 'in_progress', 'incomplete', 'failed', ''):
            for empty in (True, False):
                events = [json.loads(block[6:]) for block in text_response().split(b'\n\n') if block]
                reasoning = {'type': 'reasoning', 'id': 'reason_1', 'summary': [], 'status': status}
                events[1:1] = [{'type': 'response.output_item.added', 'item': dict(reasoning, status='in_progress')},
                               {'type': 'response.output_item.done', 'item': reasoning}]
                events[-1]['response']['output'] = [] if empty else [reasoning, events[-2]['item']]
                with self.subTest(status=status, empty=empty):
                    if status in ('completed', None):
                        self.assertEqual(response_text(sse(*events)), 'translation')
                    else:
                        with self.assertRaisesRegex(RuntimeError, '^event-invalid$'):
                            response_text(sse(*events))

    def test_done_reasoning_status_can_be_omitted_when_done_event_is_matched(self):
        events = [json.loads(block[6:]) for block in text_response().split(b'\n\n') if block]
        reasoning = {'type': 'reasoning', 'id': 'reason_1', 'summary': []}
        events[1:1] = [{'type': 'response.output_item.added', 'item': reasoning},
                       {'type': 'response.output_item.done', 'item': reasoning}]
        events[-1]['response']['output'] = []
        self.assertEqual(response_text(sse(*events)), 'translation')

    def test_responses_reject_missing_completed_wrong_id_and_tool(self):
        normal = text_response()
        cases = [normal[:normal.rfind(b'data: ')],
                 normal.replace(b'"id": "resp_1", "status"', b'"id": "resp_2", "status"'),
                 normal.replace(b'"role": "assistant"', b'"role": "user"'),
                 sse({'type': 'response.created', 'response': {'id': 'resp_1'}},
                     {'type': 'response.output_item.added', 'item': {'type': 'function_call', 'id': 'tool_1'}}),
                 normal + sse({'type': 'unknown', 'detail': 'private-secret'})]
        for raw in cases:
            with self.assertRaises(RuntimeError) as caught:
                response_text(raw)
            self.assertNotIn('private-secret', str(caught.exception))

    def test_events_bound_raw_and_reject_duplicate_keys(self):
        for raw in (b'x' * 256001, b'{"type":"turn.done","type":"session.updated"}'):
            with self.assertRaises(RuntimeError):
                LiveEvents().consume(raw)

    def test_configuration_ack_cannot_hide_tools_or_execution_mode(self):
        for session in ({'id': 'session_1', 'tools': [{'type': 'mcp'}]},
                        {'id': 'session_1', 'delegation': {'type': 'responses'}},
                        {'id': 'session_1', 'handoff': {'target': 'client'}}):
            with self.assertRaises(RuntimeError):
                LiveEvents().consume(wire({'type': 'session.updated', 'session': session}))
        raw = text_response()
        with self.assertRaises(RuntimeError):
            response_text(raw.replace(b'"id": "resp_1"}',
                                      b'"id": "resp_1", "tools": [{"type":"function"}]}', 1))


if __name__ == '__main__':
    unittest.main()
