"""ChatGPT 訂閱路徑：設定、工廠與 GUI 不可要求或轉送 Qwen API key。"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlt.config import load_config
from vlt.session.base import SessionConfig, create_session
from vlt import gui_engine, ui_state


class IntegrationTests(unittest.TestCase):
    def test_subscription_config_never_loads_api_key(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            path.write_text('session:\n  provider: chatgpt\ndirections:\n  mine:\n    source_lang: zh\n    target_lang: ja\n', encoding='utf-8')
            with patch('vlt.config.load_api_key', return_value='qwen-secret') as key_loader:
                cfg = load_config(path, require_key=True)
            self.assertEqual(cfg.session_base['api_key'], '')
            self.assertEqual(cfg.session_base['provider'], 'chatgpt')
            key_loader.assert_not_called()
            scfg = cfg.direction('mine').to_session_config(cfg.session_base)
            self.assertEqual(scfg.target_lang, 'ja')
            self.assertEqual(type(create_session(scfg)).__name__, 'ChatGPTLiveSession')

    def test_factory_uses_subscription_provider(self):
        cfg = SessionConfig(provider='chatgpt', target_lang='ja')
        self.assertEqual(type(create_session(cfg)).__name__, 'ChatGPTLiveSession')

    def test_refresh_clears_api_key_without_reading_credentials(self):
        cfg = SimpleNamespace(session_base={'provider': 'chatgpt', 'api_key': 'qwen-secret'})
        with patch('vlt.config.load_api_key') as loader:
            ui_state.refresh_api_key_in_cfg(cfg)
        self.assertEqual(cfg.session_base['api_key'], '')
        loader.assert_not_called()

    def test_gui_subscription_starts_without_key_and_disables_typed_qwen(self):
        variable = lambda value: SimpleNamespace(get=lambda: value)
        ctx = gui_engine.EngineCtx(
            cfg=SimpleNamespace(session_base={'provider': 'chatgpt', 'api_key': ''}, directions={}, output={}),
            direction_var=variable('mine'), chatbox_var=variable(True),
            lang_pair={'source': 'zh', 'target': 'ja'}, root=SimpleNamespace(after=lambda *_: None),
            set_text_input_enabled_fn=lambda value: typed.append(value),
        )
        typed = []
        with patch.object(gui_engine, 'start_engine') as start_engine, \
             patch.object(gui_engine, 'start_overlay'), patch.object(gui_engine, 'start_desktop'):
            self.assertFalse(gui_engine.start(ctx))
            start_engine.assert_called_once_with(ctx, 0)
        self.assertEqual(typed, [False])

    def test_subscription_dual_is_rejected_before_start(self):
        variable = lambda value: SimpleNamespace(get=lambda: value)
        messages = []
        ctx = gui_engine.EngineCtx(
            cfg=SimpleNamespace(session_base={'provider': 'chatgpt', 'api_key': ''}),
            direction_var=variable('dual'), set_status_fn=lambda level, text: messages.append((level, text)),
        )
        with patch.object(gui_engine, 'start_engine') as start:
            gui_engine.start(ctx)
            start.assert_not_called()
        self.assertIn('單向', messages[-1][1])


if __name__ == '__main__':
    unittest.main()
