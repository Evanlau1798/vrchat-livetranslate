"""ChatGPT 登入與訂閱語音的 GUI 狀態；Qwen 控制項沿用原實作。"""
import asyncio
import threading

from . import endpoints, gui_settings, gui_update
from .i18n import t


def uses_chatgpt(gui):
    return gui._provider() == endpoints.PROVIDER_CHATGPT


def refresh_key_status(gui):
    chatgpt = uses_chatgpt(gui)
    for name in ('_key_entry', '_key_save_btn', '_key_clear_btn', '_speech_preview_btn', '_tts_preview_btn'):
        widget = getattr(gui, name, None)
        if widget is not None:
            widget.configure(state='disabled' if chatgpt else 'normal')
    for name in ('_speech_voice_combo', '_tts_voice_combo'):
        widget = getattr(gui, name, None)
        if widget is not None:
            widget.configure(state='disabled' if chatgpt else 'normal')
    voice = getattr(gui, '_speech_voice_var', None)
    if voice is not None:
        voice.set('juniper' if chatgpt else gui._effective_speech_voice())
    if not chatgpt:
        return gui_update.refresh_key_status(gui)
    gui._key_status.configure(text=t('ChatGPT 訂閱語音使用 Codex 登入，無需填寫 API key。\n先登入 ChatGPT，再選擇中文 → 日本語與「我說」。'))
    if hasattr(gui, '_key_btn'):
        gui._key_btn.configure(text=t('登入 ChatGPT ▸'))
        gui._key_chip.pack_forget()
        if not gui._key_btn.winfo_manager():
            gui._key_btn.pack()
    if hasattr(gui, '_set_text_input_enabled'):
        gui._set_text_input_enabled(False)


def build_settings_dialog(gui):
    gui_settings.build_settings_dialog(gui)
    gui._provider_save_btn.configure(command=gui._on_save_provider)
    refresh_key_status(gui)


def build_provider_section(gui, body):
    gui_settings.build_provider_section(gui, body)
    gui._provider_save_btn.configure(command=gui._on_save_provider)


def open_signup(gui):
    if not uses_chatgpt(gui):
        return gui_update.open_qianwen_signup(gui)
    if getattr(gui, '_chatgpt_login_busy', False):
        gui._set_status('info', t('ChatGPT 登入仍在進行，請完成瀏覽器中的登入。'))
        return
    gui._chatgpt_login_busy = True
    gui._chatgpt_login_cancel = threading.Event()
    gui._set_status('info', t('請在瀏覽器完成 ChatGPT 登入；完成後可開始翻譯。'))

    async def login():
        from .session.codex_rpc import login_chatgpt
        task = asyncio.create_task(login_chatgpt())
        gui._chatgpt_login_task = (asyncio.get_running_loop(), task)
        if gui._chatgpt_login_cancel.is_set():
            task.cancel()
        try:
            return await task
        finally:
            gui._chatgpt_login_task = None

    def work():
        try:
            ok = asyncio.run(login())
            if not gui._chatgpt_login_cancel.is_set():
                gui._q.put(('status', 'info' if ok else 'error',
                            t('ChatGPT 登入完成，可以開始翻譯。') if ok else t('登入未完成，請重新登入 ChatGPT。')))
        except asyncio.CancelledError:
            pass
        except Exception:
            if not gui._chatgpt_login_cancel.is_set():
                gui._q.put(('status', 'error', t('無法啟動登入，請確認已安裝官方 Codex CLI。')))
        finally:
            gui._chatgpt_login_busy = False
    gui._chatgpt_login_thread = threading.Thread(target=work, daemon=False, name='vlt-chatgpt-login')
    gui._chatgpt_login_thread.start()


def cancel_login(gui):
    cancel = getattr(gui, '_chatgpt_login_cancel', None)
    if cancel is None or cancel.is_set():
        return
    cancel.set()
    owner = getattr(gui, '_chatgpt_login_task', None)
    if owner:
        loop, task = owner
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            pass  # 登入已完成並關閉其 event loop。


def on_save_provider(gui):
    if (any(e.running for e in gui._engines) or getattr(gui, '_pending_starts', 0)
            or getattr(gui, '_power_state', 'idle') != 'idle'):
        gui._set_status('warn', t('当前线路需要先停止翻译，改完再重新开始'))
        return
    gui_settings.on_save_provider(gui)
    refresh_key_status(gui)
