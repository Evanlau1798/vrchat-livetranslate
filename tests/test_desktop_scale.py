"""桌面字幕「缩放」（issue #42）：字号与面板一起缩放，1080p 正常而 4K 上字太小的正解。

## 治什么

issue #42 的原话：「调整大小位置，透明度，字体大小（因为不同分辨率表现力不尽相同，
比如 1080p 的屏幕看着很正常，放在 4k 分辨率下字体就会缩放的特别小」。
位置（拖动）与透明度此前已有；缺的是**大小 / 字号**，而且两者必须一起动 ——
只放大字号不放大面板会把面板撑爆，只放大面板不放大字号看着还是小。

## 钉住什么

1. `DesktopOverlayConfig.from_dict()`：默认 `scale=1.0` = 完全不改变已有行为
   （老用户的 size_px / font_size 一个像素都不许动）；
2. 给了 scale 时**字号与面板一起**按比例缩放，且在用户显式写的基础值之上施加；
3. `scale` 夹到 0.5~2.0，垃圾值（"abc" / None / NaN）回落 1.0 并留痕；
4. `scale` **不从 `overlay:` 段继承**（那是手腕屏的配置，语义不同）；
5. 界面「微调 ▸」里的缩放滑块：初值与窗口同源、只有真拖过才写盘
   （写 `desktop_overlay.scale` 一个键），并且**落盘后热重载出来的窗口配置确实是缩放过的**
   —— 只验「写了个数」不算数，得验它真的改变了解析结果。

## 离线可跑

只走「解析 → 写盘 → 再解析」这条真实路径；全部读写 `out/` 下的沙箱配置，
仓库根的 `config.yaml`（用户个人配置）全程不碰。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 干净环境（CI）没有 API key，而 load_config 默认 require_key=True 会 SystemExit。
# 给一个拼接出来的假 key：本文件只验配置读写，与 key 真假无关。
os.environ.setdefault("DASHSCOPE_API_KEY", "sk" + "-ws-" + "deskscale0123456789abcdef")

GUI_SANDBOX_DIR = ROOT / "out" / "desktop_scale_cfg"
GUI_SANDBOX = GUI_SANDBOX_DIR / "config.yaml"
FAKE_MISSING_TITLE = "vlt-no-such-window-42"
GUI_SANDBOX_SCALE = 1.20


# ------------------------------------------------------------------ 纯逻辑
def test_scale_default_is_noop() -> None:
    """默认 1.0 = 老行为一个像素都不变（升级不该悄悄改任何人的字幕大小）。"""
    from vlt.output.desktop_overlay import DesktopOverlayConfig

    cfg = DesktopOverlayConfig.from_dict({})
    assert cfg.scale == 1.0, cfg.scale
    assert tuple(cfg.size_px) == (1024, 360), cfg.size_px
    assert (cfg.font_size, cfg.source_font_size) == (36, 29), (cfg.font_size, cfg.source_font_size)

    # 显式写了基础值、但没写 scale → 原样保留
    cfg2 = DesktopOverlayConfig.from_dict({"size_px": [800, 200], "font_size": 40})
    assert tuple(cfg2.size_px) == (800, 200) and cfg2.font_size == 40
    print("  默认 scale=1.0：基础值原样保留 OK")


def test_scale_scales_panel_and_fonts_together() -> None:
    """字号与面板**一起**缩放 —— 只改其中一样都会坏事。"""
    from vlt.output.desktop_overlay import DesktopOverlayConfig

    cfg = DesktopOverlayConfig.from_dict({"scale": 1.5})
    assert tuple(cfg.size_px) == (round(1024 * 1.5), round(360 * 1.5)), cfg.size_px
    assert cfg.font_size == round(36 * 1.5), cfg.font_size
    assert cfg.source_font_size == round(29 * 1.5), cfg.source_font_size
    assert cfg.scale == 1.5
    print(f"  scale=1.5 → 面板 {cfg.size_px} 字号 {cfg.font_size}/{cfg.source_font_size} OK")


def test_scale_applies_on_top_of_explicit_base_values() -> None:
    """缩放施加在**用户显式写的基础值**之上（不是写死在 1024x360 / 36 上）。"""
    from vlt.output.desktop_overlay import DesktopOverlayConfig

    cfg = DesktopOverlayConfig.from_dict({
        "size_px": [800, 200], "font_size": 40, "source_font_size": 30, "scale": 2.0,
    })
    assert tuple(cfg.size_px) == (1600, 400), cfg.size_px
    assert (cfg.font_size, cfg.source_font_size) == (80, 60), (cfg.font_size, cfg.source_font_size)
    print("  scale 施加在显式基础值之上（800x200/40/30 × 2.0）OK")


def test_scale_clamped_and_invalid_falls_back() -> None:
    """夹到 0.5~2.0；垃圾值回落 1.0 并留一行日志（禁静默）。"""
    import contextlib
    import io

    from vlt.output.desktop_overlay import SCALE_DEFAULT, SCALE_MAX, SCALE_MIN, DesktopOverlayConfig

    assert DesktopOverlayConfig.from_dict({"scale": 0.01}).scale == SCALE_MIN
    assert DesktopOverlayConfig.from_dict({"scale": 99}).scale == SCALE_MAX
    # 夹取后仍然真的作用到尺寸上（不是只改了个显示值）
    assert tuple(DesktopOverlayConfig.from_dict({"scale": 0.01}).size_px) == \
        (round(1024 * SCALE_MIN), round(360 * SCALE_MIN))
    for bad in ("abc", None, float("nan")):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cfg = DesktopOverlayConfig.from_dict({"scale": bad})
        assert cfg.scale == SCALE_DEFAULT, f"{bad!r} → {cfg.scale}"
        assert tuple(cfg.size_px) == (1024, 360), f"{bad!r} 竟然改了尺寸：{cfg.size_px}"
        if bad == "abc":
            assert "scale" in buf.getvalue(), f"垃圾值必须留痕：{buf.getvalue()!r}"
    print("  scale 夹取 0.5~2.0 / 垃圾值回落 1.0 且留痕 OK")


def test_scale_not_inherited_from_overlay_section() -> None:
    """`overlay:` 段没有 scale 语义：绝不继承（继承会让手腕屏配置串到桌面字幕上）。"""
    from vlt.output.desktop_overlay import DesktopOverlayConfig

    cfg = DesktopOverlayConfig.from_dict({}, visual={"scale": 2.0})
    assert cfg.scale == 1.0, f"scale 被 overlay 段继承了：{cfg.scale}"
    assert tuple(cfg.size_px) == (1024, 360), cfg.size_px
    print("  scale 不从 overlay 段继承 OK")


# ------------------------------------------------------------------ 界面滑块 + 落盘
def _make_gui_sandbox() -> None:
    """把模板改成沙箱配置：目标窗口标题指向一个一定不存在的窗口、`scale` 改成 1.20。

    用行级替换而不是整文件重写：沙箱要跟用户手写配置一样保留注释。
    `scale` 故意不是默认值（1.20）：这样「滑块初值是否与窗口解析同源」和
    「没拖过滑块就把它冲回默认值」两件事都验得出来。
    """
    text = (ROOT / "config.example.yaml").read_text(encoding="utf-8")
    text, n = re.subn(r"(?m)^(  game_title:\s*)\S+", rf"\g<1>{FAKE_MISSING_TITLE}", text)
    assert n == 1, f"模板里应有且只有 1 行 `  game_title:`，命中 {n} 处"
    text, n = re.subn(r"(?m)^(  scale: )1\.00(.*)$", rf"\g<1>{GUI_SANDBOX_SCALE}\g<2>", text)
    assert n == 1, f"模板里应有且只有 1 行 `  scale: 1.00`，命中 {n} 处"
    GUI_SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    # .gitattributes 规定源码 LF：显式传 newline，别让 Windows 把整份配置写成 CRLF
    GUI_SANDBOX.write_text(text, encoding="utf-8", newline="\n")


def _read_gui_sandbox() -> dict:
    return yaml.safe_load(GUI_SANDBOX.read_text(encoding="utf-8"))


def test_gui_scale_slider_writes_only_when_touched() -> None:
    """滑块初值 1.0 / 没拖过不写盘 / 拖过写 `desktop_overlay.scale`，且解析出来真的变大。"""
    import vlt.config as _cfg_mod
    import vlt.i18n as _i18n
    import vlt.gui as _gui_mod

    _make_gui_sandbox()
    # 界面语言跟随系统语言（CI 与外国机器是英文系统）→ 钉死 zh，控件树排布才稳定。
    # ⚠️ 必须在构造窗口**之前**打桩。产品代码不依赖这个补丁。
    saved = (_cfg_mod.DEFAULT_CONFIG, _gui_mod.DEFAULT_CONFIG, _i18n.detect_system_language)
    _cfg_mod.DEFAULT_CONFIG = GUI_SANDBOX
    _gui_mod.DEFAULT_CONFIG = GUI_SANDBOX          # `_save_desktop_cfg` 用的是这个常量
    _i18n.detect_system_language = lambda: "zh"

    from vlt.gui import TranslationGUI
    from vlt.output.desktop_overlay import DesktopOverlayConfig

    gui = None
    try:
        gui = TranslationGUI()
        if gui._update_check_job is not None:      # 启动 3 秒后自动查更新：绝不真连 GitHub
            gui._root.after_cancel(gui._update_check_job)
            gui._update_check_job = None
        gui._root.update()

        # ① 初值：与窗口解析同源 = 沙箱里的 1.20（不是兜底 1.0）
        assert abs(float(gui._desktop_scale_var.get()) - GUI_SANDBOX_SCALE) < 1e-9, \
            f"滑块初值应同源于配置里的 {GUI_SANDBOX_SCALE}，实际 {gui._desktop_scale_var.get()}"
        assert gui._desktop_scale_touched is False, "没碰过滑块就不该算'动过了'"

        # ② 没拖过 → 落盘不许动它（这条把「无条件写盘 / 冲回默认值」的老毛病钉死）
        assert gui._desktop_out is None, "本用例不该起字幕窗（只验滑块 → 配置这条路）"
        gui._save_desktop_cfg()
        assert abs(float(_read_gui_sandbox()["desktop_overlay"]["scale"]) - GUI_SANDBOX_SCALE) < 1e-9, \
            "没拖过滑块就把 scale 写进配置了"

        # ③ 真的拖了 → 标签跟着变、写盘、内存同步
        gui._desktop_scale_var.set(1.5)
        gui._on_desktop_scale()
        assert gui._desktop_scale_touched is True
        assert gui._desktop_scale_lbl.cget("text") == "1.50×", gui._desktop_scale_lbl.cget("text")
        gui._save_desktop_cfg()                    # 防抖那 300ms 直接跑同一个函数
        data = _read_gui_sandbox()
        assert abs(float(data["desktop_overlay"]["scale"]) - 1.5) < 1e-9, \
            f"拖过滑块后 scale 该写进配置：{data['desktop_overlay'].get('scale')}"
        assert gui._cfg.desktop_overlay.get("scale") == 1.5, \
            "内存没同步：关掉桌面字幕再勾上会用旧值重建窗口"

        # ④ 关键一条：字幕窗热重载时**真的会变大**（只验"写了个数"是空过）
        hot = DesktopOverlayConfig.from_dict(data["desktop_overlay"],
                                             visual=data["overlay"] or {})
        assert hot.scale == 1.5
        assert tuple(hot.size_px) == (round(1024 * 1.5), round(360 * 1.5)), hot.size_px
        assert hot.font_size == round(36 * 1.5), hot.font_size
    finally:
        if gui is not None:
            try:
                gui._root.destroy()
            except Exception:  # noqa: BLE001
                pass
        (_cfg_mod.DEFAULT_CONFIG, _gui_mod.DEFAULT_CONFIG,
         _i18n.detect_system_language) = saved
        # 沙箱文件留在 out/ 下即可（已 gitignore）；用户的 config.yaml 全程没被碰过
    print(f"  界面缩放滑块：初值同源 {GUI_SANDBOX_SCALE} / 没拖不写盘 / "
          "拖过写 scale=1.5 且热重载真的变大 OK")


if __name__ == "__main__":
    print("test_desktop_scale:")
    test_scale_default_is_noop()
    test_scale_scales_panel_and_fonts_together()
    test_scale_applies_on_top_of_explicit_base_values()
    test_scale_clamped_and_invalid_falls_back()
    test_scale_not_inherited_from_overlay_section()
    test_gui_scale_slider_writes_only_when_touched()
    print("ALL PASSED")
