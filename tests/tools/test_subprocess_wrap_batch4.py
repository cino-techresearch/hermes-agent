"""TDD: verify batch-4 tools route subprocess calls through spawn_proxy.spawn.

Tests FAIL before implementation (subprocess.run called directly).
Tests PASS after wrapping (spawn_proxy.spawn called instead).

T-067, FR-409/TS-411.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# transcription_tools — _prepare_local_audio and _transcribe_local_command
# ---------------------------------------------------------------------------

class TestTranscriptionToolsSpawnWrap:
    def test_prepare_local_audio_uses_spawn_not_subprocess_run(self, tmp_path, monkeypatch):
        """_prepare_local_audio must route ffmpeg through spawn_proxy.spawn."""
        called: list[str] = []

        def fake_spawn(args, **kwargs):
            called.append("spawn")
            out = args[-1]
            Path(out).touch()
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        import importlib
        import tools.transcription_tools as m
        importlib.reload(m)
        monkeypatch.setattr(m, "_find_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

        src = tmp_path / "audio.ogg"
        src.write_bytes(b"fake")

        with patch("spawn_proxy.spawn", fake_spawn):
            result_path, err = m._prepare_local_audio(str(src), str(tmp_path))

        assert "spawn" in called, "spawn_proxy.spawn was not called — subprocess.run still used"

    def test_transcribe_local_command_uses_spawn_not_subprocess_run(self, tmp_path, monkeypatch):
        """_transcribe_local_command must route through spawn_proxy.spawn."""
        monkeypatch.setenv(
            "HERMES_LOCAL_STT_COMMAND",
            "echo {input_path} {output_dir} {language} {model}",
        )
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "en")

        called: list[str] = []

        def fake_spawn(args, **kwargs):
            called.append("spawn")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "result.txt").write_text("hello spawn", encoding="utf-8")

        def fake_tempdir(prefix=None):
            class _TD:
                def __enter__(self_inner):
                    return str(out_dir)

                def __exit__(self_inner, *a):
                    return False
            return _TD()

        import importlib
        import tools.transcription_tools as m
        importlib.reload(m)

        src = tmp_path / "audio.wav"
        src.write_bytes(b"RIFF0000WAVEdata")

        with patch("spawn_proxy.spawn", fake_spawn), \
             patch("tools.transcription_tools.tempfile.TemporaryDirectory", fake_tempdir):
            result = m._transcribe_local_command(str(src), "base")

        assert "spawn" in called, "spawn_proxy.spawn was not called — subprocess.run still used"


# ---------------------------------------------------------------------------
# voice_mode — TermuxAudioRecorder.start() and _stop_termux_recording()
# ---------------------------------------------------------------------------

class TestVoiceModeSpawnWrap:
    def test_termux_recorder_start_uses_spawn(self, monkeypatch):
        """TermuxAudioRecorder.start() must route subprocess call through spawn_proxy.spawn."""
        called: list[str] = []

        def fake_spawn(args, **kwargs):
            called.append("spawn")
            return subprocess.CompletedProcess(args, 0)

        import importlib
        import tools.voice_mode as vm
        importlib.reload(vm)

        monkeypatch.setattr(vm, "_termux_microphone_command", lambda: "/usr/bin/termux-microphone-record")
        monkeypatch.setattr(vm, "_termux_api_app_installed", lambda: True)

        recorder = vm.TermuxAudioRecorder()
        recorder._recording_path = "/tmp/test.aac"

        with patch("spawn_proxy.spawn", fake_spawn):
            try:
                recorder.start()
            except Exception:
                pass  # OK if it fails for other reasons (e.g., missing dir)

        assert "spawn" in called, "spawn_proxy.spawn was not called — subprocess.run still used"

    def test_termux_recorder_stop_uses_spawn(self, monkeypatch):
        """TermuxAudioRecorder._stop_termux_recording() must route through spawn_proxy.spawn."""
        called: list[str] = []

        def fake_spawn(args, **kwargs):
            called.append("spawn")
            return subprocess.CompletedProcess(args, 0)

        import importlib
        import tools.voice_mode as vm
        importlib.reload(vm)

        monkeypatch.setattr(vm, "_termux_microphone_command", lambda: "/usr/bin/termux-microphone-record")

        recorder = vm.TermuxAudioRecorder()

        with patch("spawn_proxy.spawn", fake_spawn):
            recorder._stop_termux_recording()

        assert "spawn" in called, "spawn_proxy.spawn was not called — subprocess.run still used"


# ---------------------------------------------------------------------------
# tts_tool — _terminate_command_tts_process_tree, _generate_neutts, _generate_piper_tts
# ---------------------------------------------------------------------------

class TestTtsToolSpawnWrap:
    def test_terminate_process_tree_taskkill_uses_spawn(self, monkeypatch):
        """_terminate_command_tts_process_tree taskkill must use spawn()."""
        called: list[str] = []

        def fake_spawn(args, **kwargs):
            called.append("spawn")
            return subprocess.CompletedProcess(args, 0)

        import importlib
        import tools.tts_tool as t
        importlib.reload(t)

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.pid = 12345

        monkeypatch.setattr("os.name", "nt")

        with patch("spawn_proxy.spawn", fake_spawn):
            try:
                t._terminate_command_tts_process_tree(fake_proc)
            except Exception:
                pass

        assert "spawn" in called, "spawn_proxy.spawn was not called for taskkill — subprocess.run still used"

    def test_neutts_conv_cmd_uses_spawn(self, tmp_path):
        """_generate_neutts ffmpeg conv_cmd must route through spawn_proxy.spawn."""
        called: list[str] = []

        def fake_spawn(args, **kwargs):
            called.append("spawn")
            if isinstance(args, list) and len(args) > 0 and "ffmpeg" in args[0]:
                Path(args[-1]).touch()
            return subprocess.CompletedProcess(args, 0)

        wav_path = tmp_path / "out.wav"
        mp3_path = tmp_path / "out.mp3"

        def fake_subprocess_run(cmd, **kwargs):
            # stub the neutts synthesis subprocess.run (not flagged, not wrapped)
            wav_path.write_bytes(b"RIFF0000WAVEdata")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        import importlib
        import tools.tts_tool as t
        importlib.reload(t)

        with patch("spawn_proxy.spawn", fake_spawn), \
             patch("tools.tts_tool.subprocess.run", fake_subprocess_run), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            try:
                t._generate_neutts("hello", str(mp3_path), {
                    "ref_audio": "/tmp/ref.wav",
                    "ref_text": "hi",
                    "model": "model",
                    "device": "cpu",
                })
            except Exception:
                pass

        assert "spawn" in called, "spawn_proxy.spawn was not called for neutts conv_cmd — subprocess.run still used"

    def test_piper_conv_cmd_uses_spawn(self, tmp_path):
        """_generate_piper_tts ffmpeg conv_cmd must route through spawn_proxy.spawn."""
        called: list[str] = []

        def fake_spawn(args, **kwargs):
            called.append("spawn")
            return subprocess.CompletedProcess(args, 0)

        wav_path = tmp_path / "out.wav"
        mp3_path = tmp_path / "out.mp3"
        wav_path.write_bytes(b"RIFF0000WAVEdata")

        fake_voice = MagicMock()
        fake_voice.synthesize_wav = MagicMock()

        import importlib
        import tools.tts_tool as t
        importlib.reload(t)

        with patch("spawn_proxy.spawn", fake_spawn), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("tools.tts_tool._resolve_piper_voice_path", return_value="/tmp/voice.onnx"), \
             patch("tools.tts_tool.PiperVoice") as mock_piper_cls:
            mock_piper_cls.load.return_value = fake_voice
            try:
                t._generate_piper_tts("hello", str(mp3_path), {"voice": "en_US-amy-medium"})
            except Exception:
                pass

        assert "spawn" in called, "spawn_proxy.spawn was not called for piper conv_cmd — subprocess.run still used"
