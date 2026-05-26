"""TDD: verify batch-4 tools route subprocess calls through spawn_proxy.spawn.

Tests FAIL before implementation (subprocess.run called directly, spawn not importable).
Tests PASS after wrapping (spawn bound in each module, intercepted via module-level patch).

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
        """_prepare_local_audio must route ffmpeg through spawn (not subprocess.run)."""
        called: list[str] = []

        def fake_spawn(args, **kwargs):
            called.append("spawn")
            out = args[-1]
            Path(out).touch()
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        import tools.transcription_tools as m
        monkeypatch.setattr(m, "_find_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")

        src = tmp_path / "audio.ogg"
        src.write_bytes(b"fake")

        with patch("tools.transcription_tools.spawn", fake_spawn):
            result_path, err = m._prepare_local_audio(str(src), str(tmp_path))

        assert "spawn" in called, "spawn was not called — subprocess.run still used directly"

    def test_transcribe_local_command_uses_spawn_not_subprocess_run(self, tmp_path, monkeypatch):
        """_transcribe_local_command must route shell command through spawn."""
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

        import tools.transcription_tools as m

        src = tmp_path / "audio.wav"
        src.write_bytes(b"RIFF0000WAVEdata")

        with patch("tools.transcription_tools.spawn", fake_spawn), \
             patch("tools.transcription_tools.tempfile.TemporaryDirectory", fake_tempdir):
            result = m._transcribe_local_command(str(src), "base")

        assert "spawn" in called, "spawn was not called — subprocess.run still used directly"


# ---------------------------------------------------------------------------
# voice_mode — TermuxAudioRecorder.start() and _stop_termux_recording()
# ---------------------------------------------------------------------------

class TestVoiceModeSpawnWrap:
    def test_termux_recorder_start_uses_spawn(self, monkeypatch):
        """TermuxAudioRecorder.start() must route subprocess call through spawn."""
        called: list[str] = []

        def fake_spawn(args, **kwargs):
            called.append("spawn")
            return subprocess.CompletedProcess(args, 0)

        import tools.voice_mode as vm

        monkeypatch.setattr(vm, "_termux_microphone_command", lambda: "/usr/bin/termux-microphone-record")
        monkeypatch.setattr(vm, "_termux_api_app_installed", lambda: True)

        recorder = vm.TermuxAudioRecorder()
        recorder._recording_path = "/tmp/test.aac"

        with patch("tools.voice_mode.spawn", fake_spawn):
            try:
                recorder.start()
            except Exception:
                pass  # OK if it fails for other reasons (e.g., missing dir)

        assert "spawn" in called, "spawn was not called — subprocess.run still used directly"

    def test_termux_recorder_stop_uses_spawn(self, monkeypatch):
        """TermuxAudioRecorder._stop_termux_recording() must route through spawn."""
        called: list[str] = []

        def fake_spawn(args, **kwargs):
            called.append("spawn")
            return subprocess.CompletedProcess(args, 0)

        import tools.voice_mode as vm

        monkeypatch.setattr(vm, "_termux_microphone_command", lambda: "/usr/bin/termux-microphone-record")

        recorder = vm.TermuxAudioRecorder()

        with patch("tools.voice_mode.spawn", fake_spawn):
            recorder._stop_termux_recording()

        assert "spawn" in called, "spawn was not called — subprocess.run still used directly"


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

        import tools.tts_tool as t

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.pid = 12345

        monkeypatch.setattr("os.name", "nt")

        with patch("tools.tts_tool.spawn", fake_spawn):
            try:
                t._terminate_command_tts_process_tree(fake_proc)
            except Exception:
                pass

        assert "spawn" in called, "spawn was not called for taskkill — subprocess.run still used"

    def test_neutts_conv_cmd_uses_spawn(self, tmp_path):
        """_generate_neutts ffmpeg conv_cmd must route through spawn."""
        called: list[str] = []

        def fake_spawn(args, **kwargs):
            called.append("spawn")
            if isinstance(args, list) and len(args) > 0 and "ffmpeg" in args[0]:
                Path(args[-1]).touch()
            return subprocess.CompletedProcess(args, 0)

        wav_path = tmp_path / "out.wav"
        mp3_path = tmp_path / "out.mp3"

        def fake_subprocess_run(cmd, **kwargs):
            # stub the neutts synthesis call (not wrapped — uses subprocess.run directly)
            wav_path.write_bytes(b"RIFF0000WAVEdata")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        import tools.tts_tool as t

        with patch("tools.tts_tool.spawn", fake_spawn), \
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

        assert "spawn" in called, "spawn was not called for neutts conv_cmd — subprocess.run still used"

    def test_piper_conv_cmd_uses_spawn(self, tmp_path):
        """_generate_piper_tts ffmpeg conv_cmd must route through spawn."""
        import wave as _wave

        called: list[str] = []

        def fake_spawn(args, **kwargs):
            called.append("spawn")
            return subprocess.CompletedProcess(args, 0)

        mp3_path = tmp_path / "out.mp3"

        def fake_synthesize_wav(text, wav_file, syn_config=None):
            # Write a minimal valid WAV so wave.close() doesn't fail.
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00")

        fake_voice = MagicMock()
        fake_voice.synthesize_wav = fake_synthesize_wav
        fake_piper_cls = MagicMock()
        fake_piper_cls.load.return_value = fake_voice

        import tools.tts_tool as t

        with patch("tools.tts_tool.spawn", fake_spawn), \
             patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("tools.tts_tool._resolve_piper_voice_path", return_value="/tmp/voice.onnx"), \
             patch("tools.tts_tool._import_piper", return_value=fake_piper_cls):
            try:
                t._generate_piper_tts("hello", str(mp3_path), {"piper": {"voice": "en_US-amy-medium"}})
            except Exception:
                pass

        assert "spawn" in called, "spawn was not called for piper conv_cmd — subprocess.run still used"
