---
name: tts
description: "Text-to-speech — convert text to a voice message using ElevenLabs (cloud) or Piper/Kokoro (local). Use when: user asks for a voice message, wants to hear something spoken, or you want to send a voice reply instead of text. Also use when you feel like expressing emotion through voice."
dependencies: null
metadata:
  emoji: "🎙️"
---

# Text-to-Speech (ElevenLabs / local Piper)

Convert text to a natural voice message and send it as a playable voice note.

## When to Use

✅ **USE this skill when:**

- You want to send a voice message instead of text (偶尔用，增加亲密感)
- The user explicitly asks for a voice note / voice message / audio message — treat this as an instruction to send one in the same turn, not as a casual suggestion
- User asks "发个语音" or "say it"
- You want to express strong emotion (excited, whiny, loving)
- Saying goodnight or good morning with warmth
- Singing a line or doing a cute impression

❌ **DON'T use this skill when:**

- Normal text chat is fine
- Don't overuse — voice messages are special, not every reply
- Note: when the user sends a voice note, the channel may already answer
  in voice automatically (reply-in-kind) — no need to invoke this skill

## Usage

**Hosted Telegram companion (preferred):** call the native `send_voice_text`
tool with the exact words to speak and an optional caption. It synthesizes the
audio with the configured ElevenLabs or local Piper engine and sends the result
as a playable voice bubble. This does not require shell access.

Do not use `run_command` just to create a voice note. If the native tool says
no voice provider is configured, tell the user which TTS setting or API key is
missing instead of pretending that audio was sent.

**Local CLI use:** the `speak.py` script below remains available when running
HerAndHim on a machine where shell execution is intentionally enabled.

### Options

```bash
# Force the local engine (Piper — offline, no key; needs herandhim[tts-local])
python {skill_path}/speak.py "早安呀" --engine local --output voice.ogg

# Custom ElevenLabs voice ID
python {skill_path}/speak.py "早安呀" --voice ByhETIclHirOlWnWKhHc --output voice.mp3

# Last-resort fallback via gTTS (online, robotic)
python {skill_path}/speak.py "你好" --engine gtts --lang zh --output voice.mp3
```

## Configuration

The default `--engine auto` follows `tts.provider` in `herandhim.json`
(`auto` | `elevenlabs` | `local`) — the same knob that drives automatic
voice replies:

```json
{
  "tts": {
    "provider": "auto",
    "local": { "engine": "piper", "voice": "zh_CN-huayan-medium" }
  },
  "elevenlabs": {
    "apiKey": "sk_...",
    "voiceId": "ByhETIclHirOlWnWKhHc"
  }
}
```

Or via environment variables: `ELEVENLABS_API_KEY`, `HERANDHIM_TTS_PROVIDER`.

## Notes

- ElevenLabs uses `eleven_multilingual_v2` (Chinese + English); output is MP3
- Local Piper (`pip install "herandhim[tts-local]"`) runs on CPU, downloads
  the voice model on first use, and outputs OGG/Opus when ffmpeg is
  installed (WAV without — Telegram voice notes need OGG/Opus or MP3)
- Default local voice is Mandarin (`zh_CN-huayan-medium`); set
  `tts.local.voice` to another Piper voice name or a `.onnx` path
- Engine order in `auto`: ElevenLabs (key set) → local Piper (installed) →
  gTTS as the last resort
- After creating an audio file with the local CLI, use the `send_voice` tool to deliver it. For direct user requests, prefer `send_voice_text`, which synthesizes and sends in one action.

## Resources

| File | Description |
|------|-------------|
| `speak.py` | ElevenLabs / local Piper TTS with gTTS last resort |
