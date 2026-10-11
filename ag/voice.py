"""Spoken report summaries with ElevenLabs text-to-speech.

The sentence is always written here from the report's own data, never taken from the
client, so the API key can't be used to speak arbitrary text. Each clip is generated
once and cached on disk; replays cost no credits.

Environment:
  AG_ELEVENLABS_KEY    API key (Text to Speech permission is enough). Voice is off without it.
  AG_ELEVENLABS_VOICE  voice ID (default: "George", JBFqnCBsd6RMkjVDRZzb)
  AG_ELEVENLABS_MODEL  model ID (default: eleven_multilingual_v2)
"""
import hashlib
import json
import os
import urllib.error
import urllib.request

API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128"
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_MODEL = "eleven_multilingual_v2"
LEVELS = ((70, "high"), (40, "medium"), (1, "low"))
VERDICT = {"confirmed": "Reported by a resident and verified.",
           "rejected": "Reported by a resident, who disputed this result.",
           "missed": "A resident reported a barrier here that the model missed."}


class Voice:
    def __init__(self, env=os.environ):
        self.key = env.get("AG_ELEVENLABS_KEY")
        self.url = env.get("AG_ELEVENLABS_URL") or API  # override only for local testing
        self.voice = env.get("AG_ELEVENLABS_VOICE") or DEFAULT_VOICE
        self.model = env.get("AG_ELEVENLABS_MODEL") or DEFAULT_MODEL

    @property
    def enabled(self):
        return bool(self.key)

    def cache_tag(self, text):
        """Short hash of everything that changes the audio, so a new voice/model re-renders."""
        return hashlib.sha1(f"{self.voice}|{self.model}|{text}".encode()).hexdigest()[:10]

    def synthesize(self, text, timeout=25):
        """-> MP3 bytes. Raises RuntimeError with a readable message on failure."""
        req = urllib.request.Request(
            self.url.format(voice=self.voice),
            data=json.dumps({"text": text, "model_id": self.model}).encode(),
            headers={"xi-api-key": self.key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ElevenLabs {e.code}: {e.read()[:200].decode(errors='replace')}") from None
        except OSError as e:
            raise RuntimeError(f"ElevenLabs unreachable: {e}") from None


def _things(obstructions):
    kinds = sorted({o.get("class", "vehicle") for o in obstructions}) or ["vehicle"]
    names = [("a " + k) for k in kinds]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def speech_text(score, flags, obstructions=(), pedestrians=(), verdict=None, resident=False):
    """One short, calm sentence set that works read aloud to someone walking."""
    parts = []
    if flags.get("sidewalk_obstruction"):
        many = len({o.get("class") for o in obstructions}) > 1 or len(obstructions) > 1
        parts.append(f"{_things(obstructions).capitalize()} {'are' if many else 'is'} blocking the sidewalk.")
    if flags.get("pedestrian_in_roadway"):
        n = len(pedestrians)
        parts.append("People are walking in the street." if n != 1 else "Someone is walking in the street.")
    if flags.get("no_visible_sidewalk"):
        parts.append("There is no visible sidewalk on this street.")
    if parts:
        level = next((name for m, name in LEVELS if score >= m), "low")
        text = "Caution. " + " ".join(parts) + f" Repair priority {score} out of 100, {level}."
    else:
        text = "No sidewalk barrier was found here."
    if resident:
        text += " " + VERDICT.get(verdict, "Reported by a resident.")
    return text
