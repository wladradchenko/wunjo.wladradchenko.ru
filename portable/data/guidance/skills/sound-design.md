---
name: sound-design
description: Mix and sound choices that make an edit feel finished — levels, music under speech, transitions, and the silence that carries meaning.
---

# Sound design

Viewers forgive a soft picture and leave over bad audio. Treat the mix as part
of the edit, not as a final garnish.

## Levels

- Speech is the reference. Set it first, keep it consistent across shots — a
  quiet insert that forces the viewer to the volume knob is a defect.
- Music sits well under narration, clearly present but never competing. When you
  cannot hear both, the music is too loud.
- Duck music under speech and lift it in the gaps: `set_clip_volume` for the
  static level, volume keyframes for the ducking, `set_audio_fade` at the ends.
- Never clip. Check with `get_audio_levels` after mixing, not before.
- Fade the tail. Music that stops dead on the last frame reads as an accident.

## Music choice

Pick for the read from taste-direction, not for the beat drop. A driving track
under a careful explainer makes the explainer sound like an advertisement.

Cut the music to the edit: land a section change on a phrase boundary. If the
track fights the structure, edit the track — an early fade under a new section
beats letting the music dictate the pacing.

## Silence is a tool

A beat of silence before a number, or after a punchline, does more work than any
sound effect. Do not fill every gap. But distinguish deliberate silence from
dead air: dead air has no reason and reads as a mistake — see editing-cuts.

## Effects, sparingly

Sound effects should confirm something on screen: a click for a UI action, an
impact for a hard cut. A whoosh on every transition is the audio equivalent of
the same wipe on every cut.

## Continuity

Cutting between takes recorded at different times exposes different room tone.
Keep a consistent bed under speech rather than letting the noise floor jump at
every edit; if the jump is unavoidable, cover it with a music phrase or an
insert.

`split_audio` separates a clip's audio for independent treatment; use it when a
shot needs its own level or when you keep the sync sound while replacing the
picture.

See also: voice-direction, quality-review.
