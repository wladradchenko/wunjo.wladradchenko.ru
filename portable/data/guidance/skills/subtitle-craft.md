---
name: subtitle-craft
description: Readable, correctly timed subtitles — line splitting, duration, styling and the checks that catch drift before it ships.
---

# Subtitles

Most viewers read them; on mute they are the video. Generated subtitles are a
starting point, never the deliverable.

## Timing

- A cue starts on the word, not before it. Late is worse than early by a frame.
- Minimum on screen ~1 s even for one word; aim for reading time, roughly one
  second per three words, and hold longer for anything technical.
- Never let a cue cross a shot change or a scene boundary — split it there.
- No overlap between cues; no gap of a few frames between two halves of one
  sentence — that flicker is what makes subtitles tiring.

## Splitting

Split at phrase boundaries, where the speaker breathes. Never split between an
article and its noun, or before the last word of a sentence. Two lines maximum,
the first no longer than the second when you can manage it. Roughly 42
characters per line for horizontal delivery, fewer for vertical.

## Text

Fix what the recogniser got wrong: names, product terms, units, numbers. Keep
the speaker's words — subtitles are not a rewrite — but drop pure filler that
survived the audio edit. Punctuate: a comma is timing information.

## Style

One style for the whole video (`get_subtitle_styles`, `set_subtitle_style`).
Readable against every background: a solid or semi-opaque backing beats an
outline over busy footage. Keep them clear of the platform's UI area and of any
lower-third titles — check with `render_frame`, not by intention.

## Working method

1. `speech_recognition` or `transcribe_media` for the first pass.
2. Correct text and boundaries with `edit_subtitle`, `move_subtitle`,
   `resize_subtitle`, `add_subtitle` / `delete_subtitle`.
3. Verify with `get_subtitles` for overlaps, empty cues and over-long lines.
4. Spot-check on frames — beginning, a busy passage, the end.
5. `export_subtitles` for delivery as a sidecar; burn in only when the platform
   needs it, and checkpoint before you do.

If the audio edit changes after subtitling, re-check the timings; ripple edits
move picture and sound, and stale cues drift a whole section out of sync.

See also: editing-cuts, onscreen-text.
