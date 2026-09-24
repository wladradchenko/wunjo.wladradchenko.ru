---
name: editing-cuts
description: Where to cut spoken footage and how to pace it — filler words, false starts, dead air, redundancy — using the transcript as the edit map.
---

# Editing spoken footage

## Work from the transcript, not the waveform

Run `transcribe_media` (or `speech_recognition`) first. Word timings are the
edit map: they tell you where a sentence starts, where the speaker stalled and
where a take restarts. Cutting by eye on the waveform costs ten times the time
and lands on breaths.

Save a checkpoint with `checkpoint_save` before the first destructive pass.

## What to remove, in this order

1. **False starts and restarts.** When a sentence is attempted twice, keep the
   last attempt. This is the single biggest quality gain in talking-head
   footage.
2. **Filler words** — "um", "uh", "you know", "like" as a verbal tic. Cut on
   word boundaries from the transcript, not mid-phoneme.
3. **Dead air.** Silence over ~1.5 s reads as a mistake; trim to ~0.5 s. Keep a
   deliberate pause after a punchline or a number — that one is content.
4. **Repetition.** The same point made twice: keep the better delivery, cut the
   other. Check against the beats from storytelling, not by feel.
5. **Tangents.** If a passage does not serve the takeaway in the brief, it goes,
   however good it is.

Use `cut_clips` plus `ripple_delete` so the timeline stays closed up; check for
leftover gaps with `remove_space` before moving on.

## Pace

- Cut on the end of a thought, not the end of a breath.
- Vary shot length. Six identical-length shots in a row read as machine output
  even when each cut is correct.
- Let a frame breathe after dense information — a beat of silence is how the
  viewer catches up.
- Do not speed-ramp speech to hit a target length. Cut content instead;
  `set_clip_speed` on dialogue is audible and cheap-sounding.

## Keep the cut honest

Do not join two halves of different sentences into a claim the speaker never
made. Do not remove a qualifier ("in most cases") to make a statement stronger.
When a cut changes meaning, leave the longer version and tell the user.

## Check the seams

Sample the frames around each cut with `render_frame`: a cut that reads fine in
the transcript can land on a blink, a hand mid-gesture or a jump in head
position. Cover an unavoidable jump with b-roll or a soft transition rather
than leaving it.

See also: quality-review, broll-planning, subtitle-craft.
