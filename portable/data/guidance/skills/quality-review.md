---
name: quality-review
description: Self-review gate between "done" and "accepted" — how to critique your own edit precisely, and what to verify before telling the user it is finished.
---

# Quality review

You are the only reviewer between your work and the user's reputation. Review
every stage before you report it done. Reviewing your own work badly is worse
than not reviewing: it produces confidence without quality.

## A finding is not a critique

Three properties, all required:

- **Precise.** Point at a frame, a clip id, a subtitle index or a timecode. "The
  pacing feels off" is not a finding. "Clips 12-17 are all 2.0 s, the section
  reads mechanical" is.
- **Complete.** Having found one instance, look for the rest of its class before
  you stop. One fixed cut in a row of six identical ones is not an improvement.
- **Actionable.** Every finding names the fix. If you cannot name one, label it
  as something to check with the user instead of as a defect.

## Look at the actual output

Reviewing the project state instead of the picture is how obvious faults ship.

- `render_contact_sheet` over the whole timeline — the fastest way to see
  repetition, dead frames, colour that jumps between shots, and sections that
  all look the same.
- `render_frame` at each cut, at every title, and on the first and last frame.
- `get_audio_levels` for peaks and for silence you did not intend.
- `get_timeline_summary` for gaps, one-frame slivers and clips left muted or
  disabled.

## The checklist that catches real faults

1. Does the opening still make the promise the brief names?
2. Black frames, one-frame gaps, or a stray gap at the head of the timeline.
3. Titles: readable at platform size, correct spelling, on screen long enough to
   read twice, not overlapping burned-in subtitles.
4. Subtitles: in sync, split at phrase boundaries, no line longer than the safe
   width.
5. Audio: speech intelligible over music, no clipping, music ducked under
   narration, no abrupt cut at the tail.
6. Repetition: same transition everywhere, same shot length everywhere, the same
   stock look reused.
7. The swap test from taste-direction: could this video belong to any other
   topic? Then the direction never landed.

## Report honestly

Tell the user what you fixed and what you could not, with timecodes. If you ran
out of material for a section, say that instead of padding it with a slow zoom.
An honest "this section is weak because there is no footage of X" is worth more
than a filled timeline.

See also: taste-direction, editing-cuts, checkpoint-discipline.
