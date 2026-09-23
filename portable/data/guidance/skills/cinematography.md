---
name: cinematography
description: Framing, movement and shot variety for assembled or generated footage — how to make a sequence look composed rather than defaulted.
---

# Cinematography

## Frame with intent

- Put the subject's eyes on the upper third; leave the look-room on the side
  they face.
- Headroom: a sliver, not a third of the frame. Do not centre a talking head
  vertically by default.
- Keep one clear subject per frame. If two things compete, the viewer reads
  neither.
- Watch the edges. A stray bright object at the border pulls the eye off the
  subject every time.

Check framing on real pixels with `render_frame`, not on the assumption that a
generated or imported shot is composed.

## Cut between different sizes

A sequence of shots at the same size reads as coverage, not as an edit. Move
between wide, medium and close, and change size across a cut — same size plus
same angle is a jump cut. Use the close for the important sentence.

## Movement

Movement needs a reason: follow a subject, reveal information, or settle onto a
detail. A slow push into a still image for lack of anything better is the
default that makes AI-assembled video recognisable — use it once, not as the
grammar of the whole piece.

If you animate a still, set `set_clip_transform` keyframes with a slight ease
and a short hold at both ends; constant linear drift for the full clip is what
looks cheap. Never scale beyond the source resolution — check with
`get_bin_clip_properties` first.

## Transitions

Straight cuts by default. Reach for `add_transition` when the cut carries a
meaning a cut cannot: a dissolve for time passing, a fade for a section break.
The same wipe on every edit is the most reliable signature of an unconsidered
edit. See `get_available_transitions` for what this project actually offers.

## Match the delivery

Check the project profile with `get_project_info` before framing decisions.
Vertical delivery moves the safe area and kills wide compositions: reframe with
`set_clip_transform` per shot rather than letting the whole video letterbox, and
keep faces and text out of the areas the platform's UI covers.

See also: taste-direction, color-grading, broll-planning.
