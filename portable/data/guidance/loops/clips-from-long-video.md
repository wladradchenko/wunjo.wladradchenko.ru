---
name: clips-from-long-video
description: One long recording becomes several short vertical clips, each standing on its own — moment selection, reframing, burned-in subtitles.
---

# Loop: clips from a long video

**Takes** a long recording (talk, interview, stream, podcast with picture).
**Produces** several short vertical clips, each self-contained.

Say which step you are on. Each clip is delivered as its own sequence, so
nothing here destroys the source edit.

## 1. Brief

Read `brief-intake`. How many clips, for which platform, how long each. Then
read `short-form` — it governs every decision below.

## 2. Understand the recording

`speech_recognition` / `transcribe_media` for the full transcript;
`detect_scenes` and `get_timeline_summary` for structure; `get_audio_levels` to
find the loud, animated passages.

## 3. Choose the moments

`find_moments` plus the transcript. A moment qualifies only if it makes sense
with no setup, states something in its first sentence, and ends on a point —
not merely because it is loud or funny in context.

**Gate:** list the candidates with timecodes and one line each in the chat, and
let the user pick before you build anything. This is the step where a wrong
guess wastes the most work.

## 4. One clip per sequence

For each approved moment: `set_zone_in` / `set_zone_out` around it,
`extract_zone`, and put it in its own sequence (`create_sequence`,
`set_active_sequence`). Name the sequence after the moment, not `seq3`.

## 5. Tighten each clip

Read `editing-cuts`. Cut harder than in long form: no preamble, no trailing
pleasantry, no dead air at all. The first spoken words must be the claim — if
they are not, move the in-point or drop the clip and say why.

## 6. Vertical

`set_project_profile` for the target aspect on the clip's sequence. Reframe
every shot with `set_clip_transform` so the speaker is in the middle band —
never deliver a letterboxed horizontal frame. Check on a real frame
(`render_frame`) that nothing important sits where the platform's UI covers it.

## 7. Subtitles, burned in

Read `subtitle-craft`. Short lines, high contrast, clear of the UI zone. Assume
the sound is off: anything essential that is spoken must be readable.

## 8. Titles and sound

A hook card only if it earns its place (`onscreen-text`). Levels and fades per
`sound-design` — a clip that starts mid-word or ends on a hard cut reads as
broken.

## 9. Review each clip separately

Read `quality-review`. Watch the first second and the last second of every
clip. Ask of each one: does it work for somebody who has not seen the source?
Report the ones you think are weak instead of shipping filler.

## 10. Render

Confirm the preset, `save_project`, `render_video` per sequence, report every
output path. Say plainly which clips you would publish and which you would not.
