---
name: talking-head
description: A recorded speaker becomes a finished video — transcript-driven cuts, b-roll over the seams, titles, subtitles, mix, render.
---

# Loop: talking head

**Takes** one or more recordings of somebody speaking. **Produces** a cut,
subtitled, mixed, rendered video.

Say which step you are on when you report progress, and put every gate in the
chat in the user's language. Do not run ahead of a gate.

## 1. Brief

Read the skill `brief-intake`. Get audience, the one takeaway, platform and
length. Look at the material first: `get_media_pool`, `describe_project_media`,
`get_project_info`. **Gate:** state the brief in three lines and get agreement.

## 2. Direction

Read `taste-direction`. Write the read and the three dials, say them in the
chat. If the user named a reference video, read `reference-analysis` first and
work from what you measure in it.

## 3. Transcript

`speech_recognition` or `transcribe_media` on every speech clip. This is the
edit map for every step below, so do it before touching the timeline. Mark the
script's beats with `add_marker` (read `storytelling` for what a beat is).

## 4. The cut

`checkpoint_save` labelled `before-cuts`. Read `editing-cuts` and follow it:
false starts, filler, dead air, repetition, tangents — in that order, with
`cut_clips` + `ripple_delete`, then `remove_space` for leftovers. Check the
seams on frames (`render_frame`), not in the transcript alone.

**Gate:** report the new length against the target before continuing. A cut
that lands far from the brief's length is a conversation, not a decision you
make alone.

## 5. Picture over the seams

Read `broll-planning`. Find material you already have —
`describe_project_media`, `find_media`, `find_moments`, `detect_scenes` — and
lay inserts over jump cuts and dead visuals. If a beat has nothing, say so
rather than covering it with a slow push on a still.

Only if the user wants generated inserts: `list_plugins` + `plugin_status`
first, one sample, `chat_assistant` to show it, then the rest. Anything paid is
the user's call.

## 6. Face work (only when asked)

`face-toolkit` (expression, face swap) and `lip-sync-latent` are per-face
effects: check `plugin_status`, `get_face_detection_status` /
`get_faces_at_frame`, checkpoint, run on **one** clip, show a frame, then the
rest.

## 7. Titles

Read `onscreen-text`. Build the first title with `add_title`, check it on a
real frame, then match every other title to it — same position, size and
duration per kind.

## 8. Subtitles

Read `subtitle-craft`. Build from the transcript, fix names and numbers, split
at phrase boundaries, one style for the whole video. Do this **after** the cut
is final; ripple edits move cues out of sync.

## 9. Sound

Read `sound-design`. Speech level first and consistent, music under it and
ducked, fades at the ends, `get_audio_levels` to prove nothing clips.

## 10. Colour

Read `color-grading`. Match shots to each other before any look; check with
`render_contact_sheet`.

## 11. Review

Read `quality-review` and do it properly: contact sheet over the whole
timeline, frames at every cut and title, `get_timeline_summary` for gaps and
one-frame slivers, `get_audio_levels`, and the swap test from
`taste-direction`. Fix what you find, then report findings with timecodes —
including what you could not fix and why.

## 12. Render

`get_render_presets`, confirm the target with the user, `save_project`, then
`render_video`. Watch it with `get_render_jobs`; report where the file is.
