---
name: footage-montage
description: A pile of existing footage plus narration becomes an edited montage — logging, selection, assembly against the voice, pacing, match and mix.
---

# Loop: footage montage

**Takes** a folder of footage the user already has (shoot, archive, screen
recordings, stills) and either a script or a recorded narration. **Produces** an
edited montage.

Say which step you are on. The failure mode of this loop is a slideshow with
music over it — read `quality-review` for the check that catches it, and take it
seriously at step 9.

## 1. Brief and direction

Read `brief-intake`, then `taste-direction`. Audience, one takeaway, platform,
length; the read and the three dials. **Gate:** state both in the chat and get
agreement before importing anything.

## 2. Bring the material in

`import_media` / `import_media_glob`, then **log it**:
`describe_project_media` for what each clip is, `detect_scenes` for the usable
segments inside long files, `get_bin_clip_properties` for resolution and frame
rate so you do not later scale past the source.

Report what you actually have — and what the brief needs that is missing. That
sentence is worth more than an hour of covering a gap badly.

## 3. The spine

Narration first, always. If the user has a recording, import and
`transcribe_media` it. If they have only a script, read `storytelling` and
write the beats, then get them approved before any voice exists; generate the
voice only if a plugin can (`list_plugins`, `plugin_status`) and read
`voice-direction` when it can.

`add_marker` at every beat. The markers are the structure the rest of the loop
builds against.

## 4. Select and order

Read `broll-planning`. For each beat pick the shot that shows what the narration
cannot say. One clip may serve one beat; do not stretch a favourite across
three. Where nothing fits, mark the beat as unresolved and raise it — never fill
it with an unrelated shot.

**Gate:** list the beat → clip mapping in the chat before assembly.

## 5. Assemble

`checkpoint_save` labelled `before-assembly`. Lay picture against the narration
with `append_clips` / `insert_clip` (or `build_timeline` for a first pass), then
trim each shot to the beat with `trim_clip`. Keep the narration untouched
underneath — it is the reference, not a clip to nudge.

## 6. Pacing

Read `editing-cuts` and `cinematography`. Vary shot length; cut on the word the
shot illustrates; hold longer where there is text in frame; let a frame breathe
after dense information. Six shots of equal length in a row is the slideshow
tell — fix it here, not at review.

Straight cuts by default; `add_transition` only where the cut carries a meaning
a cut cannot.

## 7. Titles and subtitles

`onscreen-text` for cards and lower thirds — one style, checked on a real frame.
`subtitle-craft` from the narration transcript, after the picture is locked.

## 8. Match and mix

`color-grading`: match shots to each other first — archive and phone footage in
one timeline will not match by themselves; `render_contact_sheet` shows it
instantly. Then `sound-design`: narration level consistent, music under it and
ducked, room tone continuous, fades at the ends.

## 9. Review

Read `quality-review` and run the contact sheet over the whole timeline. Then
answer honestly, in the chat:

- Is this an edit, or stills and slow pushes with music over them?
- Does every shot show something the narration does not say?
- Could this video belong to any other topic if the narration were swapped?

If the answer to the last one is yes, the direction never landed — say so and
propose what to change. Do not render your way past it.

## 10. Render

`get_render_presets`, confirm with the user, `save_project`, `render_video`,
report the path and the findings you did not fix.
