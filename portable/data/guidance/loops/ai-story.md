---
name: ai-story
description: A cartoon or story made from generated shots — character sheet, style frame, locations, storyboard, video per frame, voices, edit, render.
---

# Loop: generated story

**Takes** an idea or a script, and photos of the people if the hero is real.
**Produces** a finished short in which every shot was generated: the same
characters and the same look throughout, lines spoken on screen.

The order is the one creators who make these films follow: pictures first,
each approved, and video only from approved frames. Say which step you are on,
and put every gate in the chat in the user's language. Every run costs the
user credits: ask the price first (`generator_card_action`, `"price"`), say it
with the balance, and let the editor's window take their answer.

## 1. Brief and tools

Read `brief-intake` and `short-form`. Get the format (9:16 for shorts), the
length, the characters and the style. `list_plugins`, then `plugin_status` for
`online-toolkit` (pictures, video, speech) and `voice-toolkit` (voices). If the
user wants many runs without a question each, ask once with
`spending_allowance`.

## 2. Script

Read `storytelling`. Write the script as shots of 5–10 s: who, where, what
happens, each spoken line with its speaker. Read `ai-video-shots` and drop or
restage what the models cannot do. **Gate:** the shot list and a price estimate
for the drafts.

## 3. Characters

Read `ai-image-prompts`. One character sheet per character (step 1 there), from
the user's photos with `reference-to-image`, or from a description with
`text-to-image`. The image cards go to the bin. **Gate:** the user approves
every sheet — nothing later can fix a face they do not like.

## 4. Style frame and locations

The style frame (step 2 in `ai-image-prompts`), then the empty locations (step
3). **Gate:** the style frame — it sets the look of every shot after it.

## 5. Storyboard

One frame per shot (step 4 in `ai-image-prompts`): the style frame as `@image1`,
a framing reference as `@image2`. Get a bin picture's file with
`get_bin_clip_properties`. **Gate:** all frames, in script order
(`render_bin_frame` each).

## 6. Video drafts

Read `ai-video-shots`. One `image-to-video` per approved frame, `360p`, shortest
duration, `place="bin"`, one at a time — look at each (`describe_media`,
`render_bin_frame`) before the next. A scene longer than one shot continues
from the last frame of the part before: `render_frame(last_frame,
max_short_side=1080)` on the timeline before titles or subtitles are on it,
that file as the next card's `image`. **Gate:** the drafts in order; then
remake the approved ones at the agreed resolution, price first.

## 7. Assembly

`checkpoint_save`. `insert_clip` the finals in script order, with `source_in`
and `source_out` to drop dead frames at the ends. Read `editing-cuts` for the
seams. Clean each shot as `ai-video-shots` says under "After generation".

## 8. Voices

Keep the lines generated with the shots where they work. Longer speech and a
voice that must stay the same come from step "Speech and voices" in
`ai-video-shots`: `speech_recognition` for where each line is, `split_clip` at
the speaker changes, `voice-toolkit.clone` with the character's voice on their
pieces. Read `voice-direction` to judge the result.

## 9. Titles, subtitles, sound

Read `onscreen-text` for titles (`add_title`, checked on `render_frame`), then
`subtitle-craft` and `speech_recognition` once the cut is final. Read
`sound-design`: dialogue first; music (`music` card) only if the user agrees
to the price.

## 10. Review and render

Read `quality-review` and do it: frames at every cut, `get_timeline_summary`,
`get_audio_levels`. Report what you fixed and what you could not.
`get_render_presets`, confirm the target, `save_project`, `render_video`.
