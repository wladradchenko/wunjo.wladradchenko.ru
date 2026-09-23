---
name: localization
description: A finished video reaches another language — corrected transcript, translated and retimed subtitles, and a dub only when a plugin can truly make one.
---

# Loop: localization

**Takes** a finished video. **Produces** the same video for another language:
subtitles at minimum, a dubbed track when the machine can honestly make one.

Say which step you are on. Never overwrite the original — work in a copy of the
sequence (`get_sequences`, `create_sequence`, `set_active_sequence`).

## 1. Scope

Ask for: the target language(s), and whether the user wants **subtitles only**,
or **dubbing**. Ask nothing else — everything else you can read from the
project (`get_project_info`, `get_media_pool`).

Then check what is possible before promising it: `list_plugins` and
`plugin_status`. A plugin that reads text aloud in the target language is what
dubbing needs; the bundled `voice-toolkit` clones a voice into existing speech
and separates a voice from what is behind it — useful below, but it does not
read a script. **If nothing can speak the target language, say so plainly and
deliver the subtitled version.** Do not fake a dub by other means.

## 2. Source transcript, corrected

`speech_recognition` or `transcribe_media`, then fix it against what is actually
said: names, products, numbers, units. Everything downstream inherits these
mistakes, so this is the step to be pedantic in.

## 3. Translate

You do the translation yourself — you are the language model in this pipeline.
Read `subtitle-craft` first, then translate **for the ear and the eye**, not
literally:

- keep the speaker's register; a formal source stays formal;
- keep names, product terms and units in the form the target audience uses;
- do not expand. Translations run longer than the source and the cue has a fixed
  duration — cut words, keep meaning;
- flag anything you are unsure of instead of inventing it: idioms, puns, legal
  or medical wording. **Gate:** show the doubtful lines in the chat.

## 4. Retime

Translated text does not fit the source timings. Fix them: `add_subtitle`,
`edit_subtitle`, `move_subtitle`, `resize_subtitle`, and `get_subtitles` to
check for overlaps, over-long lines and cues crossing a shot change. Reading
speed governs — hold longer, never let a cue outlive its shot.

## 5. Style and deliver the subtitles

One style for the whole video (`set_subtitle_style`), readable at platform size,
clear of any existing burned-in text — check on real frames with `render_frame`.
`export_subtitles` for a sidecar; burn in only if the platform needs it, and
`checkpoint_save` before you do.

If the target language needs a different font or line length than the original
style, say so — do not silently squeeze it into the old one.

## 6. Dub, when a plugin can

`checkpoint_save` labelled `before-dub`.

1. Free the background: run `voice-toolkit`'s voice separation on the speech
   clips so music and effects survive under a new voice (`run_plugin`, then
   `plugin_job_status`; output lands in the bin).
2. Read `voice-direction`, then generate **one paragraph** with whatever plugin
   can speak the language. Put it in the chat and wait. Voice is what users have
   the strongest opinions about, and anything paid is their decision.
3. Once approved, generate the rest, lay it against the picture, and fix drift
   per line rather than stretching audio.
4. Mix per `sound-design`: new voice at the old speech level, background bed
   under it, `get_audio_levels` to prove nothing clips.
5. If the speaker is on camera and the user wants the mouth to match, that is
   `lip-sync-latent` on the face — one clip, one frame shown, then the rest.

## 7. Review

Read `quality-review`. Play the seams: every subtitle against its shot, the dub
against the picture at the start, middle and end. Check nothing of the original
language is left visible — titles, on-screen text, the subtitle style's own
language. Report what you could not translate confidently, with timecodes.

## 8. Render

Confirm preset and output naming per language with the user, `save_project`,
`render_video`, report every path.
