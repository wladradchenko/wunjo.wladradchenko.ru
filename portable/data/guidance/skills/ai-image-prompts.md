---
name: ai-image-prompts
description: Pictures for a generated cartoon or story — character sheet from photos, the style frame, empty locations, storyboard frames from references — with the image cards.
---

# Pictures for a generated story

The pictures decide the film: a video model animates what its start frame
shows and keeps its look. Make them in this order, each approved before the
next, with `text-to-image` and `reference-to-image` (drafts at `0.5K`).

## 1. Character sheet

From the user's own photos (front, profile, three-quarter) as references:
"This is my heroine, @image1 @image2 @image3. Make a character sheet in
<style>: clean bright 3D on a clean white background, only 4 angles, close-up."
The style is chosen here — "Stylized 3D comedy", "3D Pixar/Disney" — and
every later prompt repeats the same words.

## 2. The style frame

The first frame fixes how the whole film looks: style, light, rendering.
Write it in labelled parts, the sheet as `@image1`:

- **Character** — as in the reference: hair, face, earrings, tattoos, the dress.
- **Second character**, if any — age, clothes, what they do in this moment.
- **Composition** — where they stand, and empty space kept for a title.
- **Location** — the room, its furniture, what is blurred behind.
- **Atmosphere** — warm, family, slightly comic.
- **Light** — golden hour, soft sunlight, cinematic lighting, soft shadows,
  subtle depth of field.
- **Quality** — highly detailed, expressive faces, clean polished render,
  vibrant but soft colours.

A title can be painted into this frame later: the frame as reference, "add the
3D text '…' in purple and pink with a touch of mint; the characters' pose must
differ".

## 3. Locations

One `text-to-image` per place, same style words, and say it is empty: "Warm
cozy 3D Pixar/Disney style kitchen, light wood, large sunlit window …, no
characters or living beings present, only the empty kitchen space."

## 4. Storyboard frames

Every frame is a `reference-to-image` with the style frame as `@image1` (the
characters, their look and light) and a framing reference as `@image2` — any
picture whose camera angle and composition you want: "My two characters from
@image1. An opening frame framed like @image2, as if looking at them from the
next room. Same room as @image1 from another angle. The lighting exactly as in
@image1."

Build each frame on purpose, not at random: the angle, the shot size, where
everyone stands and where they look all serve the moment. Tools that work:

- a frame inside the frame (a doorway, a peephole, a box opening);
- leading lines that run to the subject; gaze and gesture on one line;
- depth in three layers: blurred foreground, the subject sharp, background;
- warm back light that rims hair and shoulders off the background.

Set `aspect_ratio` to the video's format: the video keeps the frame's shape.
Show every frame (`render_bin_frame`) before any video is made.

*Sources: a creator's published cartoon tutorial (steps 1–6) and a VFX
studio's breakdown of their AI-cartoon pipeline (frame rules).*
