---
name: color-grading
description: Match shots to each other and give the video one consistent look, without the saturated defaults that make graded footage look processed.
---

# Colour

## Match before you style

Shot matching is the part viewers notice; the "look" is the part they do not.
Get the sequence consistent first: exposure, then white balance, then contrast.
Two adjacent shots that differ in warmth read as an error no grade can hide.

`render_contact_sheet` across the timeline is the fastest way to see mismatches —
they are obvious side by side and invisible one clip at a time.

## Order of operations

1. **Exposure.** Set the midtones on the subject's face, not on the histogram.
2. **White balance.** Neutral where the scene has something neutral; if nothing
   is neutral, match the neighbouring shots.
3. **Contrast.** Set the black point so the darkest part is black, without
   crushing detail out of shadows.
4. **Saturation.** Last, and less than feels right in the moment.
5. **Look.** Only then, and applied to the whole sequence, never per clip.

Apply with `add_effect` and `set_effect_param`; `get_available_effects` lists
what this build offers. `paste_effects` copies a matched grade onto sibling
shots — that is how a sequence stays consistent.

## Defaults to avoid

- Teal-and-orange applied because it is available. It is a choice for a specific
  read, not a synonym for cinematic.
- Crushed blacks and blown highlights for "contrast" — the detail does not come
  back.
- Saturation pushed until skin goes orange. Check faces, always.
- A different grade per shot because each one looked good alone.
- Grading a generated or stock clip to match nothing in particular, when the
  point is to match the footage next to it.

## Skin is the reference

If faces look right, most of the frame looks right. Check them on real frames
with `render_frame` at several points, including the darkest and brightest
shots.

## Delivery

Check the project colour space with `get_project_color_space` before grading, and
do not change it mid-project. Verify the graded result on an exported frame
rather than the monitor preview when the delivery target differs from the
project profile.

See also: cinematography, quality-review.
