---
name: broll-planning
description: Choose supporting visuals that carry meaning instead of decorating narration, and place them so they cover the cut rather than distract from it.
---

# B-roll planning

B-roll is an argument, not wallpaper. Each insert should show something the
narration cannot say, or it should not be there.

## Plan from the script, not from what is available

For each beat, write what the viewer needs to see: the mechanism, the scale, the
before/after, the person's reaction. Then look for material. Choosing from
what happens to be on hand is how videos end up with a stock shot of a keyboard
under a sentence about latency.

If nothing serves a beat, a clean frame with the number on it beats an unrelated
clip. Say so rather than filling.

## Placement

- Cut in on the word it illustrates, not two sentences early.
- Hold long enough to be read — 1.5-3 s for a simple shot, longer if there is
  text in the frame, and honour the density dial from taste-direction.
- Use an insert to cover a jump cut in the speaker: hide the seam under the
  visual, then return.
- Come back to the speaker before the next beat. Long unbroken b-roll turns a
  person into a voiceover.
- Never cut away in the middle of the most important sentence — the viewer reads
  the face there.

## Find what is already in the project

`describe_project_media` and `find_media` search what the user has; `find_moments`
and `detect_scenes` locate usable segments inside long footage. Check there
before generating anything: real material from the user's own shoot beats a
generated approximation almost every time, and it costs nothing.

For generated inserts, check `list_plugins` and `plugin_status` before promising
anything, and keep the look consistent with the read — a photoreal insert in an
illustrated video breaks the frame language.

## Continuity

Keep screen direction, time of day and colour consistent across an insert
sequence; `render_contact_sheet` shows a mismatch instantly. Vary framing
between consecutive inserts — two similar wides in a row read as a mistake.

## Rights

Only use material the user owns or that the source licenses for this use, and
record where each clip came from in the chat. Do not silently pull footage whose
licence you cannot name.

See also: storytelling, cinematography, editing-cuts.
