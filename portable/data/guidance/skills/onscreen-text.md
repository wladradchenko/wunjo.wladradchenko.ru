---
name: onscreen-text
description: Titles, lower thirds and text cards that can actually be read — size, duration, hierarchy and restraint.
---

# On-screen text

## Readable first

- Size for the smallest screen the video will play on. A phone viewer is the
  default, not your monitor.
- Contrast: text over footage needs a scrim, a solid panel or a dark gradient.
  An outline over busy video is a compromise that fails on both.
- On screen long enough to read twice at a slow pace. Anything shorter than
  about 1.5 s is decoration, not information.
- Keep it out of the platform's UI zone and away from burned-in subtitles.

## One idea per card

A title is not a paragraph. Three to seven words. If the text needs a second
line to make sense, either the narration should carry it or it deserves its own
card.

Do not caption what the narration just said, word for word — that is not
reinforcement, it is noise. Put on screen what is hard to hear: numbers, names,
spellings, the term you just introduced.

## Hierarchy and consistency

Two type sizes and one weight contrast are enough for most videos. Every title
of the same kind — every lower third, every stat card — uses the same position,
size, colour and duration. Inconsistent placement between two lower thirds is
noticed even by viewers who could not say why.

Set the type once, then reuse it: build the first title with `add_title`, check
it on a real frame with `render_frame`, and match the rest to it. `edit_title`
changes an existing one rather than stacking a second on top.

## Motion

Text should arrive and leave quickly and then hold still — a short fade or a
small offset. Text that animates for its whole time on screen cannot be read,
and per-word kinetic type is a style choice for a high-motion read, not a
default. Match the motion dial from taste-direction.

## Spelling

Proofread every card. A typo in a title is the one mistake every viewer catches,
and it discredits the rest of the video. Read names and product terms back to
the user in the chat when you are not certain.

See also: taste-direction, subtitle-craft, quality-review.
