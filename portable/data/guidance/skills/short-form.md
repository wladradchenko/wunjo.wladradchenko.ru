---
name: short-form
description: Vertical clips under a minute — how to open, how to pace, and what to cut when the format punishes everything the long form allows.
---

# Short form

Under a minute, vertical, sound often off, and the viewer is one flick from
leaving. Everything below follows from that.

## The first second

Open on the payoff or the tension. No title card, no logo, no "hi everyone".
The first frame should be the most interesting frame in the clip, and the first
spoken words should be the claim.

If the clip needs setup to make sense, the clip is the wrong length for the
idea.

## One idea only

A short holds exactly one point. Second points do not shorten, they dilute.
Cut everything that does not serve the single takeaway — see brief-intake.

## Pace

- Change what is on screen every 1.5-3 s: a cut, a reframe, a text card. Not
  faster — a video the viewer cannot follow is not fast, it is unreadable.
- Cut tighter than in long form: remove the breath, the pause, the pleasantry.
- No dead frames at all. There is no room for a beat that does not pay.
- End the moment the point lands. A trailing outro is where retention dies.

## Made for mute

Assume the sound is off. Subtitles are not optional here — burn them in, high
contrast, positioned clear of the platform UI. See subtitle-craft. Anything
essential that is spoken must also be readable.

## Frame for vertical

Set the project profile for the target before editing, not after. Reframe every
horizontal source with `set_clip_transform` — a letterboxed horizontal clip
wastes two thirds of the screen and reads as a repost. Keep faces and text in
the middle band; the top and bottom are covered by platform UI.

## Delivery

Confirm the target platform's aspect and duration limits with the user, and check
`get_project_info` matches them before rendering. Export a frame and look at it
at phone size before calling it done.

See also: long-form, subtitle-craft, editing-cuts.
