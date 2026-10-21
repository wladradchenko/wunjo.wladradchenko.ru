---
name: ai-video-shots
description: Turning approved storyboard frames into video shots — the technical prompt, motion references, spoken lines, voices, and cleaning each shot after generation.
---

# Video shots from storyboard frames

## Avoid

What makes a generated shot look undirected:

- characters looking at random points instead of at each other or the subject;
- a camera angle chosen for no reason;
- light that changes from shot to shot or ignores the scene's light source;
- a space with no logic — doors, furniture and people jumping between shots.

Every rule below keeps one of these out.

## The run

- **One shot per approved frame.** `image-to-video` with the frame as `image`;
  the prompt only says what happens in it — the frame already holds the look.
- **A technical prompt, not a wish.** Open with the camera and the light, then
  the action: "Cinematography: naturalistic master cinematography, shot as if
  on a RED cinema camera, high dynamic range, filmic highlight rolloff. Lens:
  … Light: …". Light follows a classic scheme (key, fill, back) and stays the
  one from the style frame.
- **Motion you decide.** When the movement matters, use `reference-to-video`:
  the concept pictures as `@image…` and a short video of the movement or the
  camera move as `@video1` (each model takes its own number of videos; Gemini
  Omni up to 3 s each).
- **Faces and talk work; physics and action do not.** Conversations,
  reactions and slow camera moves come out clean; falls, fights, splashes and
  crowds come out broken — stage them off screen or cut away.
- **Names trip the filter.** Real people, brands and fairy-tale names
  ("Pinocchio") are refused; describe instead of naming, then resubmit.
- Draft at `360p` and the shortest duration, one shot at a time; remake at the
  final resolution only what the user approved.

## Speech and voices

- Short lines are generated with the shot: write them into the prompt with the
  speaker, `The woman says in Russian: "…"`. Gemini Omni moves the lips to them.
- Longer speech is made separately and laid over: the `speech` card (ElevenLabs
  voices) or a `voice-toolkit` voice; a real voice recorded by the user works
  too.
- One character, one voice through the film: put `voice-toolkit.clone` with
  that character's voice on their lines (`generate_effect`).

## After generation

Every shot has mess in it. Look at it frame by frame (`render_frame`,
`describe_media`) and fix what you can: a mask over a broken detail, a cut
around a bad second, a clean re-run of only that shot. The `depth-map`
plugin renders a depth pass of a clip into the bin, for masks, fog and focus.

*Sources: a VFX studio's breakdown of their AI-cartoon pipeline (direction,
technical prompt, motion reference, compositing); a creator's cartoon tutorial
(one shot per frame, voices); our own test session of ten generated shorts
(Omni lip sync, what fails, the filter, draft prices).*
