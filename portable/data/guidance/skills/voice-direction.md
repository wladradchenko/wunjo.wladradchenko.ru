---
name: voice-direction
description: Direct generated or recorded narration — pacing, emphasis, how to write for the ear, and how to judge a take before building the whole video on it.
---

# Voice direction

## Write for the ear

Narration is read once, without rewinding. Short sentences. One clause of
subordination at most. Put the subject first and the verb early.

- Numbers spoken, not written: "about a third", not "33.7%".
- No parentheses, no semicolons, no "as mentioned above" — the ear has no page.
- Read it aloud before generating. If you stumble, so will the voice.
- Spell out anything a synthesiser will mangle: units, acronyms that are said as
  words, foreign names.

## Direct the performance

Mark the intent per line before generating: where to slow down, which word
carries the sentence, where the pause goes. One emphasis per sentence — an
emphasis on every third word is a machine reading, and it is exactly what makes
generated narration recognisable.

Put paragraph breaks where the speaker should breathe. Generated voices run
sentences together unless the text gives them a reason not to.

## One take before the whole script

Generate one paragraph, listen, and put it in the chat for the user before
committing to the full narration — see checkpoint-discipline. Voice is the
element users have the strongest opinions about, and regenerating everything
after the fact wastes time and, on paid providers, their money.

Check `list_plugins` and `plugin_status` for what is available and ready before
promising a voice at all; for a paid provider, get the user's agreement first.

## Judge the take

Reject and regenerate when you hear: a wrong-word emphasis that changes the
meaning, a rushed clause, a mispronounced name, a rising intonation at the end
of a statement, or a change in energy between paragraphs. These are not
subtleties; they are the difference between narration and text-to-speech.

## Then cut to the voice

Once the narration is approved it becomes the spine: lay it down first, then
build picture against it. Transcribe it with `transcribe_media` so subtitles and
cuts share the same timings.

See also: sound-design, subtitle-craft, storytelling.
