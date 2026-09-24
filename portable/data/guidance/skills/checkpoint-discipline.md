---
name: checkpoint-discipline
description: When to save checkpoints, when to stop and ask the user, and how to make a long automated edit reversible instead of frightening.
---

# Checkpoints and approval

The user is watching a timeline they care about being rewritten by a machine.
Reversibility is what makes that tolerable.

## Checkpoint before every destructive pass

Call `checkpoint_save` with a label that says what is about to happen —
`before-filler-cuts`, `before-subtitle-burn` — not `checkpoint1`. Do it before:

- any batch of cuts, ripple deletes or speed changes;
- running a plugin that replaces media (`run_plugin`, `generate_effect`);
- restyling or regenerating subtitles;
- anything that touches more than three clips at once.

`undo` handles a single mistake; a checkpoint handles the case where the whole
approach was wrong. Use `undo_status` to check the stack before relying on it.

## Sample of one before batch of many

Before generating twenty assets, styling every subtitle, or applying an effect
across a timeline: do exactly one, show it, and ask.

Render the sample with `render_frame` or `render_bin_frame`, put it in the chat,
and say plainly what you are about to repeat: "this is the subtitle style for
the whole video — good to continue?" A wrong style caught on one clip costs
nothing; caught after twenty it costs the user's trust and their GPU time.

## Stop and ask when

- the operation is not reversible from inside the editor (overwriting source
  media, deleting bin clips, rendering over an existing file);
- a paid plugin or API is involved — cost is the user's decision, always;
- the material contradicts the brief and you are about to reinterpret it;
- the user's own skills or loop conflict with what the task needs. Say which
  instruction you cannot follow and why, then follow the user's answer.

## Narrate while you work

The user cannot see your reasoning, only the timeline moving. Use `chat_user` to
echo the request, `chat_assistant` for what you are doing and why, and
`chat_tool_start` / `chat_tool_progress` / `chat_tool_end` around long
operations — in the user's language. A silent agent that rewrites a timeline is
indistinguishable from a broken one.

## Do not chain past a failure

If a step fails, stop and report it. Continuing on top of a failed step buries
the cause and produces a mess that is harder to unpick than the original
problem.

See also: quality-review, brief-intake.
