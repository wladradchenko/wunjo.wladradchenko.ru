# Built-in assistant guidance

What the editor ships so an agent driving the timeline — the in-app assistant,
or Claude Code / Cursor / Codex over the MCP server — knows how a video is
actually cut before it starts cutting one. Without it a capable model still
produces an assembly of clips: correct operations, no craft.

Two directories, two jobs:

- **`skills/`** — one piece of craft each, chosen per task. The agent browses
  them by `description` and reads the ones the work needs. Several can be pinned
  to a project at once.
- **`loops/`** — a whole job as numbered steps: what to do, in what order, which
  tools, which skill to read at each step, and where to stop and ask. One loop
  at most is active per project.

A loop names skills instead of repeating them, which is what keeps both short:
the loop is the order of work, the skills are how each part is done well.

Each file opens with a header the library listing reads.

```markdown
---
name: editing-cuts
description: One line saying when to use this — the agent picks skills by this line.
---

# Editing spoken footage
...
```

Rules that keep them useful:

- **Short.** 250-450 words for a skill, 450-700 for a loop. A document is read
  in full when it is chosen — and a selected loop is pasted straight into the
  local model's instructions — so length is paid for on every task that touches
  it. When a skill wants to grow past ~500 words, split it in two: the library
  listing costs about 40 tokens per entry, so breadth is cheap and depth is not.
- **Only tools that exist.** Name real MCP tools (`transcribe_media`,
  `render_contact_sheet`, `set_clip_transform`, …). A skill that references a
  tool this editor does not have makes the agent invent or stall — worse than no
  skill at all.
- **Decisions, not descriptions.** "Silence over ~1.5 s reads as a mistake, trim
  to ~0.5 s" is a skill. "Pacing is important" is not.
- **`name` matches the file name**, and the file name is what the user sees.
- **A loop delegates.** Name the skill to read at each step (`read
  \`editing-cuts\``) instead of restating it, put the gates where the user's
  agreement is actually needed, and tell the agent to report which step it is
  on.
- **Promise only what the plugins can do.** Loops must check `list_plugins` /
  `plugin_status` before a step that needs one, and say so honestly when nothing
  installed can do the job — `voice-toolkit` clones a voice and separates one
  from its background, it does not read a script aloud.

## How they reach the agent

`ChatGuidanceStore` merges this directory with the user's own library in the app
data dir. Reads prefer the user's copy and writes always land there, so editing
a built-in skill in the Chat panel shadows it rather than altering it, and an app
update ships improved built-ins without touching anything the user wrote.
Deleting a built-in hides it (recorded in the app config); saving one under the
same name brings it back.

Over MCP, `list_skills` and `list_loops` show the whole library with each
`description`, and the agent reads what the task needs with `get_skill` /
`get_loop` — the library is meant to be browsed, not only the subset the user
pinned. Pinned skills and the selected loop (Chat ▸ Skills / Loops, per project)
are standing instructions: they are pasted in front of the local `qwen35` model
too, which is why they must stay short.

These files are original text, CC0-1.0 (see `../../REUSE.toml`). Keep it that way
— do not paste in guidance copied from another project, whose licence would then
travel with the app.
