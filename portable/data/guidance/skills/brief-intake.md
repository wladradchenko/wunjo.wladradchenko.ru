---
name: brief-intake
description: Turn a vague request ("make a video about X") into a brief with audience, promise, platform and length before touching the timeline.
---

# Brief intake

A request like "make a video about our product" does not contain enough to
judge any later decision against. Spend one exchange getting the brief, then
work without asking again.

## Ask for what you cannot guess

Ask in one message, at most four questions, in the user's language:

1. **Who watches it** — and what they already know. "Developers who have never
   heard of us" and "our own users" are different videos.
2. **The one thing they should take away.** One sentence. If the user gives
   three, ask which survives if the other two are cut.
3. **Where it goes and how long** — a 30-second vertical clip and a 12-minute
   YouTube piece share nothing but the topic.
4. **What material exists** — footage, a script, a voice recording, brand
   assets, or nothing. Call `get_media_pool` and `describe_project_media`
   first; do not ask about material you can see for yourself.

Do not ask about style yet, and never ask the user to choose tools or plugins.

## Write it down

State the brief back in three lines — audience, takeaway, platform and length —
and put it in the chat with `chat_assistant`. From here on, every cut, asset
and title is judged against those three lines. When a later request contradicts
them, say so once and follow the user.

## Guess when you must

If the user does not answer and says "just do it", assume: audience = general,
platform = the project profile from `get_project_info`, length = the material
you have. State the assumption in the chat before starting, so the user can
correct it cheaply rather than after a render.

See also: taste-direction, storytelling.
