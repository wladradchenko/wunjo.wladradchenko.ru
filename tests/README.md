# Tests

Checks that need no build system and no test framework: each file is a script
that prints what it looked at and exits non-zero when something is wrong.

```
tests/
  agent/check_platform.py    process handling on whichever OS runs it
  agent/check_chat.py        the assistant's chat, without a graphics card
  macos/check_abi.py         does the bundle ask macOS for too much
  macos/check_identity.py    is the bundle the application it claims to be
  macos/check_launch.py      does the bundle start
```

Run any of them directly:

```
python3 tests/agent/check_platform.py
python3 tests/agent/check_chat.py                 # seconds, no network
python3 tests/agent/check_chat.py --with-model    # ~400 MB, talks to a real model
python3 tests/macos/check_abi.py path/to/wunjo.app
```

## One of these is load-bearing

`macos/check_abi.py` runs inside `.github/workflows/package.macos.yml` and fails
the build when it finds something. **Deleting this directory breaks the macOS
release.** The others are safe to remove; that one has to move rather than go.

## Why check_abi.py exists, and why starting the application is not enough

A release once went out declaring it ran on macOS 13.3 while `libKF6ConfigWidgets`
referenced `std::pmr`, which Apple shipped in macOS 14. It installed. It passed
every check that involved running it. It died in dyld before `main()` on a user's
macOS 13.4, with `Symbol not found: __ZNSt3__13pmr15memory_resourceD2Ev`.

Nothing that runs the application on CI can catch that, because **the build
machine is newer than the macOS being targeted** — the symbol is present there
and resolves happily. `check_abi.py` reads the minimum each binary declares and
compares against that instead of against the machine it happens to be on, which
is the only way the question can be asked from a newer Mac.

`check_launch.py` covers the other half: missing frameworks, rpaths pointing at
the build machine, a Qt platform plugin that did not get packaged. Both are
needed and neither substitutes for the other.

The compiler did not warn about the `pmr` call, so libc++'s availability
annotations are switched off somewhere in this toolchain and it will not warn
about the next one either. Until that is understood, `check_abi.py` is what
stands between that class of mistake and a user's Mac. Its `INTRODUCED` table is
a denylist, not a proof — it prints every standard-library symbol it did not
recognise so a new one can be added on purpose rather than discovered by
somebody who downloaded a release.

## What the chat test does and does not reach

The chat is llama-server, then Goose, then the MCP server, then the editor.
A build machine has no editor running, so `check_chat.py` covers the first two
and says so rather than pretending otherwise. Without `--with-model` it checks
the parts that are pure logic — pulling the reply out of Goose's terminal
transcript, the recipe that limits what the assistant can touch, and the
bookkeeping the idle watchdog reads. With it, it downloads the runtime named in
`plugin.json`, fetches a small stand-in model, starts a server, asks it
something and stops it again.

No GPU is needed or assumed: `WUNJO_GPU_BACKEND` is cleared, which is the same
path a user without a card takes.

The stand-in is a 0.5B model rather than the 2.5 GB one the plugin ships,
because what is under test is that a server starts, answers and can be stopped —
not how well a model writes. It has a chat template, which matters: the server
is started with `--jinja` and a model without one does not load.

## Where the platform checks came from

`agent/check_platform.py` exists because the plugin read `/proc` on every system.
Where there is no `/proc` that raised, and the error surfaced through the branch
that reports missing weights — so the first thing the assistant ever said on
macOS and on Windows was that the user should download a model they already had.
The checks run against whichever process listing the machine actually uses, and
reach the other platforms' parsing through mocks, which is worth having and is
not the same as having been there.
