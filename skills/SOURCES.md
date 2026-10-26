# Bundled agent skills

These skills are for this project's coding agents. They are vendored from
[Anthropic's official skills repository](https://github.com/anthropics/skills),
at commit [`dbd4588f9e1033efb41dad4bef2f7947c8993d44`](https://github.com/anthropics/skills/commit/dbd4588f9e1033efb41dad4bef2f7947c8993d44)
(upstream commit date: 2026-10-09; retrieved: 2026-10-26).

| Skill | Purpose | Upstream source | License |
| --- | --- | --- | --- |
| `frontend-design` | Visual direction, typography, layout, UI copy, responsiveness and design critique when writing frontend code. | [Pinned directory](https://github.com/anthropics/skills/tree/dbd4588f9e1033efb41dad4bef2f7947c8993d44/skills/frontend-design) | Apache-2.0; full original text in `frontend-design/LICENSE.txt`. |
| `webapp-testing` | Inspect and test local web applications with Python Playwright, screenshots and console logs. | [Pinned directory](https://github.com/anthropics/skills/tree/dbd4588f9e1033efb41dad4bef2f7947c8993d44/skills/webapp-testing) | Apache-2.0; full original text in `webapp-testing/LICENSE.txt`. |

All `SKILL.md`, license, helper and example files in these two directories are
unchanged upstream files. Each downloaded file was checked against its Git blob
hash from the pinned GitHub tree. `SHA256SUMS` records SHA-256 digests of all eight
upstream files. To verify them locally, run from this `skills/` directory:

```sh
shasum -a 256 -c SHA256SUMS
```

The upstream repository contains skills under other licenses; the licenses above
apply only to these two imported directories. No font, document-processing or
media package from the upstream repository is bundled here.

## Runtime requirements

- `frontend-design` contains instructions only and requires no extra package.
- `webapp-testing` requires Python's `playwright` package and an installed browser
  such as Chromium in the environment where the generated application runs.
  For a prepared Python environment, the usual setup is
  `python -m pip install playwright` followed by
  `python -m playwright install chromium`. Reading the skill does not install
  these dependencies or prove that browser tests have run.
- The helper `webapp-testing/scripts/with_server.py` uses Python's standard
  library, starts the supplied server command(s), waits for their ports, runs a
  supplied test command, and stops the server processes. The application may
  separately require Node.js, npm, or other dependencies.
- Upstream examples contain sample URLs, selectors, and output paths such as
  `http://localhost:5173` and `/mnt/user-data/outputs`. Adapt those values to the
  actual app and the agent's allowed workspace before running a test.
- Paths mentioned inside a skill are relative to that skill's own directory.
  Skills and resources are read as text; importing or loading a skill does not
  execute its scripts. A resource used in the coding sandbox must be copied to
  that sandbox and invoked through its existing execution tools. The skill does
  not grant access outside the sandbox or add missing tool capabilities.

The source files are pinned so that runtime behavior does not silently change
with upstream `main`. To update them, review a new upstream commit, preserve the
complete skill directories and licenses, and update this file and `SHA256SUMS`.
