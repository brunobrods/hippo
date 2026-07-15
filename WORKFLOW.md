# WORKFLOW.md

Process rules for Claude Code in this repo. Code style itself lives in
`CLAUDE.md` — this file governs *how to work*, not how the code should look.

## 1. Clarify before implementing

Before writing or editing code for any non-trivial feature or fix, ask
clarifying questions rather than guessing. "Non-trivial" means anything
beyond a one-line change, typo fix, or literal instruction with no
ambiguity — e.g. a new class, a new endpoint interaction, a change to
order/trading logic, or any request where the desired behaviour, scope, or
edge cases aren't fully specified.

Keep asking follow-up questions until the requirements are actually clear —
don't settle for a single round if the answer reveals new ambiguity. Once
satisfied, restate the plan briefly before implementing so there's a final
checkpoint to correct course.

Skip this step only when the request is unambiguous and fully scoped
already (e.g. "rename this variable", "fix this exact traceback").

## 2. Verify with sub-agents after implementing

After implementing a non-trivial change, before declaring it done, run both
of the following:

1. **Code review** — invoke the `code-review` skill on the diff to catch
   correctness bugs and reuse/simplification/efficiency issues (style
   conformance to `CLAUDE.md`'s Elegant Objects rules included).
2. **Requirement verification** — spawn a fresh `Agent` (general-purpose,
   read-only) with no memory of the implementation discussion, and give it
   the original ask plus the diff. Have it check independently whether the
   implementation actually satisfies what was asked — not just whether the
   code looks clean. Report its findings before considering the task done.

Only report the task complete after both checks have run and any findings
they raised have been addressed or explicitly deferred with the user's
sign-off.