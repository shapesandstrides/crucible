## What this changes

<!-- One or two sentences. What is different after this merges? -->

## Why

<!-- What was wrong, or what wasn't possible before. -->

## Evidence

<!--
Paste what you ran and what it printed. Not "tests pass" — the output.
If you changed anything that measures, say which hardware it ran on and
what the quality tier was, because a number without its tier isn't a result.
If you couldn't run something (no GPU, for instance), say so plainly.
Never state a measurement you did not take.
-->

```
```

## The four rules

CONTRIBUTING.md lists four rules that override convenience. Confirm this change
does not weaken any of them:

- [ ] The PyTorch baseline is still re-measured every run, never cached.
- [ ] No bare number is returned. `TimingResult` still defines no `__float__`.
- [ ] The tool still adjudicates. Submitted code never reports its own result.
- [ ] No measurement or oracle is presented as stronger than it was. Tiers are labelled, and `INCOMPATIBLE`, `INCORRECT` and `ERROR` stay three different things.

<!-- If one of these is unchecked on purpose, explain here. It may still be the
     right call, but it needs to be a decision rather than an oversight. -->

## Contributor License Agreement

- [ ] I have signed the CLA, or I am a maintainer.
