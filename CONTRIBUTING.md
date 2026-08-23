# Contributing

## Licensing and the open-core boundary

This repository — the local verification engine, the library and the CLI — is
Apache-2.0. It is meant to be adopted freely, run inside other people's CI, and
cited as a neutral reference. Permissive terms are a requirement for that, not a
preference: copyleft and source-available licenses are routinely blocked by the
legal teams at exactly the organisations we need as users.

A hosted API and cloud catalog are planned. **They will live in a separate,
private repository under proprietary terms.** Nothing about Apache-2.0 on this
repo obliges us to publish that, and the split is deliberate:

| Stays open (this repo, Apache-2.0) | Stays closed (separate repo) |
|---|---|
| Shape/dtype/stride generation | Record ingestion and the hosted API |
| Oracle, tolerance, shrinking | The aggregated drift corpus |
| Interleaved measurement, tiers | Cross-user, cross-hardware aggregation |
| Verdicts, run-record *format* | Auth, accounts, dashboard |
| The library and local CLI | Anything that serves other people's data |

The defensible asset is the corpus and the name, neither of which the engine's
license exposes. Someone can fork the engine; they cannot fork the accumulated
record of what broke across which torch version on which silicon, because that
is collected, not written.

**Rule of thumb for new code:** if it would run on a user's own machine, it
belongs here. If it would run on ours, it does not.

## Contributor License Agreement

**Every contributor must sign the CLA before a pull request is merged.** This is
enforced from the first outside contribution.

Why, plainly: Apache-2.0 without a CLA is a one-way door. If contributed code
arrives under Apache-2.0 alone, we cannot later relicense the engine, dual-license
it, or move any of that code into the commercial component. A CLA granting us
broad rights keeps those options open while contributors keep their own copyright.

The honest cost: some developers decline to sign CLAs on principle, so this
loses us a few contributions. We are accepting that trade while the project is
young and the option value is high. The alternative — a DCO sign-off — proves
provenance but grants us nothing extra, so it does not preserve the option.

## Trademarks

Apache-2.0 §6 grants no trademark rights, and `NOTICE` restates it. Use the code
freely; do not present your own results as certified by this project.

## Working on the engine

```bash
pip install -e ".[dev]"
python -m pytest -q
```

Tests marked `gpu` are skipped without a CUDA device; most of the suite is pure
logic and runs anywhere.

Four rules override convenience, always. They are the product:

1. Re-measure the PyTorch baseline every run. Never cache it.
2. Never return a bare number. `TimingResult` defines no `__float__` on purpose.
3. The tool adjudicates. Submitted code never reports its own result.
4. Never present a measurement or an oracle as stronger than it was. Label the tier.

If a change makes the tool easier to use by weakening one of those, the rule wins.
