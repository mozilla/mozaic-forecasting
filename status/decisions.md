# Decisions — mozaic-forecasting-official (branch configurable-model-params)

Reconstructed 2026-10-01 from commit history, code comments, research notes and the
author's session notes. Dates are the commit or note dates, not necessarily the day the
call was made.

| id | date | stage | decision | reason |
|---|---|---|---|---|
| D-001 | 2026-05-13 | Configurable model | Expose model settings through a config dataclass plus factory functions, keeping every default equal to the old hard-coded value | A default run must stay identical to the pre-branch library while the pipeline gains programmatic sweeps |
| D-002 | 2026-06-30 | Iran shutdown | Abandon holiday-masking of the 87-day gap in favour of splicing a synthetic counterfactual into history | Validation on real data showed mobile's weekly shape destabilises under the detrender's marginally stable recurrence |
| D-003 | 2026-06-30 | Iran shutdown | Train on the synthetic fill but keep real history for reported actuals, via a separate fit-history field | Never overload a real-data field with synthetic values; actuals must show the real crater |
| D-004 | 2026-06-30 | Iran shutdown | Ship the Iran fill inside the package as data and apply it automatically in populate_tiles | A gap fill is a property of the country's history, like its holiday calendar, not a per-run artefact |
| D-005 | 2026-07-02 | Holiday skips | Skip holidays per (country, year) rather than via a permanent per-holiday flag or a global detrender change | Broad fixes moved roughly an eighth of desktop days in every country; the problem is confined to Iran 2026 |
| D-006 | 2026-07-10 | Seasonality knobs | Couple a forced seasonality regime to growth type on desktop (additive to logistic, multiplicative to linear); on mobile the regime sets seasonality mode only | Stay within the two mode-and-growth quadrants the desktop model has actually been run in |
| D-007 | 2026-07-29 | Seasonality knobs | Expose the desktop auto-switch correlation threshold as a continuous dial, rejected on mobile with an error | Mobile's switch is volume-driven, so the dial would silently do nothing there |
| D-008 | 2026-10-01 | Branch strategy | Keep the branch unmerged from `main` for now (Brendan, confirmed in session) | The changes are hoped to be temporary, and a merge would need a formal code review there has not been time for |
