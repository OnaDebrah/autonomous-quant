# Exercises: break a control and watch the benchmark report it

The tests in `tests/` are the book's checks. The first exercise is to break each control on purpose and see which
check catches it. If none does, the check is missing, and that is the more important finding.

1. **Forget the failures** (Chapter 13). In `cli.py`, pass `family_size=len(rows)` to `deflated_rule` instead of the
   ledger's count, then make a few tries raise an exception. The bar falls. Which test notices?
2. **Look ahead** (Chapter 2). Change `TrendRule.positions` to use `prices.shift(-1)`. The truncation test should fail.
   Then hide the look-ahead behind a rolling window that is centred (`center=True`): does it still?
3. **Fill at the close** (Chapter 4). In `Simulator.account`, use `target.iloc[t]` instead of `target.iloc[t - 1]`.
   Which invariant test fails, and how much better do the demo's results look?
4. **Count the search as one** (Chapter 13). Replace `deflated` with `single_test` in the contract. Run
   `aq benchmark` and read the false-pass rate.
5. **Trust a revised history** (Chapter 3). Load Yahoo data with `assume_available_at_observation=True` and run the
   truncation test. Why can it not tell you whether the result is adapted?
6. **Correlated tries** (Chapter 14). Add twenty near-identical parameter values to `TrendRule.region` and compare
   the deflated rule's verdict with the sign-flip test's as the tries become more alike.
7. **Plant a different truth** (Chapter 12). Change `World` so that edges last a few days and then vanish. What
   happens to power, and does any judge notice?
