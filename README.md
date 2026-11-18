# Autonomous Quant

The seed code of *The Autonomous Quant: The Mathematics of Autonomous Quantitative Research*.

A minimal research loop to read in an afternoon, run in five minutes and extend one interface at a time. Each module
follows a chapter of the book and its pseudocode. It is a starting point, not a product: it contains toy, beginner rules with no
claimed edge, no market data and no trading connection, and it makes no claim to find anything. Its value is that its
controls can be shown to work, on worlds where the truth is planted.

## Quick start

```bash
pip install -e ".[dev]"          # add ,yahoo for the Yahoo Finance adapter
pytest                           # the book's checks, including planted truth
aq benchmark                     # score three judges on planted truth (Chapter 12)
aq demo                          # search two toy rules on a simulated market, register every try, judge the family
aq demo --drift 0.003            # the same, in a market with trends
aq demo --source yahoo --symbol SPY
aq run                           # the research loop on a simulated market of eight assets, then its maturity scorecard
aq crash-test                    # kill the loop repeatedly; it must end exactly where an uninterrupted run ends
```

`aq benchmark` prints the scorecard every research system should be judged by (Chapter 12): the false-pass rate, the
share of all families whose passed try has no edge, and the power, the share of families with an edge that pass the
try that has it. With twenty tries on two years of data the deflated rule makes a false pass in about 0.4% of families
and finds about 8% of real edges with a Sharpe ratio of 1.5: honest, and nearly blind on short records (Chapter 13).

`aq demo` registers each try in the ledger before it runs, evaluates it in the laboratory with next-open fills and
costs, and judges the family with the deflated rule, a contract that fails closed, and the sign-flip test. In the
market with trends the best rule fails the deflated bar but passes the sign-flip test: its eighteen tries are
correlated, and a bar that treats them as independent overstates the luck (Chapter 14).

`aq run` is the whole loop (Chapter 21). Every few days it reads two instruments as of that day, turns the readings
into hypotheses, lets a seeded bandit choose which to study, writes those studies to an outbox, works them under
leases (registering every try before it runs), enrols a pass in a forward test with an immutable binding, settles
forward tests when their windows end, and pays the rewards to the bandit. Each tick recovers first: it finishes any
study already owed and settles any forward test already due, and only then starts new research, so
`aq run --no-research` still scores the forward tests in progress (Algorithm 21.1). A forward test is settled only if
its binding is intact and its data cover the window; otherwise it stays observing, or is recorded as invalidated or
incomplete with its cause (Algorithm 22.2). Its reward is the probabilistic Sharpe ratio of the strategy's returns over
the window, and its registered predicates are checked (Chapter 11): the prediction is supported if the rule made
money over the window, contradicted if it lost money, and not evaluable if too many days were missing. Beating a
falling market while losing money does not count. On the default market most studies are
rejected, and some strong ones fail only the contract's trade-count clause: slow rules make few round trips, which is
Chapter 15's point. The run ends with the maturity scorecard (Chapter 23): on a year and a half of history, the bar
leaves about 9% power for an edge with a Sharpe ratio of 1.5.

With `--source yahoo`, the as-of read is refused: a history downloaded today says nothing about when each value was
first known. The demo continues only after stating the assumption. That refusal is Chapter 3's lesson.

## Run with Docker

Build the image from the repository root. The image runs `aq`, so pass the subcommand and its arguments after the image name.

```bash
docker build -t autonomous-quant .
docker run --rm autonomous-quant                       # show CLI help
docker run --rm autonomous-quant demo                  # run on the simulated market
docker run --rm autonomous-quant benchmark
```

Mount a named volume at `/data` to keep SQLite files between runs. Use `--db` with `run` or `--ledger` with `demo`.

```bash
docker run --rm --mount source=aq-data,target=/data \
  autonomous-quant run --db /data/research.db
docker run --rm --mount source=aq-data,target=/data \
  autonomous-quant demo --ledger /data/ledger.db
```

To include the optional Yahoo Finance adapter, build with the `yahoo` extra. The Yahoo demo requires internet access.

```bash
docker build --build-arg 'AQ_EXTRAS=[yahoo]' -t autonomous-quant:yahoo .
docker run --rm autonomous-quant:yahoo demo --source yahoo --symbol SPY
```

GitHub Actions runs pytest on Python 3.10 through 3.15 and builds the Docker image to check that the simulated demo can write its ledger.

## What is here, and where it is in the book

| Module | What it contains | Chapters |
|---|---|---|
| `data/store.py` | An append-only store of observations with observation and availability times, and an as-of read | 2, 3 |
| `data/sources.py` | A seeded regime-switching market; a Yahoo Finance adapter, with retries, whose availability is unknown | 3, 6, 20 |
| `backtest/lab.py` | A reference simulator: units and cash, next-open fills, costs, cash rate; the truncation test | 2, 4 |
| `backtest/rules.py` | Two toy rules, trend and mean reversion, each with a version and a parameter region | 11 |
| `data/ledger.py` | Registration of a try in its own transaction before evaluation; family counts; fingerprints | 11, 13, 21 |
| `evaluation/judge.py` | The deflated rule, a contract that fails closed, the sign-flip test | 13, 14, 15 |
| `evaluation/planted.py` | Worlds with planted edges and the scorecard of false-pass rate and power | 12, 13 |
| `research/instruments.py` | Two versioned instruments (a window score, a volatility state) and a validation verdict | 5, 6, 10 |
| `research/predicates.py` | Necessary conditions, a mechanism test and a failure condition, checked by code | 11 |
| `research/generator.py` | Readings to hypotheses with controls, predicates and an identity that ignores the source | 11 |
| `research/models.py` | Match, tweak and write, with a rule-based default and validation that drops invalid answers | 18, 19 |
| `workflow/allocator.py` | A seeded Thompson bandit; recorded probabilities; delayed rewards; rejections as zeros | 17 |
| `workflow/forward.py` | Forward tests with immutable bindings, validity checks, and settlement into a reward in [0, 1] | 22 |
| `workflow/loop.py` | An idempotent tick with an outbox, leases and fencing tokens; the crash test | 21 |
| `evaluation/scorecard.py` | The maturity scorecard | 23 |
| `interfaces.py` | The interfaces a new domain implements | Appendix C |
| `errors.py`, `log.py` | The exceptions the package raises, and logging for applications | — |
| `book/` | The simulation programs behind every number and figure in the book | all |

Python module paths in this table are relative to `src/autonomous_quant/`.

### Errors and logging

Everything the package refuses raises a subclass of `AutonomousQuantError`: `ConfigurationError` for a parameter or
specification that cannot be right, `DataError` (and its `UnknownAvailability`) for data that cannot be used as given,
`SourceError` for a data source that failed after its retries, and `IntegrityError` for an attempt to rewrite a record.
Nothing is clipped or repaired silently: a position outside [-1, 1], a price that is not positive, or a model answer
outside the catalogue is refused, and the refusal says why. Inside the loop a study that raises is retried on later
ticks and, after `max_attempts`, recorded as failed with its error; the loop goes on.

The library only creates loggers, under `autonomous_quant`, and prints nothing by default. The command line takes
`--log-level` (INFO shows every decision and its reason; WARNING, the default, shows what was refused or assumed) and
`--log-file`. In your own program, call `autonomous_quant.log.configure("INFO")` or attach your own handlers. `aq`
exits with status 2 on an expected failure, with a one-line message, and 1 on an unexpected one, with the traceback in
the log.

### Extending it to another domain

Three interfaces carry everything specific to markets: an `Instrument` turns stamped observations into readings, a
`Laboratory` turns a hypothesis and data into outcomes, and a `Reward` turns a forward window into a number in [0, 1].
The store, ledger, judge and the parts still to come do not depend on what is being researched.

## Where to go next

Everything in the book's Appendix C table is here in its smallest form. Natural next steps, each a chapter:

* a hidden-state regime model, validated against `VolatilityState` as its baseline (Chapter 6);
* a holdout and a sequential boundary in the contract (Chapter 15);
* a language model behind `ModelCaller`, keeping `validated_match` around it (Chapter 18);
* generated rules through the truncation test, a sandbox and a tournament before a person merges them (Chapter 19);
* a sizing rule as a separately tested study (Chapter 20);
* a new domain: implement `Instrument`, `Laboratory` and `Reward` and keep everything else.

## Links and contact

* The book's page: https://orarese.com/autonomous-quant
* Oraculum, the system the book describes: https://orarese.com

Questions, issues and corrections: open an issue on this repository, or write to contact@orarese.com.

## Licence

MIT. See [LICENSE](LICENSE).
