# The book's simulation programs

One program per chapter, `chNN_*.py`, each seeded, each writing `chNN_results.json` beside itself. These are the
numbers in the book's tables and empirical sections; the results files are included so that a reader can check a
number without running anything, and can regenerate it exactly by running the program:

```bash
python book/ch13_cost_of_search.py      # rewrites book/ch13_results.json, identical to the shipped file
```

`figures_data.py` writes the data behind the book's plots to `book/figdata/`. Where a plot shows a simulated path, it
reruns the chapter's own program with the same seed and the same order of random draws, and stops unless the rerun
reproduces the chapter's saved results. Chapter 17's bandit is a pure-Python loop; the whole run takes about twenty
minutes.

The programs need numpy, pandas and scipy. They are separate from the package in `src/`, which is written for reading
and extending rather than for reproducing the book.
