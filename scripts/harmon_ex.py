"""
Code to illustrate SWDD vs SGDD Harmon example.
See online appendix section D.

- https://web.econ.ku.dk/nharmon/docs/harmon2024onlineappendix.pdf

---

The two estimators thus differ only when the set of observed untreated units
changes between period.
Can happen either:
- Staggered adoption: untreated unit at $E_{i} - 1$ that becomes
    treated between $E_{i} - 1$ and $E_{i} + h$
- Allowing for unbalanced panels and missing data: untreated
    observation enters or leaves the sample between $E_{i} - 1$ and
    $E_{i} + h$

- The example in the paper focuses on γ_{i, h} for (i, h) = (A, 1) i.e. the
estimand γ_{A, 1}.
- The units are treated in the periods of `period_map`.
- Unit D is missing for t >= 3

I also compute treatment effects and final comparisons (not done in paper).
"""

import itertools as it

import polars as pl

from did_sw import comparison

NOT_TREATED = 0

period_map = {
    "A": 2,
    "B": NOT_TREATED,
    "C": 3,
    "D": NOT_TREATED,
}
te_map = {
    "A": 1,
    "B": 0,
    "C": 3,
    "D": 0,
}
fe_map = {
    "A": 3,
    "B": 1,
    "C": 5,
    "D": 1,
}

data = (
    pl.DataFrame(it.product(list("ABCD"), range(1, 8)), schema=["id", "t"])
    .with_columns(
        # E_{i} year of treatment
        E=pl.col("id").replace_strict(period_map, return_dtype=pl.Int8),
        te=pl.col("id").replace_strict(te_map, return_dtype=pl.Int8),
        fe=pl.col("id").replace_strict(fe_map, return_dtype=pl.Int8),
    )
    .with_columns(
        K=pl.col("t").sub(pl.col("E")),
        # Treatment indicator D_{i, t}; treatment is absorbing
        D=pl.col("t").ge(pl.col("E")).and_(pl.col("E").ne(NOT_TREATED)).cast(pl.Int8),
    )
    .with_columns(
        Y=pl.col("t").truediv(2) + pl.col("te").mul(pl.col("K").ge(0)) + pl.col("fe"),
    )
    .with_columns(
        # Unit D is missing for t >= 3 (has consequences for the comparisons)
        D=pl.when(pl.col("id").eq("D").and_(pl.col("t").ge(3)))
        .then(None)
        .otherwise(pl.col("D")),
        K=pl.when(pl.col("E").eq(NOT_TREATED)).then(None).otherwise(pl.col("K")),
    )
)


print(
    "Dataset as shown in Harmon table:",
    data.pivot(on="t", index="id", values="D").fill_null("."),
    sep="\n",
)

pl.Config(tbl_rows=30)

print("Control groups for each (E, h) i.e. γ_{i, h}:")
comparisons = comparison.comparisons(data, horizon=7)
print(comparisons.swdd, comparisons.sgdd, sep="\n")


print("Control group outcomes each (E, h):")
c_outcomes = comparison.comparisons_outcomes(comparisons)
print(c_outcomes.y_sgdd, c_outcomes.y_swdd, sep="\n")
print(c_outcomes.g_sgdd, c_outcomes.g_swdd, sep="\n")

print("Estimates of γ_{E, h} for each (E, h):")
estimators = comparison.estimators(data, c_outcomes)
print(estimators.swdd, estimators.sgdd, sep="\n")

print(comparison.describe_ests(estimators))


# Overviews
comparisons.overview()
comparisons.overview_time()
