"""
Compute estimates for Harmon simulation data.

. su

    Variable |        Obs        Mean    Std. dev.       Min        Max
-------------+---------------------------------------------------------
        maxK |      1,500       1.488    1.672041         -1          4
    horizon0 |      1,250       .0008    .0017861          0   .0047847
    horizon1 |      1,250       .0016      .00264          0   .0059524
    horizon2 |      1,250       .0024    .0036467          0   .0079365
    horizon3 |      1,250       .0032    .0052799          0   .0119048
-------------+---------------------------------------------------------
    horizon4 |      1,250        .004    .0099179          0   .0285714
     average |      1,250    .0018444    .0023147          0   .0080386


Out[92]:
shape: (9, 6)
┌────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│ statistic  ┆ horizon0  ┆ horizon1  ┆ horizon2  ┆ horizon3  ┆ horizon4  ┆ average   │
│ ---        ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       ┆ ---       │
│ str        ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       ┆ f64       │
╞════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╡
│ count      ┆ 1250.0    ┆ 1250.0    ┆ 1250.0    ┆ 1250.0    ┆ 1250.0    ┆ 1250.0    │
│ null_count ┆ 0.0       ┆ 0.0       ┆ 0.0       ┆ 0.0       ┆ 0.0       ┆ 0.0       │
│ mean       ┆ 0.0008    ┆ 0.0016    ┆ 0.0024    ┆ 0.0032    ┆ 0.004     ┆ 0.0018444 │
│ std        ┆ 0.0017861 ┆ 0.00264   ┆ 0.0036467 ┆ 0.0052799 ┆ 0.0099179 ┆ 0.0023147 │
│ min        ┆ 0.0       ┆ 0.0       ┆ 0.0       ┆ 0.0       ┆ 0.0       ┆ 0.0       │
│ 25%        ┆ 0.0       ┆ 0.0       ┆ 0.0       ┆ 0.0       ┆ 0.0       ┆ 0.0       │
│ 50%        ┆ 0.0       ┆ 0.0       ┆ 0.0       ┆ 0.0       ┆ 0.0       ┆ 0.0       │
│ 75%        ┆ 0.0       ┆ 0.0059524 ┆ 0.0079365 ┆ 0.0119048 ┆ 0.0       ┆ 0.0032154 │
│ max        ┆ 0.0047847 ┆ 0.0059524 ┆ 0.0079365 ┆ 0.0119048 ┆ 0.0285714 ┆ 0.0080386 │
└────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘



"""

import itertools as it
import operator

import numpy as np
import polars as pl

import did_sw
from did_sw import comparison

cols = [
    "horizon0",
    "horizon1",
    "horizon2",
    "horizon3",
    "horizon4",
    "iwtr_s",
    "a2w",
    "average",
]


df = (
    pl.read_csv("data/harmon-sim.csv")
    .rename({"Ei": "E", "i": "id"})
    .drop("X1", "X2", "clust")
)


pl.Config(tbl_rows=30, tbl_cols=15)


base = df.sort("id", "t").with_columns(
    iwtr=pl.lit(1),
    dY=pl.col("Y").diff().over("id"),
    maxK=pl.col("K").max().over("id"),
)
subset = base.pipe(did_sw.assign_weights_horizon).pipe(did_sw.assign_weights_agg)
maxmaxK = int(df["K"].max())
subset_agg = subset.with_columns(
    iwtr_s=pl.col("iwtr").filter(pl.col("K").ge(0)).sum()
).with_columns(
    #  NOTE: outdated; see did_sw
    average=sum(
        [
            pl.col(f"horizon{h}")
            * pl.col("iwtr").filter(pl.col("K").eq(h)).sum()
            / pl.col("iwtr_s")
            for h in range(0, maxmaxK + 1)
        ]
    ),
)

ss = subset_agg.select(
    [
        "id",
        "t",
        "E",
        "K",
        "dY",
        "horizon0",
        "horizon1",
        "horizon2",
        "horizon3",
        "horizon4",
        "average",
    ]
)


df_ext = pl.read_csv("data/harmon-sim-ext.csv")

print(
    ss.filter(pl.col("dY").is_not_null())
    .select(["horizon0", "horizon1", "horizon2", "horizon3", "horizon4", "average"])
    .describe()
    .with_columns(pl.selectors.numeric().round(7))
)

print(
    df_ext.select(
        ["K", "horizon0", "horizon1", "horizon2", "horizon3", "horizon4"]
    ).tail()
)
print(ss.select(["horizon0", "horizon1", "horizon2", "horizon3", "horizon4"]).tail())

print(
    ss.select(["horizon0", "horizon1", "horizon2", "horizon3", "horizon4"]).describe()
)


ests = comparison.compare_estimators(df)
ss2 = (
    subset_agg.rename({"K": "h"})
    .join(
        ests,
        how="left",
        on=["E", "h", "id"],
    )
    .select(["E", "h", "id", "iwtr_s", "average", "swdd"])
    .sort("id")
)
print("Does not equal the average of tes?")
# tau_average |   1.126826
print(ss2.select(pl.col("swdd").mul(pl.col("average")).sum()))
print(ss2.select(pl.col("swdd").mul(pl.col("iwtr_s").pow(-1.0)).sum()))
print(
    did_sw.estimate(subset_agg).select(pl.col("average").mul(pl.col("adj_dY"))).sum(),
    did_sw.estimate(subset_agg).select(pl.col("average").mul(pl.col("adj_dY"))).sum(),
)

#  NOTE: export data; estimate in stata here:
#  /home/jsr-p/gh-repos/did-repos/did_stepwise/scripts/ex4.do
subset_agg.with_columns(avg2_w=pl.col("iwtr_s").pow(-1.0)).filter(
    pl.col("dY").is_not_null()
).write_csv("data/harmon-sim-averages.csv")
