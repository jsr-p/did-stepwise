"""
Compute estimates for Harmon simulation data.

Output from Harmon script:
```txt
Running did_imputation on first-differenced data with appropriate weights:
(did_imputation due to Borusyak, Jaravel, and Spiess (2023))


                                                         Number of obs = 1,250
------------------------------------------------------------------------------
           Y | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
tau_horizon0 |   .5010075   .0536648     9.34   0.000     .3958264    .6061887
tau_horizon1 |   .9383607   .0907088    10.34   0.000     .7605747    1.116147
tau_horizon2 |    1.44967   .1287393    11.26   0.000     1.197346    1.701994
tau_horizon3 |   1.897607   .1831673    10.36   0.000     1.538605    2.256608
tau_horizon4 |   2.756371   .2655878    10.38   0.000     2.235828    3.276913
 tau_average |   1.126826   .0920831    12.24   0.000     .9463462    1.307305
------------------------------------------------------------------------------

```bash
❯ python scripts/hsim.py
shape: (5, 3)
┌─────┬──────────┬──────────┐
│ h   ┆ swdd     ┆ sgdd     │
│ --- ┆ ---      ┆ ---      │
│ i64 ┆ f64      ┆ f64      │
╞═════╪══════════╪══════════╡
│ 0   ┆ 0.501008 ┆ 0.501008 │
│ 1   ┆ 0.938361 ┆ 0.953403 │
│ 2   ┆ 1.44967  ┆ 1.447149 │
│ 3   ┆ 1.897607 ┆ 1.892599 │
│ 4   ┆ 2.756371 ┆ 2.695074 │
└─────┴──────────┴──────────┘
shape: (1, 2)
┌──────────┬──────────┐
│ swdd     ┆ sgdd     │
│ ---      ┆ ---      │
│ f64      ┆ f64      │
╞══════════╪══════════╡
│ 1.126826 ┆ 1.126252 │
└──────────┴──────────┘
```

"""

import numpy as np
import polars as pl

from did_sw import comparison

df = pl.read_csv("data/harmon-sim.csv").rename({"Ei": "E", "i": "id"})
df.columns

ests = comparison.compare_estimators(df)


comparisons = comparison.comparisons(df)
c_outcomes = comparison.comparisons_outcomes(comparisons)
estimators = comparison.estimators(df, c_outcomes)

c_outcomes.y_swdd

comparisons.swdd

print(comparison.aggregate(ests, "dynamic").sort("h"))
print(comparison.aggregate(ests, "total"))
# print(comparison.aggregate(ests, "group"))

ests.mean()
