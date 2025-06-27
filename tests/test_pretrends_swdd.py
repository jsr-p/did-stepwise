"""
Some pretrends testing stuff.
"""

import numpy as np
import polars as pl

import did_sw
import did_imp
from did_imp import utils

basic = pl.read_csv("data/harmon-sim.csv")
cols = [
    "i",
    "t",
    "Ei",
    "K",
    "D",
    "tau",
    "eta",
    "eps",
    "Y",
]


subset = basic.select(cols).with_columns(
    # NOTE: doesn't work if E = 0 for never-treated
    ever_treat=pl.col("t").ge(pl.col("Ei")).max().over("i"),
)

pretrends = 4
cps = subset.filter(
    # Non-treated (D_it = 0); Become treated at some point; max pt + 1 periods
    # before treatment i.e. for pt = 4 we can have 5 pretreat periods.
    pl.col("Ei").sub(pl.col("t")).abs().le(pretrends + 1),
    pl.col("ever_treat").eq(1),
    pl.col("D").eq(0),  # Exclude all treated periods from the analysis sample
)
expanded = (
    pl.concat(
        (
            subset.with_columns(copy=0),
            cps.with_columns(copy=1),
        )
    )
    .with_columns(
        pl.struct("copy", "i").rank(method="dense").alias("i_new"),
        t_new=pl.col("t").mul(-1),  # reverse time
        ei_new=pl.lit(99),
    )
    .with_columns(
        ei_new=pl.when(pl.col("copy").eq(1) & pl.col("ever_treat").eq(1))
        .then(pl.col("Ei").sub(2).mul(-1))
        .otherwise(pl.col("ei_new"))
    )
    .with_columns(K=pl.col("t_new").sub(pl.col("ei_new")))
    .with_columns(maxK=pl.col("K").max().over("i_new"))
    .sort("i_new", "t_new")
)


#  NOTE: K = -1 corresponds to pretrend1; mechanically equal to 0?
expanded.select("K").unique().sort("K")
expanded.group_by("copy", "ever_treat").agg(pl.col("K").unique())

(
    expanded.filter(pl.col("D").ne(1))
    .sort("i_new", "t_new")
    .with_columns(
        iwtr=pl.lit(1),
        dY=pl.col("Y").diff().over("i_new"),
    )
    # .pipe(did_sw.assign_weights_horizon, min_K=-1, id_col="i_new", prefix="pretrend")
    .pipe(did_sw.assign_weights_horizon, min_K=-1, id_col="i_new")
    .pipe(did_sw.rename_horizons)
    .drop_nulls()
)


prepe = (
    expanded.filter(pl.col("D").ne(1))
    .sort("i_new", "t_new")
    .with_columns(
        iwtr=pl.lit(1),
        dY=pl.col("Y").diff().over("i_new"),
    )
    .pipe(did_sw.assign_weights_horizon, min_K=-1, id_col="i_new")
    .pipe(did_sw.rename_horizons)
    .drop_nulls()
    .select(
        "t",
        "i_new",
        "i",
        "t_new",
        "ei_new",
        "D",
        "K",
        "dY",
        "pretrend1",  # Identically equal to 0
        "pretrend2",
        "pretrend3",
        "pretrend4",
        "pretrend5",
    )
    .with_columns(D=pl.col("t_new").ge(pl.col("ei_new")).cast(pl.Int8))
)

"""
                                                         Number of obs = 1,051
-------------------------------------------------------------------------------
            Y | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
--------------+----------------------------------------------------------------
tau_pretrend1 |          0  (omitted)
tau_pretrend2 |  -.0625911   .0385982    -1.62   0.105    -.1382422    .0130601
tau_pretrend3 |   .0762774   .0536906     1.42   0.155    -.0289543    .1815091
tau_pretrend4 |   .0072049   .0784347     0.09   0.927    -.1465243    .1609342
tau_pretrend5 |    .117786   .1442234     0.82   0.414    -.1648867    .4004587
-------------------------------------------------------------------------------
"""


def test_pretrends_omit():
    r = did_imp.estimate(
        prepe,
        outcome="dY",
        group="ei_new",
        time="t",
        cluster_var="i",
        unit="i",
        fes="t",
        weights=[
            "pretrend1",  # Identically equal to 0
            "pretrend2",
            "pretrend3",
            "pretrend4",
            "pretrend5",
        ],
        horizons=None,
        prep=False,
    )
    # Estimating pretrends with -1 included:
    estimate, se = utils.pull_arrays_from_res(
        r, filter=pl.col("term").str.contains("pretrend")
    )
    assert np.allclose(estimate[0], 0)


"""
                                                         Number of obs = 1,051
-------------------------------------------------------------------------------
            Y | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
--------------+----------------------------------------------------------------
tau_pretrend1 |          0  (omitted)
tau_pretrend2 |  -.0625911   .0385982    -1.62   0.105    -.1382422    .0130601
tau_pretrend3 |   .0762774   .0536906     1.42   0.155    -.0289543    .1815091
tau_pretrend4 |   .0072049   .0784347     0.09   0.927    -.1465243    .1609342
tau_pretrend5 |    .117786   .1442234     0.82   0.414    -.1648867    .4004587
-------------------------------------------------------------------------------
"""

results = did_imp.estimate(
    prepe,
    outcome="dY",
    group="ei_new",
    time="t",
    cluster_var="i",
    unit="i",
    fes="t",
    weights=[
        "pretrend2",
        "pretrend3",
        "pretrend4",
        "pretrend5",
    ],
    horizons=None,
    prep=False,
)
print("Estimating pretrends without -1 included:")
print(results)


def test_pretrends():
    test = pl.DataFrame(
        {
            "term": [
                "tau_pretrend2",
                "tau_pretrend3",
                "tau_pretrend4",
                "tau_pretrend5",
            ],
            "estimate": [-0.0625911, 0.0762774, 0.0072049, 0.117786],
            "se": [0.0385982, 0.0536906, 0.0784347, 0.1442234],
        }
    )
    estimate, se = utils.pull_arrays_from_res(
        results, filter=pl.col("term").str.contains("pretrend")
    )
    est_t, se_t = utils.pull_arrays(test)
    assert np.allclose(est_t.round(6), estimate.round(6))  # Equal to 6 decimals
    assert np.allclose(se, se_t)


def test_pretrends_fes():
    r = did_imp.estimate(
        prepe,
        outcome="dY",
        group="ei_new",
        time="t_new",
        unit="i_new",
        cluster_var="i_new",
        fes="t_new",
        weights=[
            "pretrend2",
            "pretrend3",
            "pretrend4",
            "pretrend5",
        ],
        horizons=None,
        prep=False,
    )
    print(
        "Estimating pretrends without -1 and correct fes + ids (se's slightly larger):"
    )
    _, se = utils.pull_arrays_from_res(
        r, filter=pl.col("term").str.contains("pretrend")
    )
    _, se_prev = utils.pull_arrays_from_res(
        results, filter=pl.col("term").str.contains("pretrend")
    )

    assert np.all(se > se_prev)
