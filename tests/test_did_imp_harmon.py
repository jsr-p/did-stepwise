"""
Some tests done iteratively to match R output and Stata output from Harmon.

TODO:
- Move into Justfile or sh script to replicate all the test data generation
"""

import re

import numpy as np
import polars as pl
import pytest

import did_sw
import did_imp
from did_imp import utils

# Hsim
base = (
    (
        pl.read_csv("data/harmon-sim.csv")
        .rename({"Ei": "E", "i": "id"})
        .sort("id", "t")
        .with_columns(
            iwtr=pl.lit(1),
            dY=pl.col("Y").diff().over("id"),
            maxK=pl.col("K").max().over("id"),
            E=pl.when(pl.col("E").eq(7)).then(0).otherwise(pl.col("E")),
        )
        .drop_nulls()
    )
    .pipe(did_sw.assign_weights_horizon)
    .pipe(did_sw.assign_weights_agg)
)

"""
Stata:
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

Own:
shape: (6, 7)
┌──────────┬──────────┬──────────┬───────────┬──────┬──────────┬──────────┐
│ term     ┆ estimate ┆ se       ┆ tstat     ┆ pval ┆ lower    ┆ upper    │
│ ---      ┆ ---      ┆ ---      ┆ ---       ┆ ---  ┆ ---      ┆ ---      │
│ str      ┆ f64      ┆ f64      ┆ f64       ┆ f64  ┆ f64      ┆ f64      │
╞══════════╪══════════╪══════════╪═══════════╪══════╪══════════╪══════════╡
│ horizon0 ┆ 0.501008 ┆ 0.053665 ┆ 9.335862  ┆ 0.0  ┆ 0.395825 ┆ 0.606191 │
│ horizon1 ┆ 0.938361 ┆ 0.090709 ┆ 10.344759 ┆ 0.0  ┆ 0.760572 ┆ 1.11615  │
│ horizon2 ┆ 1.44967  ┆ 0.128739 ┆ 11.260515 ┆ 0.0  ┆ 1.197341 ┆ 1.701999 │
│ horizon3 ┆ 1.897607 ┆ 0.183167 ┆ 10.359964 ┆ 0.0  ┆ 1.538599 ┆ 2.256615 │
│ horizon4 ┆ 2.756371 ┆ 0.265588 ┆ 10.378378 ┆ 0.0  ┆ 2.235819 ┆ 3.276923 │
│ average ┆ 1.126826 ┆ 0.092083 ┆ 12.237059 ┆ 0.0  ┆ 0.946343 ┆ 1.307309 │
└──────────┴──────────┴──────────┴───────────┴──────┴──────────┴──────────┘

"""


def test_did_sw_basic():
    test = pl.DataFrame(
        {
            "term": [
                "tau_horizon0",
                "tau_horizon1",
                "tau_horizon2",
                "tau_horizon3",
                "tau_horizon4",
                "tau_average",
            ],
            "estimate": [0.5010075, 0.9383607, 1.44967, 1.897607, 2.756371, 1.126826],
            "se": [0.0536648, 0.0907088, 0.1287393, 0.1831673, 0.2655878, 0.0920831],
        }
    )
    r = did_imp.estimate(
        base,
        outcome="dY",
        group="E",
        time="t",
        unit="id",
        cluster_var="id",
        fes="t",
        weights=[
            "horizon0",
            "horizon1",
            "horizon2",
            "horizon3",
            "horizon4",
            "average",
        ],
        horizons=None,
    )
    print(r)

    estimate, se = utils.pull_arrays_from_res(
        r, filter=pl.col("term").str.contains("horizon|average")
    )
    est_t, se_t = utils.pull_arrays(test)
    assert np.allclose(estimate, est_t)
    assert np.allclose(se, se_t)


"""
With clustering:
                                                         Number of obs = 1,250
------------------------------------------------------------------------------
           Y | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
tau_horizon0 |   .5010075    .059125     8.47   0.000     .3851247    .6168904
tau_horizon1 |   .9383607   .0807969    11.61   0.000     .7800017     1.09672
tau_horizon2 |    1.44967   .1147704    12.63   0.000     1.224724    1.674616
tau_horizon3 |   1.897607   .1582045    11.99   0.000     1.587532    2.207682
tau_horizon4 |   2.756371    .258764    10.65   0.000     2.249203    3.263539
------------------------------------------------------------------------------
"""


def test_did_sw_clustering():
    test = pl.DataFrame(
        {
            "term": [
                "tau_horizon0",
                "tau_horizon1",
                "tau_horizon2",
                "tau_horizon3",
                "tau_horizon4",
            ],
            "estimate": [0.5010075, 0.9383607, 1.44967, 1.897607, 2.756371],
            "se": [0.059125, 0.0807969, 0.1147704, 0.1582045, 0.258764],
        }
    )
    r = did_imp.estimate(
        base,
        outcome="dY",
        group="E",
        time="t",
        unit="id",
        cluster_var="clust",
        fes="t",
        weights=[
            "horizon0",
            "horizon1",
            "horizon2",
            "horizon3",
            "horizon4",
        ],
        horizons=None,
    )
    print(r)

    estimate, se = utils.pull_arrays_from_res(
        r, filter=pl.col("term").str.contains("horizon|average")
    )
    est_t, se_t = utils.pull_arrays(test)
    assert np.allclose(estimate, est_t)
    assert np.allclose(se, se_t)


"""

With covariates:
                                                         Number of obs = 1,250
------------------------------------------------------------------------------
           Y | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
tau_horizon0 |   .4928777    .054642     9.02   0.000     .3857813     .599974
tau_horizon1 |   .9344182   .0915807    10.20   0.000     .7549233    1.113913
tau_horizon2 |    1.43822   .1294776    11.11   0.000     1.184449    1.691992
tau_horizon3 |   1.886186   .1848452    10.20   0.000     1.523896    2.248476
tau_horizon4 |    2.77223   .2764226    10.03   0.000     2.230452    3.314008
------------------------------------------------------------------------------
┌──────────┬──────────┬──────────┬───────────┬──────┬──────────┬──────────┐
│ term     ┆ estimate ┆ se       ┆ tstat     ┆ pval ┆ lower    ┆ upper    │
│ ---      ┆ ---      ┆ ---      ┆ ---       ┆ ---  ┆ ---      ┆ ---      │
│ str      ┆ f64      ┆ f64      ┆ f64       ┆ f64  ┆ f64      ┆ f64      │
╞══════════╪══════════╪══════════╪═══════════╪══════╪══════════╪══════════╡
│ horizon0 ┆ 0.492878 ┆ 0.054642 ┆ 9.020127  ┆ 0.0  ┆ 0.385779 ┆ 0.599976 │
│ horizon1 ┆ 0.934418 ┆ 0.091581 ┆ 10.203217 ┆ 0.0  ┆ 0.75492  ┆ 1.113916 │
│ horizon2 ┆ 1.43822  ┆ 0.129478 ┆ 11.107868 ┆ 0.0  ┆ 1.184444 ┆ 1.691996 │
│ horizon3 ┆ 1.886186 ┆ 0.184845 ┆ 10.20414  ┆ 0.0  ┆ 1.52389  ┆ 2.248483 │
│ horizon4 ┆ 2.77223  ┆ 0.276423 ┆ 10.028957 ┆ 0.0  ┆ 2.230442 ┆ 3.314018 │
└──────────┴──────────┴──────────┴───────────┴──────┴──────────┴──────────┘

"""


def test_did_sw_concovariates():
    test = pl.DataFrame(
        {
            "term": [
                "tau_horizon0",
                "tau_horizon1",
                "tau_horizon2",
                "tau_horizon3",
                "tau_horizon4",
            ],
            "estimate": [0.4928777, 0.9344182, 1.43822, 1.886186, 2.77223],
            "se": [0.054642, 0.0915807, 0.1294776, 0.1848452, 0.2764226],
        }
    )
    r = did_imp.estimate(
        base,
        outcome="dY",
        group="E",
        time="t",
        unit="id",
        cluster_var="id",
        covariates=["-1 + X1 : C(t)"],
        fes="t",
        weights=[
            "horizon0",
            "horizon1",
            "horizon2",
            "horizon3",
            "horizon4",
        ],
        horizons=None,
    )
    print(r)

    estimate, se = utils.pull_arrays_from_res(
        r, filter=pl.col("term").str.contains("horizon|average")
    )
    est_t, se_t = utils.pull_arrays(test)
    assert np.allclose(estimate, est_t)
    assert np.allclose(se, se_t)


"""
Catcov:
                                                         Number of obs = 1,250
------------------------------------------------------------------------------
           Y | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
tau_horizon0 |    .505401    .057061     8.86   0.000     .3935636    .6172385
tau_horizon1 |   .9372514    .094178     9.95   0.000     .7526659    1.121837
tau_horizon2 |   1.446174   .1308921    11.05   0.000      1.18963    1.702717
tau_horizon3 |   1.873075   .1876279     9.98   0.000     1.505331    2.240819
tau_horizon4 |   2.773142   .2891305     9.59   0.000     2.206456    3.339827
------------------------------------------------------------------------------
┌──────────┬──────────┬──────────┬───────────┬──────┬──────────┬──────────┐
│ term     ┆ estimate ┆ se       ┆ tstat     ┆ pval ┆ lower    ┆ upper    │
│ ---      ┆ ---      ┆ ---      ┆ ---       ┆ ---  ┆ ---      ┆ ---      │
│ str      ┆ f64      ┆ f64      ┆ f64       ┆ f64  ┆ f64      ┆ f64      │
╞══════════╪══════════╪══════════╪═══════════╪══════╪══════════╪══════════╡
│ horizon0 ┆ 0.505401 ┆ 0.057061 ┆ 8.857207  ┆ 0.0  ┆ 0.393562 ┆ 0.617241 │
│ horizon1 ┆ 0.937251 ┆ 0.094178 ┆ 9.951911  ┆ 0.0  ┆ 0.752663 ┆ 1.12184  │
│ horizon2 ┆ 1.446173 ┆ 0.130892 ┆ 11.048591 ┆ 0.0  ┆ 1.189625 ┆ 1.702722 │
│ horizon3 ┆ 1.873075 ┆ 0.187628 ┆ 9.982922  ┆ 0.0  ┆ 1.505324 ┆ 2.240826 │
│ horizon4 ┆ 2.773142 ┆ 0.289131 ┆ 9.591315  ┆ 0.0  ┆ 2.206446 ┆ 3.339837 │
└──────────┴──────────┴──────────┴───────────┴──────┴──────────┴──────────┘

"""


def test_did_sw_catcovariates():
    test = pl.DataFrame(
        {
            "term": [
                "tau_horizon0",
                "tau_horizon1",
                "tau_horizon2",
                "tau_horizon3",
                "tau_horizon4",
            ],
            "estimate": [0.505401, 0.9372514, 1.446174, 1.873075, 2.773142],
            "se": [0.057061, 0.094178, 0.1308921, 0.1876279, 0.2891305],
        }
    )
    r = did_imp.estimate(
        base,
        outcome="dY",
        group="E",
        time="t",
        unit="id",
        cluster_var="id",
        covariates=["C(X2) : C(t)"],
        fes="t",
        weights=[
            "horizon0",
            "horizon1",
            "horizon2",
            "horizon3",
            "horizon4",
        ],
        horizons=None,
    )
    print("Time x X_i")
    print(r)
    print(r.names)

    estimate, se = utils.pull_arrays_from_res(
        r, filter=pl.col("term").str.contains("horizon|average")
    )
    est_t, se_t = utils.pull_arrays(test)
    assert np.allclose(estimate, est_t)
    assert np.allclose(se, se_t)


"""
Contcov & Catcov:
                                                         Number of obs = 1,250
------------------------------------------------------------------------------
           Y | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
tau_horizon0 |   .5080449    .057189     8.88   0.000     .3959565    .6201332
tau_horizon1 |   .9401892   .0948279     9.91   0.000       .75433    1.126048
tau_horizon2 |   1.446463   .1329547    10.88   0.000     1.185876    1.707049
tau_horizon3 |   1.865229   .1890869     9.86   0.000     1.494625    2.235832
tau_horizon4 |   2.765054    .288656     9.58   0.000     2.199298    3.330809
------------------------------------------------------------------------------
┌──────────┬──────────┬──────────┬──────────┬──────┬──────────┬──────────┐
│ term     ┆ estimate ┆ se       ┆ tstat    ┆ pval ┆ lower    ┆ upper    │
│ ---      ┆ ---      ┆ ---      ┆ ---      ┆ ---  ┆ ---      ┆ ---      │
│ str      ┆ f64      ┆ f64      ┆ f64      ┆ f64  ┆ f64      ┆ f64      │
╞══════════╪══════════╪══════════╪══════════╪══════╪══════════╪══════════╡
│ horizon0 ┆ 0.508045 ┆ 0.057189 ┆ 8.883612 ┆ 0.0  ┆ 0.395954 ┆ 0.620135 │
│ horizon1 ┆ 0.940189 ┆ 0.094828 ┆ 9.91469  ┆ 0.0  ┆ 0.754327 ┆ 1.126052 │
│ horizon2 ┆ 1.446462 ┆ 0.132955 ┆ 10.87936 ┆ 0.0  ┆ 1.185871 ┆ 1.707054 │
│ horizon3 ┆ 1.865228 ┆ 0.189087 ┆ 9.864393 ┆ 0.0  ┆ 1.494618 ┆ 2.235839 │
│ horizon4 ┆ 2.765054 ┆ 0.288656 ┆ 9.579064 ┆ 0.0  ┆ 2.199288 ┆ 3.330819 │
└──────────┴──────────┴──────────┴──────────┴──────┴──────────┴──────────┘

"""


def test_did_sw_catconcovariates():
    test = pl.DataFrame(
        {
            "term": [
                "tau_horizon0",
                "tau_horizon1",
                "tau_horizon2",
                "tau_horizon3",
                "tau_horizon4",
            ],
            "estimate": [0.5080449, 0.9401892, 1.446463, 1.865229, 2.765054],
            "se": [0.057189, 0.0948279, 0.1329547, 0.1890869, 0.288656],
        }
    )
    r = did_imp.estimate(
        base,
        outcome="dY",
        group="E",
        time="t",
        unit="id",
        cluster_var="id",
        covariates=["X1 : C(t) + C(X2) : C(t)"],
        fes=None,
        weights=[
            "horizon0",
            "horizon1",
            "horizon2",
            "horizon3",
            "horizon4",
        ],
        horizons=None,
    )
    print("Time x X_i")
    print(r)
    print(r.names)

    estimate, se = utils.pull_arrays_from_res(
        r, filter=pl.col("term").str.contains("horizon|average")
    )
    est_t, se_t = utils.pull_arrays(test)
    assert np.allclose(estimate, est_t)
    assert np.allclose(se, se_t)
