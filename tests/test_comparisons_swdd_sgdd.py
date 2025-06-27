import numpy as np
import polars as pl

import did_sw
import did_imp
from did_sw import comparison, sim


# Data for tests
np.random.seed(123)
base = sim.simulate_data(
    N=250,
    E_is=[2, 3, 4, 5, 6, -99],
    cgroup=-99,
    periods=list(range(1, 6 + 1)),
)
df = base.with_columns(
    dY=pl.col("Y").diff().over("id"),
    iwtr=pl.lit(1),
    maxK=pl.col("K").max().over("id"),
)
ests = comparison.compare_estimators(base)


def test_equality_aggregate_and_weight():
    """
    Tests equality of estimates manually calculated and from swdd estimator.
    """
    data = (
        df.drop_nulls("dY")
        .pipe(did_sw.assign_weights_horizon)
        .pipe(did_sw.assign_weights_agg)
    )
    r = did_imp.estimate(
        data,
        outcome="dY",
        group="E",
        time="t",
        cluster_var="id",
        unit="id",
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

    # Assert dynamic (horizon) estimates are equal
    ests_model = (
        r.estimates.filter(
            pl.col("term").is_in(
                ["horizon0", "horizon1", "horizon2", "horizon3", "horizon4"]
            )
        )
        .select("estimate")
        .to_numpy()
        .flatten()
    )

    ests_manual = (
        comparison.aggregate(ests, agg="dynamic")
        .sort("h")
        .select("swdd")
        .to_numpy()
        .squeeze()
    )

    assert np.allclose(ests_model, ests_manual), "Should be equal"

    # Assert aggregate estimates are equal
    est_man = comparison.aggregate(ests, agg="total").select("swdd").item()
    est_mod = (
        r.estimates.filter(pl.col("term").is_in(["average"])).select("estimate").item()
    )
    assert np.allclose(est_man, est_mod)


def test_manual_and_swdd():
    """
    Tests manual calculation of swdd and estimates from swdd estimator
    are equal.
    """

    swdd = (
        df.pipe(
            did_imp.prep_data,
            did_imp.DidImpParams(group="E", time="t", unit="id", outcome="dY"),
        )
        .pipe(
            lambda df: did_imp.compute_tes(
                df,
                form=did_imp.Form(outcome="dY", fes="t", xform="0"),
            ).data,
        )
        .select("id", "E", "K", "dY", "Yhat", "Yadj")
        .rename({"K": "h"})
        .with_columns(
            pl.col("Yadj").mul(pl.col("h").ge(0)).cum_sum().over("id").alias("gih")
        )
    )

    res = ests.join(swdd, on=["id", "E", "h"], how="inner").select(
        "id", "E", "h", "swdd", "gih"
    )

    assert np.allclose(
        res["swdd"].to_numpy(),
        res["gih"].to_numpy(),
    ), "Should be equal"


def test_manual_and_sgdd():
    """
    Tests manual calculation of sgdd and estimates from sgdd estimator
    are equal.
    """

    # Construct horizon `h` difference outcomes
    data = df.with_columns(
        [pl.col("Y").diff(n=(i + 1)).over("id").alias(f"dY{i}") for i in range(0, 5)]
    )
    Kmax = df.select("K").max().item()

    for h in range(0, Kmax + 1):
        # Select horizon `h` treatment effects
        form = did_imp.Form(outcome=f"dY{h}", xform="0", fes="t")
        subset = data.drop_nulls(f"dY{h}")
        tes = did_imp.compute_tes(subset, form).data
        sgdd = (
            tes.select("id", "E", "K", "dY", "Yhat", "Yadj")
            .filter(pl.col("K").eq(h))
            .rename({"K": "h"})
        )
        res = ests.join(
            sgdd,
            on=["id", "E", "h"],
            # Pick out the right horizons
            how="inner",
        ).select("id", "E", "h", "sgdd", "Yadj")

        # Test that manually calculated and horizon h effects are equal
        assert np.allclose(
            res["sgdd"].to_numpy(),
            res["Yadj"].to_numpy(),
        ), "Should be equal"


def test_equality_single_treatdate():
    """SWDD and SGDD equal for single treatment date E"""
    base = sim.simulate_data(N=250, rho=1, E_is=[4, -99], cgroup=-99)
    swdd, sgdd = (
        comparison.compare_estimators(base)
        .pipe(comparison.aggregate, agg="dynamic")
        .pipe(comparison.get_comparison_arrays)
    )
    assert np.allclose(swdd, sgdd), "Should be equal"


def test_comparison_groups():
    """
    Test comparison groups of simulated data.
    In particular edge case of last treated group E_i = 6.
    Also what horizon effects are identified.
    """

    comps = comparison.comparisons(df)
    for subset, name in zip(
        [comps.sgdd, comps.swdd],
        ["sgdd", "swdd"],
    ):
        cgroup_pids = (
            (ss1 := subset.filter(pl.col("E").eq(6)))
            .select(pl.col(f"C_{name.upper()}"))
            .item()
        )
        assert (
            df.filter(pl.col("id").is_in(cgroup_pids))
            .select(pl.col("E").eq(-99).all())
            .item()
        )

        assert ss1.shape[0] == 1  # Single horizon identified only
        assert ss1["h"].item() == 0

        # Horizons
        assert subset.filter(pl.col("E").eq(2))["h"].to_list() == [0, 1, 2, 3, 4]
        assert subset.filter(pl.col("E").eq(3))["h"].to_list() == [0, 1, 2, 3]
        assert subset.filter(pl.col("E").eq(4))["h"].to_list() == [0, 1, 2]
        assert subset.filter(pl.col("E").eq(5))["h"].to_list() == [0, 1]
        assert subset.filter(pl.col("E").eq(6))["h"].to_list() == [0]

        # No comparison group for E = 6 when filtering out the never treated
        comps_f = comparison.comparisons(df.filter(pl.col("E").ge(0)))
        assert comps_f.swdd.filter(pl.col("E").eq(6)).is_empty()
        assert comps_f.sgdd.filter(pl.col("E").eq(6)).is_empty()


def test_comparison_groups_other():
    """ """

    comps = comparison.comparisons(df)

    assert comps.query_comparisons(E=5, h=0, estimator="sgdd")[
        "E"
    ].unique().sort().to_list() == [-99, 6]

    assert comps.query_comparisons(E=2, h=0, estimator="sgdd")[
        "E"
    ].unique().sort().to_list() == [-99, 3, 4, 5, 6]
