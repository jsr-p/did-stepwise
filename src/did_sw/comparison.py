"""
Module for computing comparison groups for SGDD and SWDD estimators in
staggered treatment adoption designs as characterized by Harmon (2024).
"""

from dataclasses import dataclass
from itertools import zip_longest
from typing import Literal

import numpy as np
import polars as pl
from tabulate import tabulate
from tqdm import tqdm


__all__ = [
    "Comparisons",
    "ComparisonsOutcomes",
    "Estimators",
    "compare_controls",
    "cumulative_controls_swdd",
    "compare_estimators",
    "compare_ests",
    "comparisons",
    "comparisons_outcomes",
    "full_comparison",
]


def cumulative_controls_swdd(df: pl.DataFrame):
    """Cumulative union of control groups for the SWDD estimator.

    Does this by using a `join_where` join to get all ids of
    previous horizons for each (E, h) and then concatenating
    the control groups.

    Args:
        df: DataFrame with columns (E, h, C_SWDD)
            (cohort, horizon, control group)
    """
    ss = df.join_where(
        df.select(pl.col("E", "h", "C_SWDD").name.suffix("_r")),
        pl.col("E") == pl.col("E_r"),
        # `<=` important to include h = 0 here.
        pl.col("h_r").le(pl.col("h")),
    ).with_columns(joined=pl.col("C_SWDD").list.concat(pl.col("C_SWDD_r")))
    return (
        ss.drop("C_SWDD", "C_SWDD_r")
        .explode("joined")
        .group_by("E", "h")
        .agg(pl.col("joined").unique().alias("C_SWDD"))
        .sort("E", "h")
    )


def compare_controls(comp: "Comparisons"):
    """
    Compare the number of control groups for SWDD and SGDD estimators.

    """
    swdd = comp.swdd.select("E", "h", "C_SWDD")
    swdd = (
        swdd.sort("E", "h")
        .with_columns(
            pl.col("C_SWDD").list.len().cum_sum().over("E").alias("#controls_all")
        )
        .drop("C_SWDD")
        .join(
            swdd.pipe(cumulative_controls_swdd),
            on=["E", "h"],
            how="left",
        )
        .with_columns(pl.col("C_SWDD").list.len().cast(pl.Int32).alias("#controls"))
        .drop("C_SWDD")
    )
    return swdd.join(
        (
            comp.sgdd.select("E", "h", "C_SGDD").with_columns(
                pl.col("C_SGDD").list.len().cast(pl.Int32).alias("#controls2")
            )
        ).drop("C_SGDD"),
        on=["E", "h"],
        how="full",
        coalesce=True,
    )


@dataclass
class Comparisons:
    comparisons: pl.DataFrame
    sgdd: pl.DataFrame
    swdd: pl.DataFrame
    data: pl.DataFrame  # Original data
    id_col: str = "id"

    def __post_init__(self):
        self.comparisons = self.comparisons.sort("E", "id", "h")
        self.sgdd = self.sgdd.sort("E", "h")
        self.swdd = self.swdd.sort("E", "h")

    def query_comparisons(
        self,
        E: int,
        h: int,
        estimator: Literal["sgdd", "swdd"],
    ):
        """Query comparisons for specific (E, h) of original data."""
        match estimator:
            case "sgdd":
                df = self.sgdd
                c_col = "C_SGDD"
            case "swdd":
                df = self.swdd
                c_col = "C_SWDD"
        subset = df.filter(pl.col("E").eq(E), pl.col("h").eq(h))
        if subset.is_empty():
            raise ValueError(f"No data for E={E}, h={h}")
        c_pids = subset[c_col][0]
        return self.data.filter(pl.col(self.id_col).is_in(c_pids))

    def overview(self):
        """
        - `#ExtraObs` equals the extra observations used in the SWDD estimator
           compared to the SGDD estimator; see also Figure 1 in the appendix of
           Harmon (2024).
        """
        comps = compare_controls(self)
        sw = comps.select("E", "h", "#controls")
        sg = comps.select("E", "h", "#controls2").rename({"#controls2": "#controls"})
        extra_data = comps.select(
            "E",
            pl.col("#controls").sub(pl.col("#controls2")).alias("#ExtraUnits"),
            pl.col("#controls_all").sub(pl.col("#controls2")).alias("#ExtraObs"),
        )
        data_diff = extra_data.select("#ExtraUnits", "#ExtraObs").to_dicts()
        tab1 = tabulate(
            sg.to_dicts(),
            headers="keys",
            tablefmt="plain",
            missingval="-",
        ).split("\n")
        tab2 = tabulate(
            sw.to_dicts(),
            headers="keys",
            tablefmt="plain",
            missingval="-",
        ).split("\n")
        tab3 = tabulate(
            data_diff,
            headers="keys",
            tablefmt="plain",
            missingval="-",
        ).split("\n")

        width1 = max(len(row) for row in tab1)
        width2 = max(len(row) for row in tab2)
        width3 = max(len(row) for row in tab3)

        width = width1 + width2 + width3 + 6  # Adjust spacing
        print(" Cohorts, Horizons & Controls ".center(width, "-"))
        print(
            " SGDD ".center(width1)
            + " | "
            + " SWDD ".center(width2)
            + " | "
            + " Efficiency gains SWDD ".center(width3)
        )

        for row1, row2, row3 in zip_longest(tab1, tab2, tab3, fillvalue=" " * width1):
            print(row1.ljust(width1) + " | " + row2 + " | " + row3)
        print("-" * width)

    def overview_time(self):
        total_cohorts = (
            self.data.group_by("E")
            .agg(pl.col("id").n_unique().alias("N"))
            .sort("E")
            .rename({"E": "Cohort", "N": "#units"})
            .to_dicts()
        )
        total_periods = (
            self.data.group_by("t")
            .agg(pl.col("id").n_unique().alias("N"))
            .sort("t")
            .rename({"t": "Period", "N": "#units"})
            .to_dicts()
        )
        tab1 = tabulate(total_cohorts, headers="keys", tablefmt="plain").split("\n")
        tab2 = tabulate(total_periods, headers="keys", tablefmt="plain").split("\n")
        width1 = max(len(row) for row in tab1)
        width2 = max(len(row) for row in tab2)
        width = width1 + width2 + 5  # Adjust spacing

        print(" Cohorts  & Periods ".center(width, "-"))
        print(" Cohorts ".center(width1) + " | " + " Periods ".center(width2))

        for row1, row2 in zip_longest(tab1, tab2, fillvalue=" " * width1):
            print(row1.ljust(width1) + " | " + row2)
        print("-" * width)


@dataclass
class ComparisonsOutcomes:
    """
    NOTE:
        - SGDD uses the E_i - 1 as baseline difference
        - SWDD uses the E_i + k - 1 as baseline difference for each k

    Returns:
        (y_sgdd, y_swdd): outcomes for each t for comparison groups
        (g_sgdd, g_swdd): aggregated outcomes for each (E, h)
    """

    y_sgdd: pl.DataFrame
    y_swdd: pl.DataFrame
    g_sgdd: pl.DataFrame
    g_swdd: pl.DataFrame


@dataclass
class Estimators:
    swdd: pl.DataFrame
    sgdd: pl.DataFrame


REL_COLS = {"id", "E", "D", "Y", "K", "t"}


def _sgdd_condition(df: pl.DataFrame) -> pl.DataFrame:
    """
    SGDD comparison: D_{i, E_i - 1} = 0 & D_{i, E_i + h} = 0.
    """
    return df.with_columns(
        valid_sgdd=pl.col("h").ge(0) & pl.col("D_1").eq(0) & pl.col("D_h").eq(0)
    )


def _swdd_condition(df: pl.DataFrame) -> pl.DataFrame:
    """
    SWDD comparison: D_{i, E_i + k - 1} = 0 and D_{i, E_i + k} = 0
    for each k = 0, 1, ..., h
    (i.e. both equal to 0 i.e. both observed and non-treated)
    """
    return df.sort("E", "id", "E_h").with_columns(
        valid_swdd=(pl.col("D_h").eq(0) & pl.col("D_h").shift(1).eq(0)).over("id")
    )


def comparisons(
    data: pl.DataFrame,
    horizon: int = 7,
    id_col: str = "id",
) -> Comparisons:
    """
    Compute SGDD and SWDD comparison groups for staggered treatment adoption
    analysis.

    Parameters:
    -----------
    data : pl.DataFrame
        A Polars DataFrame containing the following key columns:
        - `id`  : Identifier for each unit.
        - `E`   : The period in which the unit is first eligible for treatment.
        - `D`   : A binary indicator (0/1) representing treatment status.
        - `Y`   : Outcome variable.
        - `K`   : A relative time indicator for each observation.
        - `t`   : Time period corresponding to the observation.

    """
    if len(sdiff := REL_COLS - set(data.columns)) != 0:
        raise ValueError(f"Missing columns: {sdiff}")

    base_comparison = (
        data.select("E")
        .unique()
        # .filter(pl.col("E").eq(cgroup).not_())
        .with_columns(h=[-1] + list(range(horizon)))
        .explode("h")
    )
    comparisons = (
        # Construct df with unique E_i together with E_i + h and E_i
        base_comparison.with_columns(
            # Add horizons to each E_i to get E_i + h
            E_h=pl.col("E").add(pl.col("h"))
        )
        .join(
            # Merge data on units in period E_i + h
            data.select(id_col, "D", "t", "Y").rename({"D": "D_h"}),
            left_on=["E_h"],
            right_on=["t"],
        )
        .sort("E", "id", "h")
        .with_columns(
            # D_{E_i - 1} and Y_{E_i - 1} for each unit
            D_1=pl.col("D_h").first().over("E", id_col),
            Y_1=pl.col("Y").first().over("E", id_col),
        )
    )
    base_sgdd = comparisons.pipe(_sgdd_condition).filter("valid_sgdd")
    sgdd = (
        base_sgdd.group_by("E", "h")
        .agg(
            # ids in comparison group for (E, h)
            pl.col(id_col).unique().alias("C_SGDD"),
        )
        .sort("E", "h")
    )

    base_swdd = comparisons.pipe(_swdd_condition).filter("valid_swdd").sort("E", "h")

    swdd = (
        base_swdd.filter(pl.col("h").ge(0))
        .group_by("E", "h")
        .agg(
            # ids in comparison group for (E, h)
            pl.col(id_col).unique().sort().alias("C_SWDD"),
        )
        .sort("E", "h")
    )

    return Comparisons(comparisons=comparisons, sgdd=sgdd, swdd=swdd, data=data)


def comparisons_outcomes(comp: Comparisons) -> ComparisonsOutcomes:
    sgdd, swdd, comparisons = comp.sgdd, comp.swdd, comp.comparisons

    # Control groups and outcomes for period E_i + h and E_i - 1 for each
    # unique E_i
    y_sgdd = (
        sgdd.rename({"C_SGDD": "id"})
        .explode("id")
        .join(
            comparisons.select("E", "h", "Y", "Y_1", "id"),
            how="left",
            on=["E", "h", "id"],
        )
        .with_columns(
            dY=pl.col("Y").sub(pl.col("Y_1")),
        )
        .sort("E", "h")
    )
    # Average over control groups
    g_sgdd = (
        y_sgdd.group_by("E", "h")
        .agg(
            # SGDD estimator computes comparison group for each h
            pl.col("dY").mean().alias("Y_C"),
            ids=pl.col("id").unique(),
        )
        .with_columns(
            count=pl.col("ids").list.len(),
        )
        .sort("E", "h")
    )

    y_swdd = (
        swdd.vstack(
            # Add h = -1 for each E to allow for difference for k = 0
            swdd.select(pl.col("E"))
            .unique()
            .with_columns(h=-1, C_SWDD=None)
            .cast({"h": pl.Int64})
        )
        .sort("E", "h")
        .with_columns(C_SWDD=pl.col("C_SWDD").backward_fill().over("E"))
        .rename({"C_SWDD": "id"})
        .explode("id")
        # Merge outcomes for each control
        .join(
            comparisons.select("E", "h", "Y", "id"),
            how="left",
            on=["E", "h", "id"],
        )
        .sort("E", "id", "h")
        .with_columns(
            # Compute Y_{E + k} - Y_{E + k - 1} , k = 0, 1, ..., h
            dY=pl.col("Y").diff().over("E", "id")
        )
        .filter(pl.col("h").ge(0))
        .sort("E", "h")
    )
    # Average over control groups
    g_swdd = (
        y_swdd.group_by("E", "h")
        .agg(
            pl.col("dY").mean().alias("Y_C"),
            ids=pl.col("id").unique(),
        )
        .sort("E", "h")
        .with_columns(
            # Add count of control groups
            Y_C=pl.col("Y_C").cum_sum().over("E"),
            count=pl.col("ids").list.len(),
        )
    )

    return ComparisonsOutcomes(
        y_sgdd=y_sgdd,
        y_swdd=y_swdd,
        g_sgdd=g_sgdd,
        g_swdd=g_swdd,
    )


def estimators(data: pl.DataFrame, c_outcomes: ComparisonsOutcomes):
    g_swdd, g_sgdd = c_outcomes.g_swdd, c_outcomes.g_sgdd

    outcomes = (
        data.select("id", "t", "E", "K", "Y")
        .rename({"K": "h"})
        .sort("id", "t")
        .join(
            # Merge outcome in t - 1 for each unit (if treated)
            data.filter(pl.col("K").eq(-1))
            .select("id", "K", "Y")
            .rename({"Y": "Y_1"})
            .drop("K"),
            how="left",
            on="id",
        )
    )

    est_swdd = outcomes.join(
        g_swdd.drop("ids", "count"), on=["E", "h"], how="inner"
    ).with_columns(ghat=pl.col("Y") - pl.col("Y_1") - pl.col("Y_C"))
    est_sgdd = outcomes.join(
        g_sgdd.drop("ids", "count"), on=["E", "h"], how="inner"
    ).with_columns(ghat=pl.col("Y") - pl.col("Y_1") - pl.col("Y_C"))
    return Estimators(
        swdd=est_swdd,
        sgdd=est_sgdd,
    )


def compare_ests(ests: Estimators):
    return (
        ests.swdd.select("id", "E", "h", "ghat", "Y_C")
        .rename({"ghat": "swdd", "Y_C": "C_swdd"})
        .join(
            ests.sgdd.select("id", "E", "h", "ghat", "Y_C").rename(
                {"ghat": "sgdd", "Y_C": "C_sgdd"}
            ),
            on=["id", "E", "h"],
        )
    )


def compare_estimators(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns df with columns (id, E, h, swdd, sgdd)
    i.e. column with SWDD and SGDD estimates for each (id, E, h).
    """
    comps = comparisons(df)
    c_outcomes = comparisons_outcomes(comps)
    ests = estimators(df, c_outcomes)
    return compare_ests(ests)


@dataclass
class ComparisonResults:
    comparisons: Comparisons
    comparisons_outcomes: ComparisonsOutcomes
    estimators: Estimators
    comparison: pl.DataFrame


def full_comparison(df: pl.DataFrame) -> ComparisonResults:
    """
    Returns dataclass with all comparison results.
    """
    comps = comparisons(df)
    c_outcomes = comparisons_outcomes(comps)
    ests = estimators(df, c_outcomes)
    ov = compare_ests(ests)
    return ComparisonResults(
        comparisons=comps,
        comparisons_outcomes=c_outcomes,
        estimators=ests,
        comparison=ov,
    )


def describe_ests(ests: Estimators):
    return (
        compare_ests(ests)
        .select("sgdd", "swdd")
        .describe(
            percentiles=[
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                0.75,
                0.9,
                0.95,
                0.975,
                0.99,
            ]
        )
    )


AggOption = Literal["dynamic", "group", "total"]


def get_cols(agg: AggOption):
    match agg:
        case "dynamic":
            cols = ["h"]
        case "group":
            cols = ["E", "h"]
        case _:
            raise ValueError(f"Invalid option: {agg}")
    return cols


PERCENTILES = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.975,
    0.99,
]


def quantile_desc(ests: Estimators, agg: AggOption = "dynamic"):
    cols = get_cols(agg)
    return (
        compare_ests(ests)
        .drop("id")
        .melt(id_vars=["E", "h"], value_name="estimate")
        .group_by(*cols, "variable")
        .agg([pl.col("estimate").quantile(q).alias(f"q_{q}") for q in PERCENTILES])
        .sort(*cols, "variable")
    )


def aggregate(df: pl.DataFrame, agg: AggOption = "dynamic", id_col: str | None = None):
    match agg:
        case "dynamic" | "group":
            cols = get_cols(agg)
            if not id_col:
                return df.group_by(cols).agg(pl.col("swdd", "sgdd").mean()).sort(cols)
            return (  # Counts if id col specified
                df.group_by(cols)
                .agg(
                    pl.col("swdd", "sgdd").mean(),
                    pl.col(id_col).count().alias("N"),
                )
                .sort(cols)
            )
        case "total":
            return df.select(pl.col("swdd", "sgdd").mean())


def get_comparison_arrays(res: pl.DataFrame):
    """Returns tuple of arrays for SWDD and SGDD estimates"""

    def _extract(col: str):
        return res.sort("h").select(col).to_numpy().squeeze()

    return _extract("swdd"), _extract("sgdd")


def bootstrap(df: pl.DataFrame, B: int = 999, agg: AggOption = "dynamic"):
    """
    Resample ids and compute the estimates
    """
    cols = get_cols(agg)
    ids = df["id"].unique().to_numpy()

    def _bstrap():
        b_ids = np.random.choice(ids, size=ids.size, replace=True)
        return (
            compare_estimators(df.filter(pl.col("id").is_in(b_ids)))
            .group_by(cols)
            .agg(pl.col("swdd", "sgdd").mean())
        )

    return pl.concat(
        _bstrap().with_columns(b=pl.lit(b))
        for b in tqdm(range(B), desc="Bootstrapping ...")
    )


def dynamic_bootstrap_overview(boot: pl.DataFrame):
    with pl.Config(tbl_rows=100):
        for (h,), _df in boot.sort("h").group_by("h"):
            print(f"Horizontal: {h}:")
            print(_df.drop("b", "h").describe(percentiles=PERCENTILES))
