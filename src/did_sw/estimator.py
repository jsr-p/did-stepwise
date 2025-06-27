"""
SWDD Estimator of Harmon.

See paper:
- Paper: https://web.econ.ku.dk/nharmon/docs/harmon2022difference.pdf
- Appendix: https://web.econ.ku.dk/nharmon/docs/harmon2024onlineappendix.pdf
"""

from functools import reduce

import polars as pl
from dataclasses import dataclass
from pyfixest.estimation.feols_ import Feols
from typing import Literal

import did_imp


__all__ = [
    "DidSwResult",
    "estimate",
    "assign_weights_horizon",
    "assign_weights_agg",
    "rename_horizons",
]


def rename_horizons(df: pl.DataFrame):
    """Pretrends are horizons + 2 cf. code of Harmon."""
    cols = [col for col in df.columns if "horizon" in col]
    pts = ["pretrend" + str(int(col.strip("horizon")) + 2) for col in cols]
    return df.rename(dict(zip(cols, pts)))


def assign_weights_horizon(
    df: pl.DataFrame,
    id_col: str = "id",
    min_K: int = 0,
    k_vals: list[int] | None = None,
    prefix: str = "horizon",
):
    """
    Args:
        min_K: Minimum horizon to consider.
    """

    def _assign_horizon(df: pl.DataFrame, h: int):
        #  TODO: custom iwtr should be passed by user
        return df.with_columns(
            # Sum of weights for K == h
            iwtr_s=pl.col("iwtr").filter(pl.col("K").eq(h)).sum()
        ).with_columns(
            # Compute weights for each horizon;
            # 0 <= K <= x; maxK >= x;
            pl.col("K")
            .is_between(0, h, closed="both")
            .and_(pl.col("maxK").ge(h))
            .mul(pl.col("iwtr").truediv(pl.col("iwtr_s")))
            .over(id_col)
            .alias(f"{prefix}{h}")
        )

    if not k_vals:
        k_max = df["K"].max()
        if not isinstance(k_max, int):
            raise ValueError("Column 'K' has no non-null values.")
        maxmaxK = int(k_max)
        k_vals = list(range(min_K, maxmaxK + 1))
    return reduce(_assign_horizon, k_vals, df)


def assign_weights_agg(
    df: pl.DataFrame,
    id_col: str = "id",
):
    """Assigns weights used to compute the aggregate effect.

    Computes weights for aggregate effect for the SWDD estimator using
    Borusyaks imputation estimator.
    """
    return df.with_columns(
        a2w=pl.col("maxK").sub("K").add(1).mul(pl.col("K").ge(0)).over(id_col),
        iwtr_s=pl.col("iwtr").filter(pl.col("K").ge(0)).sum(),
    ).with_columns(
        average=(pl.col("a2w") / pl.col("iwtr_s")),
    )


@dataclass
class DidSwResult:
    estimates: pl.DataFrame
    N: int
    data: pl.DataFrame
    names: list[str]
    mod: Feols

    def __repr__(self):
        return repr(self.estimates)

    def __str__(self):
        res = str(self.estimates)
        return "\n".join(
            [
                "**** Estimation results ****",
                f"Nobs: {self.N}",
                res,
            ]
        )


def estimate(
    data: pl.DataFrame,
    outcome: str,
    group: str,
    time: str,
    unit: str,
    cluster_var: str | None = None,
    fes: str | None = None,
    covariates: list[str] | None = None,
    weights: list[str] | None = None,
    horizons: Literal["static", "event", "all"] | list[int] | None = "event",
    pretrends: bool | list[int] | None = None,
    aweight: str | None = None,
    prep: bool = True,
) -> DidSwResult:
    """
    Estimate treatment effects using the Stepwise Difference-in-Differences (SWDD)
    estimator of Harmon (2022).

    This estimator constructs outcome leads/lags relative to unit-specific treatment
    timing and estimates weighted treatment effects across horizons using an
    imputation-based regression approach.

    Args:
        data: A `polars.DataFrame` containing the panel dataset.
        outcome: Name of the outcome variable.
        group: Name of the treatment group variable.
        time: Name of the time variable.
        unit: Name of the unit identifier.
        cluster_var: Optional variable for clustering standard errors.
        fes: Optional fixed effects specification (e.g., "unit + time").
        covariates: Optional list of time-invariant covariates.
        weights: Optional list of custom weight variable names to use in regression.
        horizons: Specifies which horizons to estimate. One of:
            - "event": Estimate event-study weights (default).
            - list[int]: Custom list of relative treatment periods to include.
            - "static": Estimate an average treatment effect over all
                        post-treatment horizons.
            - "all": Include both event-study and static weights.
            - None: Use weights provided via `weights`.
        pretrends: If True or list[int], reserve certain leads as pretrends
            (currently not implemented).
        aweight: Optional analytic weights variable (currently not implemented).
        prep: Whether to internally preprocess the data (e.g., compute `K`, `dY`, etc).

    Returns:
        A `DidSwResult` object containing:
            - estimates: DataFrame of coefficient estimates.
            - N: Number of observations used.
            - data: The processed dataset used in estimation.
            - names: List of variable names used.
            - mod: The underlying `Feols` model object.

    TODO:
        - throw error if cont covariates varies across time
            - must be time invariant!
            - weights have to be constant across time within units
        - pretrends
    """
    if aweight:
        raise NotImplementedError("TODO: aweight")
    if pretrends:
        raise NotImplementedError("TODO: fix pretrends")

    if prep:
        params = did_imp.DidImpParams(
            group=group,
            time=time,
            unit=unit,
            outcome="dY",
        )
        data = (
            data.sort(unit, time)
            # assigns relative time K and treatment D
            .pipe(did_imp.prep_data, params)
            .with_columns(
                iwtr=pl.lit(1),  #  TODO: custom iwtr should be passed by user
                dY=pl.col(outcome).diff().over(unit),
                maxK=pl.col("K").max().over(unit),
            )
            .drop_nulls(subset="dY")
        )
    else:
        # Assumes data is already transformed ready for estimation
        params = did_imp.DidImpParams(
            group=group,
            time=time,
            unit=unit,
            outcome=outcome,
        )

    if weights is None:
        weights = []

    if horizons:
        match horizons:
            case "event":
                data = data.pipe(assign_weights_horizon, id_col=unit)
                weights = data.select(pl.selectors.matches("horizon|average")).columns
            case list() if all(isinstance(x, int) for x in horizons):
                data = data.pipe(assign_weights_horizon, k_vals=horizons)
                weights = data.select(pl.selectors.matches("horizon|average")).columns
            case "static":
                data = data.pipe(assign_weights_agg, id_col=unit)
                weights.append("treat")
            case "all":
                data = data.pipe(assign_weights_horizon, id_col=unit).pipe(
                    assign_weights_agg, id_col=unit
                )
                weights = data.select(pl.selectors.matches("horizon|average")).columns
            case _:
                raise ValueError(
                    f"Invalid type for horizons:\n{type(horizons)=}\n{horizons=}"
                )
    if not horizons and len(weights) == 0:
        raise ValueError(
            "`horizons=None` provided but also no weights are specified. "
            "At least one horizon or weight must be provided."
        )

    imp_res = did_imp.estimate(
        data,
        outcome=params.outcome,
        group=params.group,
        time=params.time,
        cluster_var=cluster_var,
        unit=unit,
        fes=fes,
        covariates=covariates,
        weights=weights,
        horizons=None,  # We construct the horizons ourselves for swdd
    )
    estimates = imp_res.estimates.with_columns(
        pl.col("term").str.replace("^horizon", "")
    )
    return DidSwResult(
        estimates,
        N=data.shape[0],
        data=data,
        names=imp_res.names,
        mod=imp_res.mod,
    )
