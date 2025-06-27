import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

__all__ = ["SimParams", "simulate_data"]


@dataclass
class SimParams:
    T: int
    scale: float = math.sqrt(2 / 5)
    rho: float = 1


def sim(params: SimParams):
    """
    Recall:
        Variance ar(1):
                Var(y_t) = sigma^2 / (1 - rho^2)
                where sigma^2 variance of error term
                i.e. sigma^2 = 2/5.
        Variance random walk:
                Var(y_t) = sigma^2 * t
                while each eps_t iid and
                Var(y_t) = \\sum_{j=1}^{t} Var(eps_t) = t * sigma^2.

    Note: in our case var(eps_1) = 0, so Var(y_t) = (t - 1) * sigma^2
    and thus Var(y_{6}) = 5 * sigma^2 = 5 * 2 /5 = 2.
    """
    T, scale, rho = params.T, params.scale, params.rho
    eta = np.random.normal(size=(T,), scale=scale, loc=0)
    eps = np.zeros(T)
    for t in range(1, T):
        eps[t] = rho * eps[t - 1] + eta[t]
    return eps


def simulate_data(
    N: int = 250,
    rho: float = 1,
    E_is: list[int] | None = [2, 3, 4, 5, 6, -99],
    periods: list[int] | None = None,
    cgroup: int = -99,
    scale: float = math.sqrt(2 / 5),
):
    """
    cgroup: control group E number
    E_is: list of treatment dates

    Note:
        - We fill all never-treated as None.
        - The treatment effects γ_{i, h} = γ_{h} = 1 + K, where K = t - E.
    """
    if E_is is None:
        E_is = list(range(2, 6 + 1)) + [cgroup]
    if periods is None:
        periods = list(range(1, 6 + 1))
    T = len(periods)
    params = SimParams(T=T, scale=scale, rho=rho)
    df = (
        pl.DataFrame(
            {
                "id": np.arange(1, N + 1),
                "E": np.random.choice(E_is, size=(N,)),
                "F": np.random.choice([0, 1], size=(N,)),  # Binary group indicator
            }
        )
        .with_columns(
            t=periods,
            alpha=pl.when(pl.col("E").eq(cgroup))
            .then(0)
            .otherwise(pl.col("E").mul(-1)),
        )
        .explode("t")
        .with_columns(
            eps=pl.Series(np.concatenate([sim(params) for _ in range(N)])),
            beta=pl.col("t").mul(3),
            K=pl.when(pl.col("E").eq(cgroup))
            .then(None)
            .otherwise(pl.col("t") - pl.col("E")),
        )
        .with_columns(
            gamma=pl.when(pl.col("K") >= 0).then(1 + pl.col("K")).otherwise(0),
            gamma2=pl.when(pl.col("K") >= 0)
            .then(1 + pl.col("K") - pl.col("F") * 0.5)
            .otherwise(0),
        )
        .with_columns(
            # Note that H^{k}_{i, t} = 1{K_{i, t} = k}; so can just add gamma
            # directly (since gamma = 0 if K < 0)
            Y=pl.col("alpha") + pl.col("beta") + pl.col("gamma") + pl.col("eps"),
            # Outcome with different treatment effects across group indicator
            Y2=pl.col("alpha") + pl.col("beta") + pl.col("gamma2") + pl.col("eps"),
        )
        .with_columns(
            D=pl.col("K").ge(0).cast(pl.Int8).fill_null(0),
        )
    )
    return df


def sim_simple():
    T = 200
    for rho in [1, 0.8, 0.5]:
        params = SimParams(T=T, scale=math.sqrt(2 / 5), rho=rho)
        arr = sim(params)
        ts = np.arange(1, T + 1)
        fig, ax = plt.subplots()
        ax = sns.lineplot(x=ts, y=arr, marker="o", ax=ax)
        ax.set(title=f"Simulated Data: rho = {rho}", xlabel="t", ylabel="Outcome")
        sns.despine()
        ax.get_figure().savefig(f"figs/sim_rho{rho}.png")
        plt.close("all")
        print(f"Saved sim_rho{rho}.png")


def main():
    df = simulate_data()
    df.glimpse()

    data = df

    with pl.Config(tbl_rows=50):
        print(gp := data.group_by("E", "t").agg(pl.col("Y").mean()).sort("E", "t"))

    fig, ax = plt.subplots()
    ax = sns.lineplot(
        data=data.cast({"E": pl.Utf8}),
        x="t",
        y="Y",
        hue="E",
        marker="o",
        ax=ax,
        errorbar="ci",  # Bootstrapped CI
    )
    ax.set(title="Simulated Data", xlabel="t", ylabel="Outcome")
    sns.despine()
    ax.get_figure().savefig("sim2.png")
    plt.close("all")

    # Second
    data = data.cast({"E": pl.Utf8})
    g = sns.FacetGrid(data.to_pandas(), col="F", sharey=True, sharex=True)
    g.map_dataframe(
        sns.lineplot,
        x="t",
        y="Y2",
        hue="E",
        marker="o",
        errorbar="ci",  # Bootstrapped CI
    )
    g.set_titles(col_template="F = {col_name}")
    g.set_axis_labels("t", "Outcome")
    g.despine(left=True)
    g.add_legend()
    g.savefig("sim_facet.png")
    plt.close("all")

    sim_simple()


if __name__ == "__main__":
    main()
