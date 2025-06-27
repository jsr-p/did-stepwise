r"""
This script runs simulation experiments to compare estimator variance for
treatment effect estimation. It evaluates three estimators (SWDD, SGDD, and BJS)
under staggered and non-staggered designs, varying correlation (rho) settings.
Results are saved as CSV files for further analysis.

It corresponds to the "Numerical results" Section of the main paper.

See:
- Main paper: https://web.econ.ku.dk/nharmon/docs/harmon2022difference.pdf
- Appendix: https://web.econ.ku.dk/nharmon/docs/harmon2024onlineappendix.pdf
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from tqdm import tqdm

import did_imp
from did_sw import comparison, sim


@dataclass
class SimRes:
    swdd: np.ndarray
    sgdd: np.ndarray
    bjs: np.ndarray


np.random.seed(2025)
nsim = 500
N = 250


def compute_estimates(data: pl.DataFrame):
    ests = comparison.compare_estimators(data)
    aggs = comparison.aggregate(ests, agg="dynamic")
    swdd, sgdd = aggs.pipe(comparison.get_comparison_arrays)
    mod = did_imp.estimate(
        data,
        outcome="Y",
        group="E",
        time="t",
        unit="id",
        cluster_var="id",
        fes="t + id",
    )
    bjs = mod.estimates.select("estimate").to_numpy().squeeze()
    return SimRes(swdd=swdd, sgdd=sgdd, bjs=bjs)


def sim_round(rho: float = 1):
    data = sim.simulate_data(N=N, rho=rho)
    res = compute_estimates(data)
    return res


def sim_ns_round():
    """non-staggered; one treated group at t = 4"""
    data = sim.simulate_data(N=N, rho=1, E_is=[4, -99], cgroup=-99)
    res = compute_estimates(data)
    return res


def proc_results(
    res: list[SimRes],
    schema: list[str] = ["h0", "h1", "h2", "h3", "h4"],
):
    swdd = np.array([r.swdd for r in res])
    sgdd = np.array([r.sgdd for r in res])
    bjs = np.array(
        [
            r.bjs
            for r in res
            # TODO: find out why for a few cases it equals 4
            # Ensure BJS has the same length as schema
            if r.bjs.shape[0] == len(schema)
        ]
    )

    v_swdd = swdd.var(axis=0)
    v_sgdd = sgdd.var(axis=0)
    v_bjs = bjs.var(axis=0)

    vs = np.vstack([v_swdd, v_sgdd, v_bjs])
    min_var = vs.min(axis=0)
    rel_vs = vs / min_var

    avgs = np.vstack(
        [
            swdd.mean(axis=0),
            sgdd.mean(axis=0),
            bjs.mean(axis=0),
        ]
    )

    res_rel = (
        pl.DataFrame(
            rel_vs,
            schema=schema,
        )
        .with_columns(estimator=pl.Series(["swdd", "sgdd", "bjs"]))
        .select("estimator", *schema)
    )
    res_data = pl.concat(
        [
            pl.DataFrame(data, schema=schema).with_columns(
                estimator=pl.lit(est), sim=pl.int_range(1, pl.len() + 1)
            )
            for data, est in zip([swdd, sgdd, bjs], ["swdd", "sgdd", "bjs"])
        ]
    )

    print(f"SWDD: {v_swdd}")
    print(f"SGDD: {v_sgdd}")
    print(f"BJS: {v_bjs}")
    print(f"Relative variance:\n{res_rel}")
    print(f"Average estimates:\n{avgs}")

    return res_rel, res_data


def experiment_staggered():
    print("Performing simulation experiment")
    for rho in [1, 0.8, 0.5]:
        print(f"rho: {rho}")
        res = [sim_round(rho) for _ in tqdm(range(nsim), desc="Simulating")]
        res_rel, res_data = proc_results(res)
        # Store estimates for e.g. plotting
        res_data.write_csv(f"data/harmon-sim-res-rho{rho}.csv")
        res_rel.write_csv(f"data/harmon-sim-res-rel-rho{rho}.csv")


def experiment_non_staggered():
    print("Performing simulation experiment")
    res = [sim_ns_round() for _ in tqdm(range(nsim), desc="Simulating")]
    res_rel, res_data = proc_results(res, schema=["h0", "h1", "h2"])
    # Store estimates for e.g. plotting
    res_data.write_csv("data/harmon-sim-res-nonstag.csv")
    res_rel.write_csv("data/harmon-sim-res-rel-nonstag.csv")


def main():
    print(f"Performing simulation experiment with {nsim} simulations")
    print(" Simulation experiment staggered ".center(80, "-"))
    experiment_staggered()
    print(" Simulation experiment non-staggered ".center(80, "-"))
    experiment_non_staggered()


if __name__ == "__main__":
    main()
