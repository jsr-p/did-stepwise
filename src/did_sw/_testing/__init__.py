import did_sw
import polars as pl

from dataclasses import dataclass


@dataclass
class Data:
    raw: pl.DataFrame
    transformed: pl.DataFrame


def load_harmon_sim_data():
    base = pl.read_csv(did_sw.utils.proj_folder() / "data/harmon-sim.csv").rename(
        {"Ei": "E", "i": "id"}
    )
    bbase = (
        (
            base.sort("id", "t")
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
    return Data(raw=base, transformed=bbase)
