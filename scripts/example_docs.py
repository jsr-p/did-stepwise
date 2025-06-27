# %% [markdown]
# ---
# format: gfm
# ---

# %% [markdown]
# The following example takes outset in the data simulated from
# [this](scripts/stata/harmon_sim_data.do) script taken from
# [Harmon's](https://github.com/nikoharm/did_stepwise/blob/main/stepwise_examples.do) repository.

# %%
# | echo: true

import did_imp
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import plotnine as pn

# %%
# | echo: true

import did_sw
from did_sw._testing import load_harmon_sim_data


data = load_harmon_sim_data()
transformed = data.transformed
df = data.raw

print(df.head())


# %%
# | echo: true

gp = df.group_by("E", "t").agg(pl.col("Y").mean())

fig, ax = plt.subplots(figsize=(10, 6))
ax = sns.lineplot(
    data=gp.to_pandas(),
    x="t",
    y="Y",
    hue="E",
    ax=ax,
)
for E in sorted(df["E"].unique()):
    ax.axvline(E, color="grey", linestyle="--")
ax.set(ylabel="Outcome", xlabel="$t$")
ax.legend(
    title="Treatment Cohort",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=3,
    frameon=False,
)
sns.despine()
_ = fig.savefig(
    did_sw.utils.proj_folder() / "figs/example_avg.png", dpi=300, bbox_inches="tight"
)
plt.close(fig)  # quarto

# %% [markdown]
# ![image](figs/example_avg.png)

# %%
# | echo: true

result = did_sw.estimate(
    df,
    outcome="Y",
    group="E",
    time="t",
    unit="id",
    fes="t",
    horizons="event",
)
print(result.estimates)

# %% [markdown]
# Underneath the hood, the `did_sw` estimator uses the imputation estimator of
# [BJS estimator](https://academic.oup.com/restud/article/91/6/3253/7601390)
# implemented in the `did_imp` [package](https://github.com/jsr-p/did-imputation)
# on a transformed dataset.
# Let's compute event study estimates using the imputation estimator directly on
# the non-transformed data.

# %%
# | echo: true

# Compare with imputation estimator
result_imp = did_imp.estimate(
    df,
    outcome="Y",
    group="E",
    time="t",
    unit="id",
    # NOTE: observe how the fixed effects here are `id + t` compared to `t` in
    # the `did_sw` estimator; also, the outcome above is the non-differenced
    # `Y`
    fes="id + t",
    horizons="event",
)
print(result_imp.estimates)

# %% [markdown]
# Plotting the estimates

# %%
# | echo: true


def plot_eventstudy(
    res: pl.DataFrame,
    breaks: list[int],
):
    """Helper to plot event study estimates"""
    p = (
        pn.ggplot(res, pn.aes(x="rel_year", y="estimate", color="group"))
        + pn.geom_point(position=pn.position_dodge(width=(w := 0.3)))
        + pn.geom_errorbar(
            pn.aes(ymin="lower", ymax="upper"),
            width=0.1,
            position=pn.position_dodge(width=w),
        )
        + pn.theme_classic()
        + pn.labs(
            x="Relative Time",
            y="Estimate",
            color="",
        )
        + pn.scale_x_continuous(breaks=breaks)
        + pn.geom_hline(yintercept=0, color="black", size=0.5, linetype="dotted")
        + pn.geom_vline(xintercept=0, color="black", size=0.5, linetype="dotted")
        + pn.theme(
            legend_position="bottom",
            axis_title=pn.element_text(size=14),
            axis_text=pn.element_text(size=12),
            legend_text=pn.element_text(size=12),
            legend_title=pn.element_text(size=13),
        )
    )
    return p


estimates = result.estimates.filter(pl.col("term").ne("average")).select(
    pl.col("term").cast(pl.Int8).alias("rel_year"),
    "estimate",
    "se",
    "lower",
    "upper",
    pl.lit("SWDD").alias("group"),
)
estimates_imp = result_imp.estimates.select(
    pl.col("term").cast(pl.Int8).alias("rel_year"),
    "estimate",
    "se",
    "lower",
    "upper",
    pl.lit("DID Imputation").alias("group"),
)
results = pl.concat([estimates, estimates_imp], how="vertical_relaxed").with_columns(
    pl.col("rel_year").cast(pl.Int8)
)

# Plot the estimates
p = plot_eventstudy(results, breaks=list(range(5)))
p.save(
    did_sw.utils.proj_folder() / "figs/example_estimates.png",
    width=10,
    height=6,
    transparent=False,
    verbose=False,
    dpi=300,
)

# %% [markdown]
# ![image](figs/example_estimates.png)


# %% [markdown]
# ### With covariates and clustered standard errors


# %%
# | echo: true

# Clustered

r = did_sw.estimate(
    df,
    outcome="Y",
    group="E",
    time="t",
    unit="id",
    cluster_var="clust",
    fes="t",
    horizons="event",
)
print(r.estimates)

# Categorical
r = did_sw.estimate(
    df,
    outcome="Y",
    group="E",
    time="t",
    cluster_var="id",
    unit="id",
    covariates=["C(X2) : C(t)"],
    fes="t",
    horizons="event",
)
print(r.estimates)


# Continuous
r = did_sw.estimate(
    df,
    outcome="Y",
    group="E",
    time="t",
    cluster_var="id",
    unit="id",
    covariates=["-1 + X1 : C(t)"],
    fes="t",
    horizons="event",
)
print(r.estimates)
