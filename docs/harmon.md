## Some theory

In the following:

- $`Y_{i, t}`$ is outcome for unit $`i`$ in period $`t`$.

- $`E_{i}`$ treatment date for unit $`i`$

- $`K_{i, t} = t - E_{i}`$ relative treatment time for unit $`i`$ in
  period $`t`$

- $`D_{i, t} = \mathbf{1}\{K_{i, t} \geq 0\}`$ treatment indicator for
  unit $`i`$ in period $`t`$

### SGDD vs SWDD estimators

Harmon characterizes two groups of DID estimators:

- **Subgroup Difference-in-Differences** (SGDD)

- **Stepwise Difference-in-Differences** (SWDD)

SGDD estimators have the form:
``` math
\begin{aligned}
    \hat{\gamma}^{SGDD}_{i, h}
    = (Y_{i, E_{i} + h} - Y_{i, E_{i} - 1})
    - \frac{1}{\mathcal{N}}
    \sum_{j: D_{j, E_{i + h} = 0}}
    (Y_{j, E_{i} + h} - Y_{j, E_{i} - 1}).
\end{aligned}
```
where we in the right part sum over all those $`j`$ who are not treated
by $`E_{i
            + h}`$ (i.e. $`D_{j, E_{i + h} = 0}`$).

OOH, SWDD estimators have the form:
``` math
\begin{aligned}
    \hat{\gamma}^{SWDD}_{i, h}
    =
    \sum_{k = 0}^{h}
    \left[
        (Y_{i, E_{i} + k} - Y_{i, E_{i} + k - 1})
        -
        \frac{1}{\mathcal{N}}
        \sum_{j: D_{j, E_{i + k} = 0}}
        (Y_{j, E_{i} + k} - Y_{j, E_{i} + k - 1})
        \right]
\end{aligned}
```
where $`\mathcal{N}^{-1}`$ is a normalization factor for the inner sum
i.e. it differs for each $`k`$. Note that the first term telescopes such
that
``` math
\begin{aligned}
    \sum_{k = 0}^{h}
    (Y_{i, E_{i} + k} - Y_{i, E_{i} + k - 1})
     & =
    (Y_{i, E_{i}} - Y_{i, E_{i}- 1})
    +
    (Y_{i, E_{i} + 1} - Y_{i, E_{i}})
    +
    \cdots
    +
    (Y_{i, E_{i} + h} - Y_{i, E_{i} + h - 1}) \\
     & =
    (Y_{i, E_{i} + h} - Y_{i, E_{i} - 1})
\end{aligned}
```
and thus
``` math
\begin{aligned}
    \hat{\gamma}^{SWDD}_{i, h}
    =
    (Y_{i, E_{i} + h} - Y_{i, E_{i} - 1})
    -
    \sum_{k = 0}^{h}
    \left[
        \frac{1}{\mathcal{N}}
        \sum_{j: D_{j, E_{i + k} = 0}}
        (Y_{j, E_{i} + k} - Y_{j, E_{i} + k - 1})
        \right]
\end{aligned}
```

Comparing $`\hat{\gamma}^{SWDD}_{i, h}`$ to
$`\hat{\gamma}^{SGDD}_{i, h}`$ we see that the SWDD estimator
potentially includes more information than the SGDD estimator, while it
includes the stepwise differences for the non-treated units; the
stepwise comparisons use the time tuples $`\{(E_{i} + k, E_{i} + k -
    1)\}_{k=0}^{h}`$. Because of the extra comparisons in the control
group, the SWDD estimator potentially has a lower variance (is more
efficient) than the SGDD estimator for longer horizons; see the paper
(in particular the online appendix) for details.
