from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DATA_PATH = Path(__file__).parent / "data" / "pma92c20_pfa92c20.csv"
_MAX_AGE = 120


def _load_table() -> dict[tuple[str, int], float]:
    df = pd.read_csv(_DATA_PATH)
    return {(row.sex, int(row.age)): row.qx for row in df.itertuples()}


_QX: dict[tuple[str, int], float] = _load_table()


def _get_qx(sex: str, age: int) -> float:
    return _QX.get((sex, age), 1.0)


def _survival_series(sex: str, start_age: int, improvement_rate: float) -> np.ndarray:
    """
    t_p_x for t = 0 … MAX_AGE - start_age with prospective cohort improvement.
    q(x+k) is scaled by (1 - improvement_rate)^k before computing survival.
    """
    horizon = _MAX_AGE - start_age
    tpx = np.empty(horizon + 1)
    tpx[0] = 1.0
    for k in range(horizon):
        q = _get_qx(sex, start_age + k) * (1.0 - improvement_rate) ** k
        tpx[k + 1] = tpx[k] * (1.0 - min(q, 1.0))
    return tpx


def pv_annuity(
    gender: str,
    age: int | float | np.ndarray,
    discount_rate: float,
    improvement_rate: float,
    joint: bool = False,
    gender2: str | None = None,
    age_diff: int = 0,
    annuity_type: str = "first_death",
) -> float | np.ndarray:
    """
    Present value of a whole-life annuity-due (payments at t = 0, 1, 2, …).

    Parameters
    ----------
    gender           : 'male' or 'female'
    age              : scalar or array-like — age of life 1
    discount_rate    : e.g. 0.05 for 5 %
    improvement_rate : e.g. 0.015 for 1.5 % p.a. cohort improvement
    joint            : True for joint-life annuity
    gender2          : sex of life 2 (required when joint=True)
    age_diff         : age2 = age + age_diff
    annuity_type     : 'first_death' (both alive) or 'second_death' (last survivor)

    Returns
    -------
    Scalar float when age is scalar, numpy array otherwise.
    """
    scalar = np.isscalar(age)
    ages = np.atleast_1d(np.asarray(age, dtype=float)).ravel()
    v = 1.0 / (1.0 + discount_rate)
    out = np.empty(len(ages))

    for i, a in enumerate(ages):
        a1 = int(a)
        tpx = _survival_series(gender, a1, improvement_rate)
        n = len(tpx) - 1
        disc = v ** np.arange(n + 1)

        if not joint:
            out[i] = float(np.dot(disc, tpx))
        else:
            a2 = a1 + int(age_diff)
            tpy = _survival_series(gender2, a2, improvement_rate)
            n = min(n, len(tpy) - 1)
            d, px, py = disc[: n + 1], tpx[: n + 1], tpy[: n + 1]
            if annuity_type == "first_death":
                out[i] = float(np.dot(d, px * py))
            else:
                out[i] = float(np.dot(d, px + py - px * py))

    return float(out[0]) if scalar else out
