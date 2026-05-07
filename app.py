from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shiny import App, render, ui

from mortality import pv_annuity

AGES = np.arange(50, 81)

GENDER_PAIRS = [
    ("male", "male"),
    ("male", "female"),
    ("female", "male"),
    ("female", "female"),
]

FACET_LABEL = {
    ("male", "male"): "Male / Male",
    ("male", "female"): "Male / Female",
    ("female", "male"): "Female / Male",
    ("female", "female"): "Female / Female",
}

PALETTE = {"First Death": "#1f77b4", "Second Death": "#d62728"}

app_ui = ui.page_fluid(
    ui.h3("Joint Life Annuity — Present Value"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_slider(
                "age_diff",
                "Age difference  (Life 2 − Life 1)",
                min=-20,
                max=20,
                value=0,
                step=1,
            ),
            ui.input_numeric("discount_pct", "Discount rate (%)", value=5.0, min=0.1, max=20.0, step=0.5),
            ui.input_numeric("improvement_pct", "Annual improvement (%)", value=1.5, min=0.0, max=5.0, step=0.1),
        ),
        ui.output_plot("annuity_plot", height="640px"),
    ),
)


def server(input, output, session):
    @output
    @render.plot
    def annuity_plot():
        age_diff = input.age_diff()
        dr = input.discount_pct() / 100.0
        ir = input.improvement_pct() / 100.0

        rows = []
        for g1, g2 in GENDER_PAIRS:
            for atype in ("first_death", "second_death"):
                pvs = pv_annuity(
                    gender=g1,
                    age=AGES,
                    discount_rate=dr,
                    improvement_rate=ir,
                    joint=True,
                    gender2=g2,
                    age_diff=age_diff,
                    annuity_type=atype,
                )
                label = "First Death" if atype == "first_death" else "Second Death"
                rows.extend(
                    {"Age": int(a), "PV": float(pv), "Annuity Type": label, "facet": FACET_LABEL[(g1, g2)]}
                    for a, pv in zip(AGES, pvs)
                )

        df = pd.DataFrame(rows)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)

        for ax, (g1, g2) in zip(axes.flat, GENDER_PAIRS):
            subset = df[df["facet"] == FACET_LABEL[(g1, g2)]]
            sns.lineplot(data=subset, x="Age", y="PV", hue="Annuity Type", palette=PALETTE, ax=ax)
            ax.set_xlim(50, 80)
            ax.set_ylim(0, 30)
            ax.set_title(FACET_LABEL[(g1, g2)], fontsize=11)
            ax.set_xlabel("Age (Life 1)")
            ax.set_ylabel("Present Value")
            ax.legend(title=None, fontsize=9)

        fig.suptitle(
            f"Discount {input.discount_pct():.1f}%  |  "
            f"Improvement {input.improvement_pct():.1f}%  |  "
            f"Age difference {age_diff:+d}",
            fontsize=11,
            y=1.01,
        )
        fig.tight_layout()
        return fig


app = App(app_ui, server)
