"""Prospective design resources and explicitly hypothetical power scenarios."""
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from .curation import ROOT


def crossed_standard_error(matrix):
    """ANOVA moment variance for a balanced lab x peptide matrix of paired losses."""
    labs, peptides = matrix.shape
    grand = matrix.mean()
    row = matrix.mean(axis=1)
    col = matrix.mean(axis=0)
    residual = matrix-row[:, None]-col[None, :]+grand
    ms_lab = peptides*np.sum((row-grand)**2)/(labs-1)
    ms_peptide = labs*np.sum((col-grand)**2)/(peptides-1)
    ms_residual = np.sum(residual**2)/((labs-1)*(peptides-1))
    v_lab = max(0, (ms_lab-ms_residual)/peptides)
    v_peptide = max(0, (ms_peptide-ms_residual)/labs)
    return np.sqrt(v_lab/labs + v_peptide/peptides + ms_residual/(labs*peptides))


def simulate_power(labs, peptides, targets, contexts, days, effect, lab_sd, repeats, rng):
    submeasurements = targets*contexts*days
    peptide_sd, interaction_sd, residual_sd = .35, .5, .9
    successes = 0
    critical = t.ppf(.975, df=min(labs-1, peptides-1))
    for _ in range(repeats):
        # This simulates paired absolute-error DIFFERENCES, not MIC measurements
        # or the activity of specific organisms or peptides.
        matrix = (effect + rng.normal(0, lab_sd, (labs, 1))
                  + rng.normal(0, peptide_sd, (1, peptides))
                  + rng.normal(0, np.sqrt(interaction_sd**2+residual_sd**2/submeasurements),
                               (labs, peptides)))
        se = crossed_standard_error(matrix)
        successes += matrix.mean() > critical*se
    rate = successes/repeats
    return rate, np.sqrt(rate*(1-rate)/repeats)


def build_prospective(repeats=4000):
    output = ROOT / "prospective"
    output.mkdir(exist_ok=True)
    designs = [
        {"design": "pilot", "labs": 3, "peptides": 24, "targets": 2, "contexts": 2, "days": 2},
        {"design": "expanded", "labs": 5, "peptides": 40, "targets": 3, "contexts": 4, "days": 3},
        {"design": "more_laboratories", "labs": 8, "peptides": 40, "targets": 3, "contexts": 4, "days": 3},
    ]
    rows = []
    rng = np.random.default_rng(20261020)
    for design in designs:
        for lab_sd, effect in itertools.product([.15, .35], [0, .01, .05, .10, .25, .5, .75]):
            power, mcse = simulate_power(**{k: v for k, v in design.items() if k != "design"},
                                         effect=effect, lab_sd=lab_sd, repeats=repeats, rng=rng)
            assays = np.prod([design[k] for k in ["labs", "peptides", "targets", "contexts", "days"]])
            rows.append({**design, "independent_assay_records": int(assays),
                         "assumed_mae_improvement_log2": effect, "assumed_lab_sd": lab_sd,
                         "simulated_power_or_type_I_error": power, "monte_carlo_se": mcse,
                         "simulation_replicates": repeats})
    table = pd.DataFrame(rows)
    table.to_csv(output / "power_scenarios.csv", index=False)
    assumptions = {
        "status": "hypothetical planning simulation, not an empirically powered final protocol",
        "effect": "mean paired reduction in absolute log2 prediction error",
        "assumed_effects_log2": [0, .01, .05, .10, .25, .5, .75],
        "designs": designs,
        "lab_sd": [.15, .35], "peptide_sd": .35, "lab_peptide_sd": .5, "residual_sd": .9,
        "test": "positive lower end of two-sided 95% t interval using crossed ANOVA variance; conservative df=min(L-1,P-1)",
        "crossing": "all planned peptides, targets, contexts and independent days in every laboratory",
        "generalization": "laboratories and peptides sampled; selected targets and contexts are fixed design factors",
        "costs": "not quoted; total = peptide procurement + assay records * local assay cost + controls + logistics",
        "required_update": "Replace variance assumptions with pilot paired-error data before freezing the confirmatory sample size.",
        "minimum_design": "pilot is exploratory; one held-out laboratory alone cannot estimate population-level lab variability",
        "seed": 20261020,
    }
    (output / "power_assumptions.json").write_text(json.dumps(assumptions, indent=2))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    colors = ["#236B8E", "#DA8544", "#307B62"]
    for ax, sd in zip(axes, [.15, .35]):
        for color, design in zip(colors, designs):
            subset = table[(table.design == design["design"]) & (table.assumed_lab_sd == sd)]
            ax.plot(subset.assumed_mae_improvement_log2, subset.simulated_power_or_type_I_error,
                    marker="o", color=color, label=f"{design['labs']} labs / {design['peptides']} peptides")
        ax.axhline(.8, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(f"Assumed lab SD: {sd} log2 units")
        ax.set_xlabel("Assumed paired MAE improvement (log2)")
        ax.set_ylim(0, 1.03)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Simulated rejection probability")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.005))
    fig.suptitle("Planning scenarios — variance assumptions require pilot validation", fontsize=12)
    fig.tight_layout(rect=[0, .08, 1, .97])
    fig.savefig(output / "power_scenarios.png", dpi=220)
    fig.savefig(output / "power_scenarios.pdf")
    plt.close(fig)
    columns = [
        "record_id", "laboratory_code", "operator_code", "peptide_blind_code",
        "chemical_identity_id", "peptide_lot_id", "target_blind_code", "target_identity",
        "context_code", "protocol_version", "medium_name", "medium_lot",
        "cfu_value", "cfu_unit", "supplement_name", "supplement_value", "supplement_unit",
        "ph", "independent_day_id", "technical_repeat_id", "run_order",
        "mic_lower", "mic_upper", "mic_point", "mic_unit", "censoring_kind",
        "dilution_grid_id", "control_status", "exclusion_reason", "raw_readout_path",
        "prediction_sequence_model", "prediction_context_model", "prediction_locked_utc",
    ]
    pd.DataFrame(columns=columns).to_csv(output / "measurement_template.csv", index=False)
    design = designs[0]
    planned = []
    for lab, day in itertools.product(range(1, design["labs"]+1), range(1, design["days"]+1)):
        block = list(itertools.product(range(1, design["peptides"]+1),
                                       range(1, design["targets"]+1), range(1, design["contexts"]+1)))
        rng.shuffle(block)
        for order, (pep, target, context) in enumerate(block, 1):
            planned.append({
                "status": "planned_placeholder_not_a_measurement", "laboratory_code": f"L{lab:02d}",
                "independent_day_id": f"L{lab:02d}_D{day:02d}", "peptide_blind_code": f"P{pep:03d}",
                "target_blind_code": f"T{target:02d}", "context_code": f"C{context:02d}",
                "run_order": order,
            })
    pd.DataFrame(planned).to_csv(output / "pilot_randomized_allocation.csv", index=False)
    print(table.to_string(index=False), flush=True)
