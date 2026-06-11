import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

def plot_dis(df, odir):
    
    sns.displot(df, x="Erkrankung", hue="Therapie")
    for cur_extension in [".png", ".pdf", ".svg"]:
        plt.savefig(f"{odir}/data_distribution_disease_therapy{cur_extension}")
    plt.close()

    sns.catplot(
        df, x="age", y="Erkrankung", hue="sex", kind="violin"
    )
    for cur_extension in [".png", ".pdf", ".svg"]:
        plt.savefig(f"{odir}/data_distribution_age_sex{cur_extension}")
    plt.close()

    relevant_variables = [
        "untersuchung",
        "age",
        "Erkrankung",
        "Verlaufsform",
        "ALSFRS-R Gesamt",
        "ALSFRS-R Sprechen",
        "QOL-DYS-G",
        "Muttersprache",
        "Nebendiagnosen (Neuro, Psych)",
    ]

    for var in relevant_variables:
        plt.figure()
        if df[var].dtype == "object" or df[var].nunique() < 10:
            sns.countplot(data=df, x=var, hue="sex")
        else:
            sns.histplot(data=df, x=var, hue="sex", kde=True)
        for cur_extension in [".png", ".pdf", ".svg"]:
            plt.savefig(f"{odir}/data_distribution_{var.replace(' ', '_')}{cur_extension}")
        plt.close()

def plot_error_vs_session_with_sem(eval_df, model, odir):
    df = (
        eval_df
        .reset_index()
        .sort_values(["IDs", "session"])
    )
    df["session_idx"] = df.groupby("IDs").cumcount() + 1

    stats = (
        df.groupby("session_idx")["abs_error"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    sem = stats["std"] / np.sqrt(stats["count"])

    plt.figure(figsize=(6, 4))
    plt.plot(stats["session_idx"], stats["mean"], marker="o")
    plt.fill_between(
        stats["session_idx"],
        stats["mean"] - 1.96 * sem,
        stats["mean"] + 1.96 * sem,
        alpha=0.25,
        label="95% CI",
    )

    plt.xlabel("Session index")
    plt.ylabel("Mean absolute error")
    plt.title(f"{model}: Forecasting error vs session")

    plt.xticks(stats["session_idx"])
    plt.legend()
    plt.tight_layout()

    for ext in [".png", ".pdf", ".svg"]:
        plt.savefig(
            f"{odir}/{model}_forecasting_error_vs_session_sem{ext}",
            dpi=300,
        )

    plt.close()

        
def plot_predicted_vs_reference(eval_df, model, odir):

    y_true = eval_df["agg_ref"].values
    y_pred = eval_df["agg_pred"].values

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.6,
        edgecolors="k"
    )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    plt.xlabel("References")
    plt.ylabel("Predictions")
    plt.title(f"{model}: Predicted vs Reference")

    plt.axis("equal")
    plt.tight_layout()

    for ext in [".png", ".pdf", ".svg"]:
        plt.savefig(
            f"{odir}/{model}_predicted_vs_reference_speakerlevel{ext}",
            dpi=300,
        )

    plt.close()

        
def plot_per_speaker_trajectories(eval_df, model, odir):
    """
    eval_df:
        MultiIndex: ["IDs", "session"]
        Columns: ["agg_ref", "agg_pred"]
    """

    # Ensure deterministic ordering
    eval_df = eval_df.sort_index(level=["IDs", "session"])

    speakers = eval_df.index.get_level_values("IDs").unique()
    n_speakers = len(speakers)

    cmap = cm.get_cmap("tab20", n_speakers)

    plt.figure(figsize=(10, 6))

    for i, spk in enumerate(speakers):
        df_spk = eval_df.xs(spk, level="IDs").copy()

        # Create per-speaker session index: 1, 2, 3, ...
        df_spk = df_spk.reset_index()
        df_spk["session_idx"] = np.arange(1, len(df_spk) + 1)

        color = cmap(i)

        # True values
        plt.plot(
            df_spk["session_idx"],
            df_spk["agg_ref"],
            linestyle="--",
            marker="o",
            color=color,
            alpha=0.8,
        )

        # Predicted values
        plt.plot(
            df_spk["session_idx"],
            df_spk["agg_pred"],
            linestyle="-",
            marker="x",
            color=color,
            alpha=0.8,
        )

    plt.xlabel("Session index")
    plt.ylabel("Questionnaire score")
    plt.title("Per-speaker true vs predicted trajectories")

    plt.xticks(
        ticks=np.arange(1, eval_df.groupby(level="IDs").size().max() + 1),
        labels=[f"Session {i}" for i in range(1, eval_df.groupby(level="IDs").size().max() + 1)],
        rotation=0,
    )

    plt.tight_layout()

    for ext in [".png", ".pdf", ".svg"]:
        plt.savefig(
            f"{odir}/{model}_per_speaker_true_vs_predicted_trajectories{ext}",
            dpi=300,
        )

    plt.close()
    
def plot_population_trajectory(eval_df, model, odir):

    # Build per-speaker session index
    df = (
        eval_df
        .reset_index()
        .sort_values(["IDs", "session"])
    )
    df["session_idx"] = df.groupby("IDs").cumcount() + 1

    # Aggregate across speakers
    agg = (
        df.groupby("session_idx")
        .agg(
            ref_mean=("agg_ref", "mean"),
            pred_mean=("agg_pred", "mean"),
            ref_std=("agg_ref", "std"),
            pred_std=("agg_pred", "std"),
            n=("agg_ref", "count"),
        )
        .reset_index()
    )

    # 95% CI
    ref_ci = 1.96 * agg["ref_std"] / np.sqrt(agg["n"])
    pred_ci = 1.96 * agg["pred_std"] / np.sqrt(agg["n"])

    # ---- Plot ----
    plt.figure(figsize=(6, 4))

    plt.plot(
        agg["session_idx"],
        agg["ref_mean"],
        linestyle="--",
        marker="o",
        label="Reference",
    )

    plt.plot(
        agg["session_idx"],
        agg["pred_mean"],
        linestyle="-",
        marker="x",
        label="Prediction",
    )

    plt.fill_between(
        agg["session_idx"],
        agg["ref_mean"] - ref_ci,
        agg["ref_mean"] + ref_ci,
        alpha=0.2,
    )

    plt.fill_between(
        agg["session_idx"],
        agg["pred_mean"] - pred_ci,
        agg["pred_mean"] + pred_ci,
        alpha=0.2,
    )

    plt.xlabel("Session index")
    plt.ylabel("Questionnaire score")
    plt.title(f"{model}: Population-level trajectories")

    plt.xticks(agg["session_idx"])
    plt.legend()
    plt.tight_layout()

    for ext in [".png", ".pdf", ".svg"]:
        plt.savefig(
            f"{odir}/{model}_population_trajectory{ext}",
            dpi=300,
        )

    plt.close()
