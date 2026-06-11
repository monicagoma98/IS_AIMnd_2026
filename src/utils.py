import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import audeer

import csv

import librosa
import soundfile as sf

from pathlib import Path

import noisereduce as nr

import random
import torch


def process_audio_files(input_folder, output_folder):
    """
    Processes audio files in the input_folder to remove background noise and saves them to output_folder.
    """
    for root, dirs, files in os.walk(input_folder):
        # print(input_folder)
        for file in files:
            if not file.lower().endswith(
                (
                    ".wav",
                    ".mp3",
                    ".flac",
                    ".m4a",
                    ".mp4",
                    ".ogg",
                    "MP4",
                    "WAV",
                    "FLAC",
                    "M4A",
                    "MP3",
                    "OGG",
                )
            ):
                continue

            input_path = Path(root) / file

            # Compute corresponding output path (same structure preserved)
            relative_path = input_path.relative_to(input_folder)
            output_path = output_folder / relative_path

            # Create subdirectories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Change extension to .wav if you want everything standardized
            output_path = output_path.with_suffix(".wav")
            print(f"Processing → {output_path}")

            # Only process if output_path does not exist
            if output_path.exists():
                continue

            # try:
            # Load audio
            y, sr = librosa.load(input_path, sr=None, mono=True)

            # Noise reduction via spectral gating
            if y.ndim == 1:
                y_denoised = nr.reduce_noise(y=y, prop_decrease=0.7, sr=sr)
            else:
                y_denoised = np.vstack(
                    [nr.reduce_noise(y=ch, prop_decrease=0.7, sr=sr) for ch in y]
                )
            sf.write(output_path, y_denoised, sr)
            print(f"Success → {output_path}")

            # except Exception as e:
            #    print(f"Failed → {input_path} | Error: {e}")


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_predictions(X_test, y_pred, y_test, save_dir, model):

    df_pred = X_test.copy()
    df_pred.reset_index(inplace=True)
    df_pred = df_pred[["file", "session", "IDs"]]
    assert len(y_test) == len(
        df_pred
    ), f"y_test ({len(y_test)}) and X_test ({len(df_pred)}) length mismatch"
    assert len(y_pred) == len(
        df_pred
    ), f"Predictions ({len(y_pred)}) and X_test ({len(df_pred)}) length mismatch"
    df_pred["references"] = y_test
    df_pred["predictions"] = y_pred

    df_pred.to_csv(f"{save_dir}/{model}_predictions.csv", index=True)


def dt2sec(df, reset_index=True):
    """converts start and end index values from datetime
    to seconds.
    Args:
    df: pd.DataFrame in unified format
    reset_index: (boolean) if True, "file, start, end" will be
        columns. If False, they will be index
    """

    if "start" not in df.columns:
        df.reset_index(inplace=True)

    df["start"] = pd.to_timedelta(df["start"])
    df["end"] = pd.to_timedelta(df["end"])

    df["start"] = df["start"].dt.total_seconds()
    df["end"] = df["end"].dt.total_seconds()

    if not reset_index:
        df.set_index(["file", "start", "end"], inplace=True)

    return df


def find_wav_files(root_dir):
    """Finds all .wav files in a directory and its subdirectories and returns a pandas DataFrame.
    Args:
      root_dir: The root directory to search for .wav files.
    Returns:
      A pandas DataFrame with a single column named "file" containing the paths to the .wav files.
    """

    print("Finding .wav files in directory:", root_dir)
    df_audio = []
    prompts = set()
    for id_folder in os.listdir(root_dir):
        #print(len(os.listdir(root_dir)))
        id_folder_path = os.path.join(root_dir, id_folder)
        if os.path.isdir(id_folder_path):
            for session_file in os.listdir(id_folder_path):
                #print(session_file)
                for audio_file in os.listdir(
                    os.path.join(id_folder_path, session_file)
                ):
                    #print(audio_file)
                    if audio_file.lower().endswith(".wav"):
                        name_parts = os.path.splitext(audio_file)[0].split("_")
                        #print(name_parts)
                        if len(name_parts) == 3:
                            prompt = name_parts[1]
                            #session = name_parts[0]
                            #sessions are ONLY YYYYMMDD, if it is not, give a warning
                            #ids = name_parts[2]

                            speech_task_rep = "NA"
                        elif len(name_parts) == 4:
                            prompt = name_parts[1]
                            #session = name_parts[0]
                            #ids = name_parts[3]
                            speech_task_rep = name_parts[2]
                        
                        prompts.add(prompt)                            
                        
                        ids = id_folder
                        session = session_file.split("/")[-1]
                        
                        #if ids != id_folder:
                        #    print(
                        #        f"Warning: IDs in filename ({ids}) do not match folder name ({id_folder}). Skipping file: {audio_file}"
                        #    )

                        df_audio.append(
                            {
                                "IDs": ids,
                                "speech_task": prompt,
                                "session": session,
                                "audio_file": audio_file,
                                "speech_task_rep": speech_task_rep,
                                # "file": audio_file,
                                "file": f"{id_folder_path}/{session_file}/{audio_file}",
                            }
                        )
    #input(df_audio)
    print(f"{prompts =}")
    return pd.DataFrame(df_audio)


def get_speaker_id_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    return parts[-3] if len(parts) > 2 else None


def get_mean_std_os(df_feat):
    grouped_df = df_feat.groupby("file").agg(["mean", "std"])

    grouped_df.columns = ["_".join(col).strip() for col in grouped_df.columns.values]

    grouped_df = grouped_df.reset_index()

    return grouped_df


def plot_feat_distribution(df, feature_name, output_dir):
    """Plots the distribution of a feature."""
    out_plot = audeer.mkdir(f"{output_dir}/feature_distributions")

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature_name, kde=True, bins=30)
    plt.title(f"Distribution of {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")

    for cur_extension in [".png", ".pdf", ".svg"]:
        plt.savefig(
            f"{out_plot}/distribution_{feature_name}{cur_extension}",
            bbox_inches="tight",
        )
    plt.close()


def plot_sign(
    df,
    label,
    sig_feature,
    task,
    featureset,
    aggregate,
    disease,
    speech_task,
    output_dir,
):
    """returns box plot of significant features"""

    out_plot = audeer.mkdir(f"{output_dir}/plot_significant_features")
    try:
        original_df = pd.read_csv(
            f"/home/mgonzalez/project/tum/als/MRI_ALS/DB/FeatureRepresentations/{featureset}_{aggregate}/1106605/20240415/20240415_Ah_1106605.csv"
        )
        feature_names = original_df.columns
        feature_df = pd.DataFrame(feature_names, columns=["Feature"])

        feature_df.to_csv(
            f"/home/mgonzalez/project/tum/als/MRI_ALS/FeatureResults/feature_list_{featureset}_{aggregate}.csv",
            index=True,
        )

    except:
        original_df = pd.read_csv(
            f"/home/mgonzalez/project/tum/als/MRI_ALS/DB/FeatureRepresentations/{featureset}/1106605/20240415/20240415_Ah_1106605.csv"
        )
        feature_names = original_df.columns
        feature_df = pd.DataFrame(feature_names, columns=["Feature"])

        feature_df.to_csv(
            f"/home/mgonzalez/project/tum/als/MRI_ALS/FeatureResults/feature_list_{featureset}.csv",
            index=True,
        )

    features, labels = [], []
    for i in range(len(df)):
        feature, label, _ = df[i]
        features.append(feature.numpy())

        labels.append(label)

    df = pd.DataFrame([tensor.flatten() for tensor in features], columns=feature_names)

    df["labels"] = labels

    if disease == "ALS":
        labels_dict = {1: "ALS", 0: "control"}
    elif disease == " SMA":
        labels_dict = {1: "SMA", 0: "control"}
    df["labels"] = df["labels"].map(labels_dict)

    for f in sig_feature:
        feature_name = feature_names[f]

        box = sns.boxplot(x="labels", y=feature_name, data=df, showfliers=False)
        box.set_xlabel("label", fontsize=14)
        box.set_ylabel(feature_name, fontsize=14)
        plt.title(f"{task} {featureset} {speech_task}")
        for cur_extension in [".png", ".pdf", ".svg"]:
            plt.savefig(
                f"{out_plot}/{task}_correlation_feat{feature_name}_{featureset}{aggregate}{disease}{speech_task}{cur_extension}",
                bbox_inches="tight",
            )
        plt.close()

    # check redo analysis to save the original feature name


def plot_qol(df, featureset, aggregate, speech_task, output_dir):

    sessions, scores, features, patients = [], [], [], []
    for i in range(len(df)):
        feature, label, samples, session_date, session_type, patient = df[i]
        sessions.append(session_type)
        scores.append(int(label))
        patients.append(patient)
        features.append(feature.numpy())

    print(scores)
    print(sessions)

    try:
        feature_df = pd.read_csv(
            f"/home/mgonzalez/project/tum/als/MRI_ALS/FeatureResults/feature_list_{featureset}_{aggregate}.csv",
        )

    except:

        feature_df = pd.read_csv(
            f"/home/mgonzalez/project/tum/als/MRI_ALS/FeatureResults/feature_list_{featureset}.csv",
        )
    feature_name = feature_df["Feature"].to_list()

    df_features = pd.DataFrame(
        [tensor.flatten() for tensor in features], columns=feature_name
    )
    print(df_features)

    metadata = pd.DataFrame(
        {
            "session_type": sessions,
            "qold_score": scores,
            "patient_id": patients,
        }
    )
    data = pd.concat([metadata, df_features.reset_index(drop=True)], axis=1)

    data = data.sort_values(by="qold_score").reset_index(drop=True)
    patients_with_baseline_fu1 = data[data["session_type"].isin(["Baseline", "FU1"])]
    patients_to_keep = patients_with_baseline_fu1.groupby("patient_id")[
        "session_type"
    ].nunique()
    valid_patients = patients_to_keep[patients_to_keep == 2].index

    # Filter the DataFrame to include only valid patients
    filtered_df = data[data["patient_id"].isin(valid_patients)]

    # plot sessions, scores

    sns.swarmplot(
        data=filtered_df,
        x="session_type",
        y="qold_score",
        hue="patient_id",
        palette="deep",
        # height=6,
        # aspect=1.5,
    )

    plt.title(f"QOL-Dys questionnaire {featureset} {speech_task}")
    for cur_extension in [".png", ".pdf", ".svg"]:
        plt.savefig(
            f"{output_dir}/sessions_qol_dysarthria{speech_task}{cur_extension}",
            bbox_inches="tight",
        )
    plt.close()

    # plot session, feature score
    for feature_name in df_features.columns:

        sns.swarmplot(
            data=filtered_df,
            x="session_type",
            y=feature_name,
            hue="patient_id",
            palette="deep",
            size=8,
        )

        for cur_extension in [".png", ".pdf", ".svg"]:
            plt.savefig(
                f"{output_dir}/{feature_name}_qol_dysarthria{speech_task}{cur_extension}",
                bbox_inches="tight",
            )
            plt.close()
    return
