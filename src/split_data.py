import pandas as pd

from sklearn.model_selection import train_test_split

from pyannote.audio import Pipeline

import torch

from tqdm import tqdm

import os

from datetime import timedelta

import matplotlib.pyplot as plt


def get_vad(df, output_dir, audio_dir):

    if os.path.exists(f"{output_dir}/vad_segments.csv"):
        print("VAD segments file already exists. Loading...")
        segments_df = pd.read_csv(f"{output_dir}/vad_segments.csv")

    else:
        HF_TOKEN = "ADD_TOKEN"

        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=HF_TOKEN,
            # revision="main"
        )
        pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"Running VAD on {len(df)} recordings...")

        segment_rows = []

        for file_name in tqdm(df["file"], total=len(df)):
            print(file_name)

            if not os.path.exists(str(file_name)):
                print(f"Warning: File not found {file_name}")
                continue

            # Run VAD
            vad_result = pipeline(str(file_name))

            for turn, _, _ in vad_result.itertracks(yield_label=True):
                start = turn.start
                end = turn.end
                duration = end - start

                if duration < 0.3:  # 300 ms
                    continue

                segment_rows.append(
                    {
                        "file": str(file_name),
                        "start": timedelta(seconds=start),
                        "end": timedelta(seconds=end),
                        "duration": round(duration, 3),
                    }
                )

        segments_df = pd.DataFrame(segment_rows)

        segments_df = segments_df.merge(df, on=["file"])
        # Save
        segments_df.to_csv(f"{output_dir}/vad_segments.csv", index=False)

    return segments_df


def split_diagnosis(df):

    bins = [0, 30, 50, float("inf")]
    labels = ["young", "middle", "old"]
    df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels)

    # Step 2: Separate into healthy and diagnosis groups
    healthy_df = df[df["Diagnosis"] == "control"].copy()
    diagnosis_df = df[df["Diagnosis"] == "ALS"].copy()

    # Step 3: Split each group
    healthy_train, healthy_test = stratified_speaker_split(healthy_df)
    diagnosis_train, diagnosis_test = stratified_speaker_split(diagnosis_df)

    # Step 4: Recombine into overall train and test
    train_df = pd.concat([healthy_train, diagnosis_train], ignore_index=True)
    test_df = pd.concat([healthy_test, diagnosis_test], ignore_index=True)

    # Optional: Drop temporary columns if needed
    train_df = train_df.drop(columns=["age_bin"])
    test_df = test_df.drop(columns=["age_bin"])

    # Verify balances (optional, for debugging)
    print("Overall diagnosis balance:")
    print("Train:", train_df["diagnosis"].value_counts(normalize=True))
    print("Test:", test_df["diagnosis"].value_counts(normalize=True))

    print("\nWithin healthy - gender balance:")
    print("Healthy Train:", healthy_train["sex"].value_counts(normalize=True))
    print("Healthy Test:", healthy_test["sex"].value_counts(normalize=True))

    return train_df, test_df


def stratified_speaker_split(group_df, test_size=0.2, random_state=42):
    input(group_df)
    # Get unique speakers with their fixed attributes
    speakers_df = group_df[["IDs", "sex", "age_bin"]].drop_duplicates()

    # Create combined strata for balancing (gender + age_bin)
    speakers_df["strata"] = (
        speakers_df["sex"] + "_" + speakers_df["age_bin"].astype(str)
    )

    # Split speakers, stratifying on the combined strata
    train_speakers, test_speakers = train_test_split(
        speakers_df["IDs"],
        test_size=test_size,
        stratify=speakers_df["strata"],
        random_state=random_state,
    )

    # Map back to full group data (all sessions per speaker)
    train_group = group_df[group_df["speaker_id"].isin(train_speakers)]
    test_group = group_df[group_df["speaker_id"].isin(test_speakers)]

    return train_group, test_group


def split_forecasting(df_2split):
    """
    Speaker-dependent forecasting split:
    - For each speaker:
        - First session -> train
        - Remaining sessions -> test
    """

    target_cols = [
        "ALSFRS_R_Complete",
        "QOL_DYS_G",
    ]
    valid_ids = df_2split.groupby("IDs")[list(target_cols)].apply(
        lambda x: x.notna().all().all()
    )

    valid_ids = valid_ids[valid_ids].index

    filtered_df = df_2split[df_2split["IDs"].isin(valid_ids)].copy()

    print(
        f"Keeping {len(valid_ids)} / {df_2split['IDs'].nunique()} speakers "
        f"with complete targets: {target_cols}"
    )

    # Step 1: Ensure sessions are ordered per speaker
    df = filtered_df.sort_values(["IDs", "session"]).reset_index(drop=True)

    train_list = []
    test_list = []

    # Step 2: Split per speaker
    for speaker_id, speaker_data in df.groupby("IDs"):
        speaker_data = speaker_data.sort_values("session")

        if speaker_data.empty:
            continue

        # First session -> train
        train_list.append(speaker_data.iloc[[0]])

        # Remaining sessions -> test (if any)
        if len(speaker_data) > 1:
            test_list.append(speaker_data.iloc[1:])

    # Step 3: Concatenate
    train_df = (
        pd.concat(train_list, ignore_index=True)
        if train_list
        else pd.DataFrame(columns=df.columns)
    )
    test_df = (
        pd.concat(test_list, ignore_index=True)
        if test_list
        else pd.DataFrame(columns=df.columns)
    )

    # =====================
    # Distribution summaries
    # =====================
    print("=== Forecasting Split Summary ===")
    print(f"Train: {len(train_df)} sessions from {train_df['IDs'].nunique()} speakers")
    print(f"Test:  {len(test_df)} sessions from {test_df['IDs'].nunique()} speakers")

    print("\nSessions per speaker (TRAIN):")
    print(train_df.groupby("IDs")["session"].count().value_counts().sort_index())

    print("\nSessions per speaker (TEST):")
    print(test_df.groupby("IDs")["session"].count().value_counts().sort_index())

    # Optional demographic checks
    for col in ["age", "sex", "Diagnosis", "Therapy"]:
        if col in df.columns:
            print(f"\n{col.upper()} distribution:")
            print(
                "Train:",
                train_df[col].value_counts(normalize=True).round(3).to_dict(),
            )
            print(
                "Test: ",
                test_df[col].value_counts(normalize=True).round(3).to_dict(),
            )
    
    # Numerical distributions
    for col in ["ALSFRS_R_Complete", "QOL_DYS_G"]:
        if col in df.columns:
            print(f"\n{col} distribution (mean ± std):")
            print(
                f"Train: {train_df[col].mean():.2f} ± {train_df[col].std():.2f}"
            )
            print(
                f"Test:  {test_df[col].mean():.2f} ± {test_df[col].std():.2f}"
            )
                
   
    return train_df, test_df
