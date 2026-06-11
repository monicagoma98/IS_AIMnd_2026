from feature_extractor import FeatureExtractor

import utils as ut

import pandas as pd

import os

import split_data


def main(input_folder):  # task, featureset):
    """Prepare data for analysis:
    This script performs:
    - Processing audio files (VAD, normalisation, noise removal)
    - Finding audio files and linking them to metadata
    - Splitting data into train and test sets at speaker level
    - Extracting features from audio files"""

    print("Starting data preparation...")

    output_processed_files = "/media/mgonzalez/Elements/CHI_2025/processed_audio/"
    output_dir = "/media/mgonzalez/Elements/CHI_2025/"

    speech_tasks = [
        "Ah",
        "CookieTheft",
        "DaBa",
        "DaDa",
        "NordWindSonne",
        "Spontansprache",
    ]
    features = ["os", "rhythmic"]  # , "wav2vec", "bert"]
    # features = ["wav2vec"]

    old_csv_filename = f"{output_dir}/all_files_metadata.csv"

    # step2: noise reduction
    ut.process_audio_files(
        input_folder=input_folder,
        output_folder=output_processed_files,
    )

    # step1:create df: file, speaker id, session, speech task, speech task rep
    df_files = ut.find_wav_files(output_processed_files)
    print(
        f"Audio files dataframe contains {df_files['IDs'].nunique()} unique speakers and {df_files[['IDs','session']].drop_duplicates().shape[0]} unique sessions."
    )

    # input("Audio processing completed. Press Enter to continue to metadata merging and VAD...")

    # load metadata file
    # merge segments + files with metadata
    metadata_file = "AI-MND_Schuller_(10.05.2026)_updated.xlsx"  # to add
    merged_metadata = pd.read_excel(
        os.path.join(output_dir, metadata_file), engine="openpyxl"
    )
    print(f"Loaded metadata with {len(merged_metadata)}")
    # rename columns if needed
    merged_metadata = merged_metadata.rename(
        columns={
            "Age_Examination": "age",
            "Sex": "sex",
            "Date_Examination": "session",
        }
    )
    relevant_variables_metadata = [
        "IDs",
        "Diagnosis",
        "age",
        "sex",
        "session",
        "ALSFRS_R_Complete",
        "ALSFRS_R_Speech",
        "ALSFRS_R_Swallowing",
        "ALSFRS_R_Salivation",
        "ALSFRS_R_Bulbar",
        "QOL_DYS_G",
        "Mother_Language",
        "Therapy",
        "Other_Diagnosis_Neuro_Or_Psych",
    ]
    merged_metadata = merged_metadata[relevant_variables_metadata]
    # drop all nan in session
    merged_metadata = merged_metadata[merged_metadata["session"].notna()]

    # Convert 'session' in merged_metadata from 'DD.MM.YYYY' to 'YYYYMMDD' string format
    merged_metadata["session"] = (
        pd.to_datetime(merged_metadata["session"], dayfirst=True)
        .dt.strftime("%Y%m%d")
        .astype(int)
    )
    merged_metadata["IDs"] = merged_metadata["IDs"].astype(str)
    # check number of unique speakers and number of unique sessions
    print(
        f"Metadata contains {merged_metadata['IDs'].nunique()} unique speakers and {len(merged_metadata)} unique sessions."
    )
    ####check######
    df_files["IDs"] = df_files["IDs"].astype(str)
    df_files["session"] = df_files["session"].astype(int)
    print(df_files["IDs"].nunique())
    # check number of unique IDs and sessions in df_files
    print(
        f"Audio files dataframe contains {df_files['IDs'].nunique()} unique speakers and {df_files[['IDs','session']].drop_duplicates().shape[0]} unique sessions."
    )
    # merge with metadata
    df_all_files = df_files.merge(merged_metadata, on=["IDs", "session"])
    df_all_files = df_all_files[df_all_files["speech_task"].isin(speech_tasks)]
    print(
        f"After merging with metadata, {len(df_all_files)} files remain for processing."
    )
    # check number of unique speakers and sessions after merge
    print(
        f"After merging with metadata, {df_all_files['IDs'].nunique()} unique speakers and {df_all_files[['IDs','session']].drop_duplicates().shape[0]} unique sessions remain."
    )

    if not os.path.exists(old_csv_filename):

        # step3: apply vad per file
        df_segments = split_data.get_vad(df_files, output_dir, output_processed_files)

        df_segments["IDs"] = df_segments["IDs"].astype(str)
        df_segments["session"] = df_segments["session"].astype(int)

        df_all_segments = df_segments.merge(merged_metadata, on=["IDs", "session"])

        df_all_segments = df_all_segments[
            df_all_segments["speech_task"].isin(speech_tasks)
        ]

        df_all_files.to_csv(f"{output_dir}/all_files_metadata.csv", index=False)
        df_all_segments.to_csv(f"{output_dir}/all_segments_metadata.csv", index=False)

    elif os.path.exists(old_csv_filename):

        old_csv = pd.read_csv(old_csv_filename)
        input(old_csv)

        old_idx = pd.MultiIndex.from_frame(old_csv[["IDs", "session"]])
        print(len(df_all_files))
        out = []
        for df in [df_all_files]:
            # df = df.to_frame().T
            keep = ~pd.MultiIndex.from_frame(df[["IDs", "session"]]).isin(old_idx)
            out.append(df.loc[keep])

        df_cleaned = out[0]

        if df_cleaned.empty:
            print("No new files to process.")
        else:
            new_segments_filename = f"{output_dir}/vad_segments_missingfiles.csv"
            if not os.path.exists(new_segments_filename):
                print("Creating new VAD segments file for missing files...")

                def _drop_artifact_cols(df):
                    return df.drop(
                        columns=[
                            c
                            for c in df.columns
                            if c == "level_0" or c.startswith("Unnamed:")
                        ],
                        errors="ignore",
                    )

                # Clean known artifact columns before processing
                df_cleaned = _drop_artifact_cols(df_cleaned)
                merged_metadata = _drop_artifact_cols(merged_metadata)
                old_csv = _drop_artifact_cols(old_csv)

                # Basic schema checks
                required_base = {"IDs", "session"}
                if not required_base.issubset(df_cleaned.columns):
                    raise ValueError(
                        f"df_cleaned missing required columns: {required_base - set(df_cleaned.columns)}"
                    )
                if not required_base.issubset(merged_metadata.columns):
                    raise ValueError(
                        f"merged_metadata missing required columns: {required_base - set(merged_metadata.columns)}"
                    )

                df_segments_missing = split_data.get_vad(
                    df_cleaned, output_dir, output_processed_files
                )
                if df_segments_missing is None or df_segments_missing.empty:
                    print("No VAD segments generated for missing files.")
                    df_segments_missing = pd.DataFrame(
                        columns=["IDs", "session", "speech_task"]
                    )

                df_segments_missing = _drop_artifact_cols(df_segments_missing)

                if not required_base.issubset(df_segments_missing.columns):
                    raise ValueError(
                        f"VAD output missing required columns: {required_base - set(df_segments_missing.columns)}"
                    )

                # Normalize dtypes and drop invalid rows
                df_segments_missing["IDs"] = df_segments_missing["IDs"].astype(str)
                df_segments_missing["session"] = pd.to_numeric(
                    df_segments_missing["session"], errors="coerce"
                )
                bad_session = df_segments_missing["session"].isna().sum()
                if bad_session:
                    print(f"Dropping {bad_session} VAD rows with invalid session.")
                df_segments_missing = df_segments_missing.dropna(subset=["session"])
                df_segments_missing["session"] = df_segments_missing["session"].astype(
                    int
                )

                merged_metadata["IDs"] = merged_metadata["IDs"].astype(str)
                merged_metadata["session"] = pd.to_numeric(
                    merged_metadata["session"], errors="coerce"
                )
                merged_metadata = merged_metadata.dropna(subset=["session"])
                merged_metadata["session"] = merged_metadata["session"].astype(int)

                df_all_segments = df_segments_missing.merge(
                    merged_metadata,
                    on=["IDs", "session"],
                    how="left",
                    validate="many_to_one",
                    suffixes=("", "_x"),
                )
                df_all_segments = df_all_segments[
                    [c for c in df_all_segments.columns if not c.endswith("_x")]
                ]
                df_all_segments = _drop_artifact_cols(df_all_segments)

                if "speech_task" in df_all_segments.columns:
                    df_all_segments = df_all_segments[
                        df_all_segments["speech_task"].isin(speech_tasks)
                    ]

                # Sanity check: confirm every missing file key appears in generated segments
                expected_keys = set(
                    df_cleaned[["IDs", "session"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                )
                got_keys = set(
                    df_all_segments[["IDs", "session"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                )
                missing_keys = expected_keys - got_keys
                if missing_keys:
                    print(
                        f"Warning: {len(missing_keys)} file keys still missing after VAD/merge."
                    )

                old_segments_csv = pd.read_csv(
                    f"{output_dir}/all_segments_metadata.csv"
                )
                old_segments_csv = _drop_artifact_cols(old_segments_csv)

                df_total = pd.concat([old_csv, df_cleaned], ignore_index=True)
                df_total = _drop_artifact_cols(df_total).drop_duplicates()

                df_total_segments = pd.concat(
                    [old_segments_csv, df_all_segments], ignore_index=True
                )
                df_total_segments = _drop_artifact_cols(
                    df_total_segments
                ).drop_duplicates()

                # Final guard: prevent artifact columns from being saved
                bad_cols_total = [
                    c
                    for c in df_total.columns
                    if c == "level_0" or c.startswith("Unnamed:")
                ]
                bad_cols_seg = [
                    c
                    for c in df_total_segments.columns
                    if c == "level_0" or c.startswith("Unnamed:")
                ]
                if bad_cols_total or bad_cols_seg:
                    raise ValueError(
                        f"Artifact columns detected before save: {bad_cols_total + bad_cols_seg}"
                    )

                df_total_segments.set_index(["file", "start", "end"], inplace=True)
                df_total.set_index(["file"], inplace=True)
                df_total.to_csv(
                    f"{output_dir}/all_files_metadata_missingfiles.csv", index=False
                )
                df_total_segments.to_csv(
                    f"{output_dir}/all_segments_metadata_missingfiles.csv", index=False
                )
            else:
                print("VAD segments file for missing files already exists. Loading...")
                df_total = pd.read_csv(
                    f"{output_dir}/all_files_metadata_missingfiles.csv"
                )
                df_total_segments = pd.read_csv(
                    f"{output_dir}/all_segments_metadata_missingfiles.csv"
                )

    # once segments: features oS + glottal + phoneme + clip + articulatory precission + whisper + wav2vec + textual features
    # input(df_total_segments.columns)
    # input(df_total)

    # extract features
    output_features = os.path.join(output_dir, "features/")
    for feat in features:
        for speech_task in speech_tasks:
            print(f"Extracting {feat} features for task {speech_task}...")
            extractor = FeatureExtractor(0, speech_task, feat, output_features)
            df_feat, feat_columns = extractor._extract_features(
                df_all_files, df_all_segments
            )


if __name__ == "__main__":

    input_folder = "/media/mgonzalez/Elements/Videos_Schuller/"

    main(input_folder)
