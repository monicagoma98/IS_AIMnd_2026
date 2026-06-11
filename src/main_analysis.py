import audeer
import pandas as pd
from numpy.random import seed

import argparse
import os

# from dataset_MRIneurology import (
#    MRIneurology,
#    MRIneurology_ALSFRSbased,
#    MRIneurology_QOL_DYS_G,
# )

from feature_extractor import FeatureExtractor

# from dataset_MRIneurology import ALS_Prediction

import split_data
from utils import plot_qol

from feature_analysis import feature_analysis_per_sex
from modelling import main_supervised_modeling
import config

# prepare metadata
# generate VAD + audio normalisation, remove background noise
# split metadata: per session and per speaker, for longitudinal, train-dev-test splits

# extract features: oS, phoneme-based, articulatory precision, whisper, wav2vec, clip, textual, glottal + tongue + lip features

# get feature distributions: to sanity check


# machine learning (alexandra does only oS, whisper and wav2vec), predicts alsfrs
# i do: longitudinal analysis using multimodality

# two paper ideas: alsfrs predictions using all features
# longitudinal analysis of speech changes in relation to alsfrs decline


def main(args):
    """
    Performs:
    - creates data split
    - feature analysis
    - machine learning
    - explainability analysis
    Requires:
    - metadata ready, features extracted
    """

    base_path = os.path.abspath("../ALS_results")
    audeer.mkdir(base_path)

    seed(args.seed)

    speech_tasks = ["Ah", "CookieTheft", "DaBa", "DaDa", "NordWindSonne"]
    featureset = args.features

    # load metadata
    metadata_file = "../AI_MND_dataset_ID_Schuller.xlsx"  # to add

    merged_metadata = pd.read_excel(metadata_file, engine="openpyxl")
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
        "Clinical_Onset"
    ]

    merged_metadata = merged_metadata[relevant_variables_metadata]

    # drop all nan in session
    merged_metadata = merged_metadata[merged_metadata["session"].notna()]
    # Convert 'session' in merged_metadata from 'YYYY-MM-DD' to 'YYYYMMDD' string format
    merged_metadata["session"] = (
        pd.to_datetime(merged_metadata["session"]).dt.strftime("%Y%m%d").astype(int)
    )

    merged_metadata["IDs"] = merged_metadata["IDs"].astype(str)
    merged_metadata["age"] = merged_metadata["age"].astype("Int64")

    # create split based on IDs: create train
    if args.task == "DiseaseDetection":
        train_df, test_df = split_data.split_diagnosis(merged_metadata)
    elif args.task == "DiseaseMonitoring":
        train_df, test_df = split_data.split_forecasting(merged_metadata)

    if args.task == "DiseaseMonitoring":
        target = "ALSFRS_R_Complete"
        #"ALSFRS_R_Complete"
    elif args.task == "DiseaseDetection":
        target = "Diagnosis"

    train_df.to_csv(f"{base_path}/train_metadata.csv", index=False)
    test_df.to_csv(f"{base_path}/test_metadata.csv", index=False)

    # load features
    output_features = "../features/"

    for speech_task in speech_tasks:

        extractor = FeatureExtractor(0, speech_task, featureset, output_features)
        df_feat = extractor._load_features()
        if "file" in df_feat.columns and "start" in df_feat.columns and "end" in df_feat.columns:
            df_feat.set_index(["file","start","end"], inplace=True)
        #input(df_feat)
        if df_feat is not None and not df_feat.empty:
            # print(df_feat)
            feat_columns = df_feat.columns
            #input(feat_columns)

            # extract ID and session from filename
            df_feat.reset_index(inplace=True)
            df_feat["IDs"] = df_feat["file"].apply(
                lambda x: os.path.basename(x).split("_")[-1].split(".wav")[0]
            )
            df_feat["session"] = df_feat["file"].apply(
                lambda x: os.path.basename(x).split("_")[0]
            )
            df_feat["session"] = df_feat["session"].astype(int)

            # match train_df with df_feat
            df_train_feat = df_feat.merge(train_df, on=["IDs", "session"])
            df_test_feat = df_feat.merge(test_df, on=["IDs", "session"])

            if "start" in df_feat.columns and "end" in df_feat.columns:
                df_feat.set_index(["file", "start", "end"], inplace=True)
            else:
                df_feat.set_index("file", inplace=True)

            # input(df_train_feat)
            # do feature analysis on train: only for interpretable features
            if featureset in ("eGeMAPSv02_functionals", "rhythmic"):
                # target variable ALSFRS_R_Complete, QOL_DYS_G, account for sex
                output_path_feat_analysis = audeer.mkdir(
                    f"{base_path}/feature_analysis/{args.task}{featureset}{speech_task}/"
                )
                feature_analysis_per_sex(
                    df=df_train_feat,
                    feat_list=feat_columns,
                    speech_task=speech_task,
                    target=target,
                    odir=output_path_feat_analysis,
                    corr_threshold=0.20,
                    alpha=0.05,
                )

                feature_analysis_per_sex(
                    df=df_train_feat,
                    feat_list=feat_columns,
                    speech_task=speech_task,
                    target="QOL_DYS_G",
                    odir=output_path_feat_analysis,
                    corr_threshold=0.20,
                    alpha=0.05,
                )
            else:
                print(
                    f"Skipping feature analysis for {featureset}, not interpretable features."
                )

            if args.task == "DiseaseMonitoring":
                normalisation = args.normalisation
                print(df_train_feat)
                print(df_test_feat)
                main_supervised_modeling(
                    train=df_train_feat,
                    test=df_test_feat,
                    X_col=feat_columns,
                    target=target,
                    save_dir=audeer.mkdir(
                        f"{base_path}/modelling/{args.task}_{featureset}_{speech_task}_{normalisation}_{args.seed}/"
                    ),
                    n_jobs=1,
                    seed=args.seed,
                    normalisation=normalisation,
                )

        # if task disease monitoring:
        # use these features to predict the next session ALSFRS_R_Complete or QOL_DYS_G
        # first we use simple ML model(svm, xgb, mlp), then we add xlstm, we do interpretability

        # elif task disease detection:
        # predict healthy vs disease, then cascade predict ALSFRS_R_Complete between speakers with ALS, SMA or PMA
        # first we use simple ML models, then we add FFNN, we do interpretability

    # load segments + files
    # df_all_files = pd.read_csv(f"{base_path}/all_files_metadata.csv")
    # df_all_segments = pd.read_csv(f"{base_path}/all_segments_metadata.csv")

    stages_conf = {
        "DEVEL": {"train": ["Train"], "test": ["Devel"]},
        "TEST": {"train": ["Train", "Devel"], "test": ["Test"]},
    }

    data_als = ALS_Prediction(args.features, config.speechTask, args.predictionTask)

    for _, stage in enumerate(stages_conf.keys()):

        print("Stating {} phase...".format(stage))

        DB_train = data_als.get_dataset(stages_conf[stage]["train"])

        plot_qol(
            DB_train,
            args.features,
            args.aggregate,
            config.speechTasks,
            output_qoldys,
        )
        # plot questionnaire analysis

        # feature analysis per speech task
        if stage == "DEVEL":
            feature_analysis(
                DB_train,
                output_path,
                args.task,
                args.features,
                args.aggregate,
                args.disease,
                config.speechTasks,
                alpha=0.05,
            )

        if stage == "TEST":
            # shap analysis
            if config.speechTasks == "all":
                results_dir = f"/home/mgonzalez/project/tum/als/MRI_ALS/ExperimentalResults/Task_{args.task}/Model_SVClinear_MinMaxScaler/"
            else:
                results_dir = f"/home/mgonzalez/project/tum/als/MRI_ALS/ExperimentalResults/Task_{args.task}/PerTaskModel_SVClinear_MinMaxScaler/"
            if args.features == "phoneticFeatures":
                feature_name = args.features
            else:
                feature_name = "".join(args.features + f"_{args.aggregate}")
            output_shap = audeer.mkdir(f"{out_shap}/{args.task}_{config.speechTasks}")
            shap_analysis(
                feature_name,
                DB_train,
                DB_test,
                config.speechTasks,
                output_shap,
                results_dir,
            )
            output_lime = audeer.mkdir(f"{out_lime}/{args.task}_{config.speechTasks}")
            class_names = ["control", "ALS"]
            lime_analysis(
                DB_train,
                DB_test,
                feature_name,
                class_names,
                config.speechTasks,
                results_dir,
                output_lime,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="DiseaseMonitoring",
        choices=[
            "DiseaseDetection",
            "ALSFRSspeechAnalysis",
            "ALSFRS4vsControlSpeechAnalysis",
            "QOLDetection",
        ],
    )
    parser.add_argument(
        "--features",
        "-f",
        type=str,
        default="eGeMAPSv02_functionals",
        choices=[
            "rhythmic",
            "eGeMAPSv02_functionals",
            "whisper_embeddings",
            "wav2vec_embeddings",
            "bert",
            "artp",
            "clip",
        ],
    )

    parser.add_argument(
        "--normalisation",
        "-norm",
        type=str,
        default="standard",
        choices=["standard", "robust"],
    )

    args = parser.parse_args()

    args.seed = 42
    main(args)


# example call: python main.py -d ALS -t DiseaseDetection -f artp -a meanANDstd -m SVClinear_MinMaxScaler
