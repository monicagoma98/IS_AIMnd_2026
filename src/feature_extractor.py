import os
from collections import defaultdict

import utils as ut

import pandas as pd

from tqdm import tqdm
import pickle

from opensmile import FeatureLevel, FeatureSet, Smile
from syllabifier import SyllableNucleiExtractor
import parselmouth
import torch
import torchaudio

from transformers import WhisperProcessor, WhisperModel
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class FeatureExtractor:
    def __init__(self, device, speech_task, featureset, output):
        self.device = device
        self.task = speech_task
        self.features_type = featureset
        self.output = output

    def _load_features(self):

        #filter for ah for rhythmic features
        df_feat = os.path.join(self.output, f"{self.task}_{self.features_type}.pkl")
        if os.path.exists(df_feat):
            print(f"Loading features from {df_feat}...")
            df_feat = pd.read_pickle(df_feat)
    
        else:
            print(f"No precomputed features found at {df_feat}.")
            df_feat = None

        return df_feat

    def _extract_features(
        self,
        df_files: pd.DataFrame,
        df_segmented: pd.DataFrame,
    ):
        """
        Description: extracts corresponding features and loads them if they already exist.
        Args:
            db(pd.DataFrame): data for processing
            features_type(str): name of the feature set to extract
            target(str): name of the target column
            output(str): path where to save all the features
            number_variance(float):number of variance to keep for pca
        Returns:
            df_all
            feat_columns
        """

        df_feat, feat_columns = None, None

        df_files = df_files[df_files["speech_task"] == self.task]
        df_segmented = df_segmented[df_segmented["speech_task"] == self.task]
        # input(df_segmented)

        # features oS + rhythmic + clip + articulatory precission + whisper + wav2vec + bert features
        # glottal features: excluded for now

        if self.features_type == "rhythmic":
            if self.task == "CookieTheft" or self.task == "NordWindSonne" or self.task == "Spontansprache":
                print(f"Extracting rhythmic features for task {self.task}")
                df_feat = self.extract_syllable_nuclei(df_files)

        elif self.features_type == "os":
            # for opensmile we extract functionals for all tasks
            print(f"Extracting opensmile features for task {self.task}")
            df_feat = self.extract_os(df_segmented)

        # elif self.features_type == "glottal":
        #    df_feat, feat_columns = self.glottal_features(df_files)

        elif self.features_type == "whisper":
            # for whisper we extract embeddings for all tasks
            df_feat, feat_columns = self._extract_whisper(df_segmented)

        elif self.features_type == "wav2vec":
            # for wav2vec we extract all embeddings for all tasks
            df_feat, feat_columns = self.extract_wav2vec(df_segmented)

        elif self.features_type == "bert":
            if self.task == "CookieTheft":
                df_feat, feat_columns = self._extract_bert(df_files)

        elif self.features_type == "clip":
            df_feat, feat_columns = self.clip_embeddings(df_files)

        elif self.features_type == "articulatory_precission":
            df_feat, feat_columns = self.articulatory_precission_features(df_files)

        # if not os.path.exists(files4features):
        #    if "start" in df_feat.columns:
        #        frame = db.merge(df_feat, on=["file", "start", "end"])
        #    else:
        #        frame = db.merge(df_feat, on=["file"])
        #
        #    self.check_file_percentage(frame, db, df_feat)
        #    df_all = self.save_processed_data(
        #        frame, feat_columns, files4features, file_feature_names
        #    )
        # else:
        #    print(f"Data is already processed and save in {files4features}")

        return df_feat, feat_columns

    def extract_syllable_nuclei(self, db: pd.DataFrame) -> pd.DataFrame:
        """
        Description: extract syllable nuclei for feature extraction

        Args:
            db (pd.DataFrame): database(requires a file column)
            output_dir(str):where to store the extended dataframe
        Returns:
            pd.DataFrame:
        """

        if "file" not in db.columns:
            db.reset_index(inplace=True)

        path = os.path.join(
            self.task, self.output, f"{self.task}_{self.features_type}.pkl"
        )

        if not os.path.exists(path):
            nuclei_extractor = SyllableNucleiExtractor(
                silencedb=-25, mindip=2, minpause=0.3
            )
            # syllable feature extraction is performed at file-level
            df_syllabifier = []
            for sound in tqdm(db["file"]):
                s = parselmouth.Sound(sound)
                speech_rate_dict = nuclei_extractor.syllable_nuclei(s)
                speech_rate_dict["file"] = sound
                df_syllabifier.append(speech_rate_dict)

            df_feat = pd.DataFrame(df_syllabifier)
            df_feat.to_pickle(path)
        else:
            df_feat = pd.read_pickle(path)

            # Check if all files in db are in df_feat
            db_files = set(db["file"].values)
            feat_files = set(df_feat["file"].values)
            missing_files = db_files - feat_files

            if missing_files:
                print(f"Extracting features for {len(missing_files)} missing files...")
                nuclei_extractor = SyllableNucleiExtractor(
                    silencedb=-25, mindip=2, minpause=0.3
                )
                df_syllabifier = []
                for sound in tqdm(missing_files):
                    s = parselmouth.Sound(sound)
                    speech_rate_dict = nuclei_extractor.syllable_nuclei(s)
                    speech_rate_dict["file"] = sound
                    df_syllabifier.append(speech_rate_dict)

                df_missing = pd.DataFrame(df_syllabifier)
                df_feat = pd.concat([df_feat, df_missing], ignore_index=True)
                df_feat.to_pickle(path)
                print(f"Updated features saved to {path}")
            else:
                print(f"All files already have extracted features")

        return df_feat

    def extract_os(self, db: pd.DataFrame) -> pd.DataFrame:
        """
        Description: extract eGeMAPS features from the openSMILE feature extractor.
        Args:
            db(pd.DataFrame):dataframe from where the features are extracted. It should contain a column called "file"
            output_dir(str): path to where the extracted features are stored
        Returns:
            df_feat(pd.DataFrame): dataframe containing "file","start", "end", and feature columns
            feat_columns(list): list of the feature names available in "df_feat"
        """
        feature_set = "eGeMAPSv02"

        if "start" not in db.index.names and "start" in db.columns:
            db.reset_index(inplace=True)
            # input(db.columns)
            db["start"] = pd.to_timedelta(db["start"])
            db["end"] = pd.to_timedelta(db["end"])
            db["start"] = db["start"].dt.total_seconds()
            db["end"] = db["end"].dt.total_seconds()
            db.set_index(["file", "start", "end"], inplace=True)

        #input(db)
        path = os.path.join(self.output, f"{self.task}_{feature_set}_functionals.pkl")

        if not os.path.exists(path):
            smile = Smile(
                feature_set=getattr(FeatureSet, feature_set),
                feature_level=FeatureLevel.Functionals,
                num_workers=16,
                verbose=True,
                logfile="log",
            )

            df_feat_functionals = smile.process_index(index=db.index)

            df_feat_functionals.to_pickle(path)
            print(f"Features are extracted and saved to:", path)
            assert len(db) == len(df_feat_functionals)
        else:
            print(f"Features are already extracted and saved to:", path)
            df_feat_functionals = pd.read_pickle(path)
            df_feat_functionals.reset_index(inplace=True)
            db = db.reset_index()
            # Check if all files in db are in df_feat
            db_files = set(db["file"].values)
            feat_files = set(df_feat_functionals["file"].values)
            missing_files = db_files - feat_files

            if missing_files:
                print(f"Extracting features for {len(missing_files)} missing files...")
                
                # Filter db to only include missing files
                db_missing = db[db["file"].isin(missing_files)]
                db_missing.set_index(["file", "start", "end"], inplace=True)
                
                smile = Smile(
                    feature_set=getattr(FeatureSet, feature_set),
                    feature_level=FeatureLevel.Functionals,
                    num_workers=16,
                    verbose=True,
                    logfile="log",
                )

                df_missing = smile.process_index(index=db_missing.index)
                               
                if isinstance(df_missing, pd.Series):
                        df_missing = df_missing.to_frame()
                df_missing = df_missing.reset_index()  # <- critical: ensures file/start/end are columns

                df_feat = pd.concat([df_feat_functionals, df_missing], ignore_index=True)
                df_feat.to_pickle(path)
                print(f"Updated features saved to {path}")
            else:
                print(f"All files already have extracted features")

        return df_feat_functionals

    def _extract_whisper(self, db):

        path = os.path.join(self.output, f"{self.task}_whisper_embeddings.pkl")

        if "start" not in db.columns:
            db = db.reset_index()

        db_process = ut.dt2sec(db, reset_index=True)
        db_keys = set(zip(db_process["file"], db_process["start"], db_process["end"]))

        if os.path.exists(path):
            df = pd.read_pickle(path)
            print(f"Loaded Whisper embeddings from {path}")

            df_keys = set(zip(df["file"], df["start"], df["end"]))
            missing_keys = db_keys - df_keys

            if not missing_keys:
                print("All segments already have extracted features")
            else:
                print(f"Extracting {len(missing_keys)} missing segments...")
                processor, model, decoder_input_ids, device = self.load_whisper()
                #input(missing_keys)
                df_missing = self.extract_rows(
                    missing_keys, processor, model, decoder_input_ids, device
                )
                df = pd.concat([df, df_missing], ignore_index=True)
                df.to_pickle(path)
                print(f"Updated features saved to {path}")

        else:
            print("Extracting Whisper embeddings from scratch...")
            processor, model, decoder_input_ids, device = self.load_whisper()
            segments = db_keys
            df = self.extract_rows(
                segments, processor, model, decoder_input_ids, device
            )
            df.to_pickle(path)
            print(f"Saved Whisper embeddings to {path}")

        feat_columns = [c for c in df.columns if c.startswith("whisper_embedding_")]
        return df, feat_columns

    def extract_rows(self, segments, processor, model, decoder_input_ids, device):
        rows = []

        for filepath, start, end in tqdm(segments, desc="Extracting segments"):
            #input(filepath)
            #input(start)
            segment = self.load_segment(filepath, start, end)
            if segment is None:
                continue

            emb = self.extract_embedding(
                segment, processor, model, decoder_input_ids, device
            )
            row = {"file": filepath, "start": start, "end": end}
            row.update({f"whisper_embedding_{i}": v for i, v in enumerate(emb)})
            rows.append(row)

        return pd.DataFrame(rows)

    def extract_wav2vec(self, db):

        path = os.path.join(self.output, f"{self.task}_wav2vec_embeddings.pkl")

        if not os.path.exists(path):
            print("Extracting Wav2Vec2 embeddings...")
            model_name = "facebook/wav2vec2-large-xlsr-53-german"
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2Model.from_pretrained(
                model_name, gradient_checkpointing=True
            )
            model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            model.to(device)

            TARGET_SR = 16000
            if "start" not in db.columns:
                db.reset_index(inplace=True)

            db_process = ut.dt2sec(db, reset_index=True)

            file_to_segments = defaultdict(list)

            rows = []
            for idx, row in tqdm(
                db_process.iterrows(), total=len(db_process), desc="Extracting segments"
            ):
                filepath = row["file"]
                start = row["start"]
                end = row["end"]

                signal, sr = torchaudio.load(filepath)
                if sr != TARGET_SR:
                    signal = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=TARGET_SR
                    )(signal)

                # Extract segment
                start_frame = int(start * TARGET_SR)
                end_frame = int(end * TARGET_SR)
                segment = signal[:, start_frame:end_frame].squeeze()
                # Skip if segment is empty
                if segment.numel() < 100:
                    continue
                # Tokenize and move to device
                inputs = processor(
                    segment, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                # (T, D) → take last frame ONLY (no averaging)
                emb = outputs.last_hidden_state[0, -1].cpu().numpy()  # (1024,)

                row_dict = {
                    "file": filepath,
                    "start": start,
                    "end": end,
                }

                for i, v in enumerate(emb):
                    row_dict[f"wav2vec_embedding_{i}"] = v

                rows.append(row_dict)

            final_data = pd.DataFrame(rows)

            with open(path, "wb") as f:
                pickle.dump(final_data, f)

            print(f"Saved Wav2Vec2 embeddings to {path}")

        else:
            final_data = pd.read_pickle(path)
            # Get embedding columns from loaded DataFrame
        embedding_cols = [
            col for col in final_data.columns if col.startswith("wav2vec_embedding_")
        ]

        return final_data, embedding_cols

    def extract_bert(self, db):

        # Placeholder for BERT feature extraction
        # Implement BERT feature extraction logic here
        df_feat = pd.DataFrame()  # Replace with actual extracted features
        feat_columns = []  # Replace with actual feature column names

        return df_feat, feat_columns

    @staticmethod
    def save_processed_data(
        data: pd.DataFrame, feat_columns: list, file_name: str, lst_feature_file: str
    ) -> pd.DataFrame:
        """
        Description: function to save the processed data into a dict. It preserves the tensors
        Args:
            data (pd.DataFrame): data with the processed input
            feature_columns(list): list of the name of the feature columns
            file_name(str): absolute filename where dictionary will be stored
            lst_feature_file(str): name of file where to store the list
        Returns:
            df_all(pd.DataFrame)
        """

        # * extended the list of columns that are saved with the features to make analysis of results per subgroup easier
        columns_to_save = [
            "label",
            "ID",
            "file",
            "dataset",
            "age",
            "gender",
            "mmse",
            "split",
        ]
        columns_to_save.extend(feat_columns)

        df_all = data[columns_to_save]
        df_all.to_pickle(file_name)

        with open(lst_feature_file, "wb") as file:
            pickle.dump(feat_columns, file)

        print(f"Data has been processed and saved in {file_name}")

        return df_all

    def load_whisper(self):

        MODEL_NAME = "openai/whisper-large-v3"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        model = WhisperModel.from_pretrained(MODEL_NAME).to(device).eval()

        decoder_input_ids = torch.tensor(
            [[model.config.decoder_start_token_id] * 2],
            device=device,
        )
        return processor, model, decoder_input_ids, device

    def extract_embedding(self,
        segment, processor, model, decoder_input_ids, device, TARGET_SR=16000
    ):
        inputs = processor(
            segment,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            return_attention_mask=True,
        )

        outputs = model(
            input_features=inputs.input_features.to(device),
            attention_mask=inputs.attention_mask.to(device),
            decoder_input_ids=decoder_input_ids,
        )

        #return outputs.last_hidden_state[0, -1].cpu().numpy()
        return outputs.last_hidden_state[0, -1].detach().cpu().numpy()


    def load_segment(self, filepath, start, end, TARGET_SR=16000, MIN_SAMPLES=100):
        signal, sr = torchaudio.load(filepath)
        if sr != TARGET_SR:
            signal = torchaudio.transforms.Resample(sr, TARGET_SR)(signal)

        start_frame = int(start * TARGET_SR)
        end_frame = int(end * TARGET_SR)
        segment = signal[:, start_frame:end_frame].squeeze()

        if segment.numel() < MIN_SAMPLES:
            return None

        return segment