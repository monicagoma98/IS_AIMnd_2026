import pandas as pd
import pdb
import os

import plots as plots
import librosa
import soundfile as sf

# age, patient id, condition, gender, score results from ALS test(self-assessed), comorbidities.
#new variables
#Therapie, ALSFRS-R Gesamt, ALSFRS-R "Sprechen", ALSFRS-R "Schlucken",ALSFRS-R Bulbär, 
# QOL-DYS-G, ALSFRS-R "Speichelfluss", Muttersprache, Nebendiagnosen (Neuro, Psych), 

def create_metadata(odir):
    """Creates metadata"""

    relevant_variables = [
        "IDs",
        "Untersuchung (*bei FU Zeile unten hinzufügen)",
        "Alter bei Untersuchung",
        "Geschlecht",
        "Erkrankung",
        "Verlaufsform",
        "ALSFRS-R Gesamt",
        "ALSFRS-R Sprechen",
        "QOL-DYS-G",
        "Muttersprache",
        "Nebendiagnosen (Neuro, Psych)",
    ]

    # update with new csv files
    csv_metadata_control = "Videos_Schuller/Datentabelle AI-Mnd_Proband.csv"
    csv_metadata_als = "Videos_Schuller/Datentabelle AI-Mnd_Patienten.csv"
    base_folder_path = "/media/mgonzalez/Elements/"
    
    id_als = pd.read_csv(
        os.path.join(base_folder_path, csv_metadata_als), dtype={"IDs": str}
    )

    id_control = pd.read_csv(
        os.path.join(base_folder_path, csv_metadata_control), dtype={"IDs": str}
    )

    all_metadata_old_ids = pd.concat([id_als, id_control])

    excel_file_path = "ID_Liste_Schuller.xlsx"
    new_ids = pd.read_excel(
        os.path.join(base_folder_path, excel_file_path),
        engine="openpyxl",
        dtype={"Alte ID": str, "Neue ID": str},
    )

    new_ids = new_ids.rename(columns={"Alte ID": "IDs"})

    merged_metadata = new_ids.merge(all_metadata_old_ids, on=["IDs"])

    merged_metadata.drop(["IDs"], axis=1, inplace=True)

    metadata_file = "CHI_ALSprojekt/all_metadata_new_ids.csv"
    
    # Rename columns for easier analysis
    merged_metadata = merged_metadata.rename(columns={
        "Untersuchung (*bei FU Zeile unten hinzufügen)": "untersuchung",
        "Alter bei Untersuchung": "age",
        "Geschlecht": "sex",
    })
    metadata_file = merged_metadata[relevant_variables]
         
    metadata_file["sex"] = metadata_file["sex"].replace({"weiblich": "W"})
    metadata_file["sex"] = metadata_file["sex"].replace({"männlich": "M"})

    merged_metadata.to_csv(os.path.join(base_folder_path, metadata_file))

    plots.plot_dis(merged_metadata, odir)
    
    # Loop over all folders in base_folder_path, assuming each folder is named by ID
    video_info = []
    for id_folder in os.listdir(base_folder_path):
        id_folder_path = os.path.join(base_folder_path, id_folder)
        if os.path.isdir(id_folder_path):
            # Loop over video files in each ID folder
            for video_file in os.listdir(id_folder_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # Extract prompt and session from filename, assuming format: prompt_session.ext
                    name_parts = os.path.splitext(video_file)[0].split('_')
                    prompt = name_parts[0] if len(name_parts) > 0 else ""
                    session = name_parts[1] if len(name_parts) > 1 else ""
                    video_info.append({
                        "IDs": id_folder,
                        "prompt": prompt,
                        "session": session,
                        "video_file": video_file
                    })

    # Create a dataframe containing id, prompt, session, and video_file
    video_df = pd.DataFrame(video_info)


    return merged_metadata

