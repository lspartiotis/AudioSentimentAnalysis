from features import csv_to_pd
import pandas as pd
import os
import csv

# Creates Final CSV of audio features + valence&arousal labels
# deampath: the path of th eDEAM dataset containing
#           valence&arousal label annotations
# csv_audio_features_path: path of the extracted audio features csv
def get_features(deam_path, csv_audio_features_path):
    emotions = ["happy", # Valence > 0 and Arousal >0
        "calm"   # Valence > 0 and Arousal < 0
        "angry",  # Valence <0 and Arousal > 0
        "sad",  # Valence < 0 and Arousal < 0
    ]

    au_df = csv_to_pd(csv_audio_features_path)
    print(au_df.head())
    df = pd.read_csv(deam_path, usecols=['song_id', ' valence_mean', ' arousal_mean'])
    print(df[' arousal_mean'].min(), df[' arousal_mean'].max())
    print(df[' valence_mean'].min(), df[' valence_mean'].max())
    # Rename columns
    df.rename(columns = {' valence_mean':'valence'}, inplace = True)
    df.rename(columns = {' arousal_mean':'arousal'}, inplace = True)

    total_data = pd.DataFrame(au_df)

    # go row by row, coz some annotations missing from DEAM
    for indfex, row in df.iterrows():
        if row['song_id'] in total_data['song_id'].values:
            total_data.loc[total_data['song_id'].values == row['song_id'], 'valence'] = row['valence']
            total_data.loc[total_data['song_id'].values == row['song_id'], 'arousal'] = row['arousal']
    print(total_data.head())
    print('oh')
    print(df['valence'])

    # create the labels
    for indfex, row in total_data.iterrows():
      if row['valence'] >= 5 and row['arousal'] >= 5:
          total_data.at[indfex, 'emotion'] = 1
      if row['valence'] >= 5 and row['arousal'] < 5:
          total_data.at[indfex, 'emotion'] = 2
      if row['valence'] < 5 and row['arousal'] >= 5:
          total_data.at[indfex, 'emotion'] = 3
      if row['valence'] < 5 and row['arousal'] < 5:
          total_data.at[indfex, 'emotion'] = 4
    print('done')
    print(total_data)

    # since some annotations were missing,
    # create a test dataset with the missing annotations
    empty_values_df = total_data[total_data['valence'].isnull()]
    empty_values_df = empty_values_df.drop(columns=['valence','arousal','emotion'])
    print(empty_values_df)

    # and only keep fully annotated data for training
    total_data = total_data.dropna()
    TotalData_df = pd.DataFrame(total_data)
    #TotalData_df.columns = ['song',  'emotion', 'arousal', "valence"]
    TotalData_df.to_csv("./data/total_data.csv",index=False)


    return TotalData_df, empty_values_df

# read metadata provided DEAM file and return pd
def get_metadata_pd (metadata_path, output_csv_path):
    dataframes = []

    for file in os.listdir(metadata_path):
        meta_df = pd.read_csv(file)
        dataframes.append(meta_df)

    final_meta_df = pd.concat(dataframes, ignore_index=True)
    final_meta_df.to_csv(output_csv_path, index=False)
    print(final_meta_df.head())
    return final_meta_df

# get pd of metadata&audio features
def merge_feat_and_metadata (mid_term_csv_path, meta1, meta2, meta3):
    # metadata is split into 3 groups, total 2058 songs
    # midterm features are 1802 of these songs
    # so for all 3 metadata files, loop and attach the entry with 'song_id' same as mid_term_feature 'song_id'

    audio_df = pd.read_csv(mid_term_csv_path)
    audio_df=pd.DataFrame(audio_df)
    meta_df1 = pd.read_csv(meta1)
    meta_df2 = pd.read_csv(meta2)
    meta_df3 = pd.read_csv(meta3)

    meta_df1 = pd.DataFrame(meta_df1)
    meta_df2 = pd.DataFrame(meta_df2)
    meta_df3 = pd.DataFrame(meta_df3)
    # merge audio features and metadata features
    #frames = [audio_df, metaDF['genre']]
    # TotalData_df = pd.concat(frames)
    metas = [meta_df1, meta_df2, meta_df3]
    audio_df.reset_index(drop=True)
    for meta_df in metas:
        meta_df.reset_index(drop=True)
        for indfex, row in audio_df.iterrows():
            if audio_df['song_id'] == meta_df['song_id']:
                audio_df['Genre'] = meta_df['Genre']
    return audio_df


### get final dataframe of both features and labels (already saved in file total_data.csv)
## and a dataframe of just features, no labels
tt, ee = get_features('./data/static_annotations_averaged_songs_1_2000.csv', './data/mid_audio_full.csv')
# save features+labels and features datasets to csv
tt.to_csv('./data/total_annotated_data.csv')
ee.to_csv('./data/total_not_annotated_data.csv')
print(tt.shape)