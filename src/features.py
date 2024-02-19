import os
import pandas as pd
from pyAudioAnalysis import MidTermFeatures as mtf
from pyAudioAnalysis import ShortTermFeatures, audioBasicIO
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt


deam_audio_path = '../../data/archive/DEAM_audio/MEMD_audio/'
pkl_path = './data/data.pkl'



def audio_feature_extraction(deam_audio_path, pkl_path, shortTerm = False):
    dir = os.listdir(deam_audio_path)
    tott = len(dir)
    all_mid_term_features = []
    all_files = []
    all_feature_names = []
    if os.path.isfile(pkl_path):
        with open(pkl_path,'rb') as f:
            mid_term_features_2 = pickle.load(f)
            wav_file_list2 = pickle.load(f)
            #mid_feature_names = pickle.load(f)

            all_mid_term_features = mid_term_features_2
            # get col labels
            _, _, mid_feature_names = mtf.mid_feature_extraction(x[:, 0], Fs,
                              mws,
                              mss,
                              ws,
                              ss)
            print('extracted!')
            print(mid_term_features_2.shape, mid_term_features_2, mid_feature_names )
            print(len(wav_file_list2))
            wavFeatures_df = pd.DataFrame(all_mid_term_features)
            if shortTerm != True:
                for i,filename in enumerate(wav_file_list2):
                    filename = filename.split('/')[-1]
                    wav_file_list2[i] = filename.split('.')[0]

                wavFeatures_df.columns = [mid_feature_names]
                wavFeatures_df['song_id'] = wav_file_list2
            print(wavFeatures_df.head())
    else:
        with open(pkl_path,'wb') as f:
            if shortTerm:
                    # this loop is for using ShortTermFeature. Otherwise, comment it
                j = 0
                for fille in dir:
                    print(j, "/", tott, fille)
                    # computes all audio features of wav files in given folder
                    [Fs, x] = audioBasicIO.read_audio_file(deam_audio_path + fille)
                    #print(Fs,x.shape)
                    if len(x.shape) == 1:
                        mid_term_features_2, mid_feature_names = ShortTermFeatures.feature_extraction(x, Fs, 0.08*Fs, 0.08*Fs)
                    else:
                        mid_term_features_2, mid_feature_names = ShortTermFeatures.feature_extraction(x[:,0], Fs, 0.08*Fs, 0.08*Fs)
                    #print(F.shape, feature_names, len(feature_names))

                    all_mid_term_features.append(mid_term_features_2)
                    all_feature_names.append(mid_feature_names)
                    all_files.append(fille)

                    j += 1
            else:
                mid_term_features, wav_file_list2, mid_feature_names = mtf.directory_feature_extraction(deam_audio_path ,1,1, 0.080, 0.08)
                # Append the features and file information to the lists
                print(mid_term_features, wav_file_list2, mid_feature_names)
                #beat, beat_conf = mtf.beat_extraction(mid_term_features, 0.1)
                #mid_features = np.append(mid_features, beat)
                #mid_features = np.append(mid_features, beat_conf)

                # Save before standardization
                wavFeatures_df = pd.DataFrame(mid_term_features)
                if shortTerm != True:
                    for i,filename in enumerate(wav_file_list2):
                        filename = filename.split('/')[-1]
                        wav_file_list2[i] = filename.split('.')[0]
                wavFeatures_df.columns = [mid_feature_names]
                wavFeatures_df['song_id'] = wav_file_list2
                ffile = open('unnormalized_midterm.pkl', 'wb')
                pickle.dump(wavFeatures_df, ffile)
                ffile.close()
                # Standardization
                m = mid_term_features.mean(axis=0)
                s = np.std(mid_term_features, axis = 0)
                print(s)
                mid_term_features_2 = (mid_term_features - m) / s
                all_mid_term_features = mid_term_features_2
                all_feature_names = mid_feature_names
                all_files = mid_feature_names

                print(all_mid_term_features, all_feature_names)
            # in shorttermffeatures, here it is an array. Change to mid_term+features_2 and wav_file_list2
            #all_mid_term_features = all_mid_term_features.reshape(-1, all_mid_term_features.shape[2])

            #pickle.dump(all_mid_term_features, f)
            #pickle.dump(all_files, f)
            #pickle.dump(all_feature_names, f)

            # we now have all sound features extracted from wav files
            # print dimensions
            print(len(all_mid_term_features))
            wavFeatures_df = pd.DataFrame(all_mid_term_features)
            # cahnge the wav_filename_list to just the filename (song_id)
            if shortTerm != True:
                for i,filename in enumerate(wav_file_list2):
                    filename = filename.split('/')[-1]
                    wav_file_list2[i] = filename.split('.')[0]

            wavFeatures_df.columns = [mid_feature_names]
            wavFeatures_df['song_id'] = wav_file_list2

            pickle.dump(wavFeatures_df, f)
            print(wavFeatures_df.head())
    wavFeatures_df.to_csv('./data/mid_audio_full.csv')
    # create dataframe and return it
    return wavFeatures_df

# for newer versions of above, where okl saves the whole dataframe
# use below to load
def pkl_to_pd(pkl_path):
    with open(pkl_path,'rb') as f:
        mid_term_features_2 = pickle.load(f)
    return mid_term_features_2

def csv_to_pd(csv_path):
    return pd.read_csv(csv_path, index_col=0)
#audio_feature_extraction(deam_audio_path)


#aa = audio_feature_extraction('../../data/archive/DEAM_audio/MEMD_audio/', './data/sample.pkl', False)

pickledf = pkl_to_pd('./data/sample.pkl')
print(pickledf.head())
print(pickledf.tail())

#run = pickledf.to_csv('./data/mid_features.csv', sep='\t', encoding='utf-8')

