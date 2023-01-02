"""
README : This file contains the code for the data
collection and extraction from the Million Song Dataset
"""
import os
import pandas as pd
import hdf5_getters

# get the list of all the files inside the folder
files = []
for i in os.listdir('MillionSongSubset'):
    if i != '.DS_Store':
        for j in os.listdir('MillionSongSubset/' + i):
            if j != '.DS_Store':
                for k in os.listdir('MillionSongSubset/' + i + '/' + j):
                    if k != '.DS_Store':
                        for l in os.listdir('MillionSongSubset/' + i + '/' + j + '/' + k):
                            if (l != '.DS_Store') & (l.endswith('.h5')):
                                files.append('MillionSongSubset/' + i + '/' + j + '/' + k + '/' + l)


def data_extract(file):
    """_summary_ : This function extracts the data from the hdf5 files and returns a dataframe

    Args:
        file: list of files to be extracted

    Returns:
        _dataframe_: dataframe of the extracted data
    """
    data_song = pd.DataFrame()

    # get the data from the files to the dataframe
    for song in file:
        h5 = hdf5_getters.open_h5_file_read(song)
        data_song = data_song.append({'song_id': hdf5_getters.get_song_id(h5).decode('utf-8'),
                                      'title': hdf5_getters.get_title(h5).decode('utf-8'),
                                      'artist_name': hdf5_getters.get_artist_name(h5).decode('utf-8'),
                                      'artist_familiarity': hdf5_getters.get_artist_familiarity(h5),
                                      'artist_hotttnesss': hdf5_getters.get_artist_hotttnesss(h5),
                                      'artist_id': hdf5_getters.get_artist_id(h5).decode('utf-8'),
                                      'artist_latitude': hdf5_getters.get_artist_latitude(h5),
                                      'artist_longitude': hdf5_getters.get_artist_longitude(h5),
                                      'artist_location': hdf5_getters.get_artist_location(h5).decode('utf-8'),
                                      'artist_playmeid': hdf5_getters.get_artist_playmeid(h5),
                                      'artist_7digitalid': hdf5_getters.get_artist_7digitalid(h5),
                                      'artist_mbid': hdf5_getters.get_artist_mbid(h5).decode('utf-8'),
                                      'artist_terms': hdf5_getters.get_artist_terms(h5),
                                      'artist_terms_freq': hdf5_getters.get_artist_terms_freq(h5),
                                      'artist_terms_weight': hdf5_getters.get_artist_terms_weight(h5),
                                      'audio_md5': hdf5_getters.get_audio_md5(h5).decode('utf-8'),
                                      'danceability': hdf5_getters.get_danceability(h5),
                                      'duration': hdf5_getters.get_duration(h5),
                                      'end_of_fade_in': hdf5_getters.get_end_of_fade_in(h5),
                                      'energy': hdf5_getters.get_energy(h5),
                                      'key': hdf5_getters.get_key(h5),
                                      'key_confidence': hdf5_getters.get_key_confidence(h5),
                                      'loudness': hdf5_getters.get_loudness(h5),
                                      'mode': hdf5_getters.get_mode(h5),
                                      'mode_confidence': hdf5_getters.get_mode_confidence(h5),
                                      'release': hdf5_getters.get_release(h5).decode('utf-8'),
                                      'release_7digitalid': hdf5_getters.get_release_7digitalid(h5),
                                      'song_hotttnesss': hdf5_getters.get_song_hotttnesss(h5),
                                      'start_of_fade_out': hdf5_getters.get_start_of_fade_out(h5),
                                      'tempo': hdf5_getters.get_tempo(h5),
                                      'time_signature': hdf5_getters.get_time_signature(h5),
                                      'time_signature_confidence': hdf5_getters.get_time_signature_confidence(h5),
                                      'track_7digitalid': hdf5_getters.get_track_7digitalid(h5),
                                      'year': hdf5_getters.get_year(h5)},
                                     ignore_index=True)
        h5.close()
    return data_song


if __name__ == '__main__':
    data = data_extract(files)
    data.to_csv('songs_data.csv', encoding='utf-8')
    print(data.shape)
    print(data.head())
