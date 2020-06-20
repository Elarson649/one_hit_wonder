import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from spotify_api import spotify_api


def genre_data(df_raw):
    """
    Returns a dataframe with the year, genres, and number of times that genre appeared in that year's end of year chart
    :param df_raw: Yearly billboard top 100 information with Spotify information
    :return: A summary of the most popular genres for each year as a dataframe
    """
    df_genre = df_raw[df_raw['genres'].notnull()]
    df_genre = df_genre[['Rank', 'Year', 'genres']]
    df_genre = df_genre.reset_index(drop=True)
    df_genre_summary = pd.DataFrame(columns=['genre', 'frequency', 'year'])

    # For each year, count how often each genre was represented in the billboard 100
    for year, year_df in df_genre.groupby('Year'):
        one_hot = MultiLabelBinarizer()
        dummy = one_hot.fit_transform(year_df['genres'])
        dummy_sum = np.sum(dummy, axis=0)
        years = np.ones(len(dummy_sum), dtype=np.int8) * year
        df_temp = pd.DataFrame({'genre': one_hot.classes_, 'frequency': dummy_sum, 'year': years})
        df_genre_summary = pd.concat([df_genre_summary, df_temp])

    df_genre_summary = df_genre_summary.reset_index(drop=True)
    return df_genre_summary


def lyric_data(df_lyrics):
    """
    Returns the lyrics of year-end billboard top 100 hits
    :param df_lyrics: The year and lyrics of year-end billboard top 100 hits
    :return: The year and lyrics of the songs as a dataframe, minus any songs that do not have lyrics
    """
    mask_lyrics = (df_lyrics['Lyrics'].notnull()) & (df_lyrics['Lyrics'] != '')
    df_lyrics = df_lyrics[mask_lyrics]

    # Make sure the lyrics have some alpha characters
    char_mask = df_lyrics['Lyrics'].apply(lambda x: re.search('[a-zA-Z]', x) is not None)
    df_lyrics = df_lyrics[char_mask]
    df_lyrics = df_lyrics.reset_index(drop=True)

    # Dummy column used in feature engineering
    df_lyrics['year_end_100'] = 'billboard'
    return df_lyrics


def feature_data(df_raw):
    """
    Parses the audio features returned by Spotify into separate columns, performs some feature engineering to match the
    weekly billboard information
    :param df_raw: Yearly billboard top 100 information with Spotify information
    :return: The songs and their features as a dataframe
    """
    df_raw = df_raw[df_raw['features'].notnull()]
    df_raw = df_raw[df_raw['features'].apply(lambda x: x[0] is not None)]
    df_features = df_raw[['Year', 'features']]
    df_features.columns = ['year', 'features']
    df_features = df_features.reset_index(drop=True)
    list_features = [x[0] for x in df_features['features']]
    features = pd.DataFrame(list_features)
    df_features = df_features.merge(features, left_index=True, right_index=True)
    df_features['mode_1'] = 1
    df_features.loc[df_features['mode'] != 1, 'mode_1'] = 0
    df_features = df_features.drop(
        columns=['track_href', 'uri', 'id', 'type', 'features', 'analysis_url', 'key', 'energy', 'time_signature',
                 'mode'])
    df_features['year_end_100'] = 'billboard'
    return df_features


def import_yearly_data(filepath):
    """
    Turns the year-end billboard charts from 1979 onwards into three dataframes, one for lyrics, another with the genres
    by year, and the last for the features for each song. Each data frame will be used in feature engineering, later.
    :param filepath: The filepath of the billboard year end charts
    :return: Three data frames: one for lyrics, another for genres, and the last for features, respectively
    """
    df_raw = pd.read_csv(filepath, encoding='ISO-8859-1')
    df_raw = df_raw[df_raw['Year'] >= 1979]
    df_raw = df_raw.reset_index(drop=True)
    df_lyrics = df_raw[['Year', 'Lyrics']]
    df_lyrics = lyric_data(df_lyrics)

    df_raw = df_raw.drop(columns=['Lyrics', 'Source'])
    df_raw.columns = ['Rank', 'Song', 'Performer', 'Year']

    # Perform cleaning to get the data ready for the Spotify API
    df_raw['Primary_Performer'] = df_raw['Performer'].apply(lambda x: re.split(' and | \( | featuring | with ', x)[0])
    spotify_data = df_raw.apply(spotify_api, axis=1, result_type='expand')
    spotify_data.columns = ['features', 'spot_artist', 'genres']
    df_raw = df_raw.merge(spotify_data, left_index=True, right_index=True)

    df_genre_summary = genre_data(df_raw)
    df_features = feature_data(df_raw)
    return df_lyrics, df_genre_summary, df_features
