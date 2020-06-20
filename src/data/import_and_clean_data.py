import pandas as pd
import numpy as np
import re
import string
from fuzzywuzzy import fuzz
from spotify_api import spotify_api
from genius_api import genius_api


def fuzzy(df, column):
    """
    Compares the primary performer column to a given column and returns a score based on Levenshtein distance. Meant to
    check if results from APIs are accurate. Function typically used with an apply function.
    :param df: A row of our one hit wonder dataframe
    :param column: Column to compare the primary performer to
    :return: A score from 0 to 100
    """
    bb_artist = df['Primary_Performer'].lower()
    column = df[column].lower()
    bb_artist = re.sub('[%s]' % re.escape(string.punctuation), '', bb_artist)
    column = re.sub('[%s]' % re.escape(string.punctuation), '', column)
    ratio = fuzz.ratio(bb_artist, column)

    if ratio < 50:
        bb_artist = bb_artist.split(' ')[0]
        column = column.split(' ')[0]
        ratio = fuzz.ratio(bb_artist, column)

    return ratio


def performer_cleanup(df):
    """
    Splits the performer into a primary performer and a secondary performer and performs some clean-up
    :param df: The dataframe with all the billboard top 100 hits
    :return: The dataframe with the new performer columns
    """
    # Let's split out the performer into the primary performer and a featured performer
    df['Performer'] = df['Performer'].apply(lambda x: x.lower())
    df['Performer'] = df['Performer'].replace({'feat\.': 'featuring'}, regex=True)

    # 'Featuring' and 'With 'nearly always insinuate a featured performer. 'And' and '&' are more complex, often
    # indicating band names
    df['Primary_Performer'] = df['Performer'].apply(
        lambda x: re.split(' and | & | \(| featuring | duet with | with | x ', x)[0])
    df['Featured_Performer'] = df['Performer'].apply(
        lambda x: re.split('featuring | with ', x)[1] if ((' featuring ' in x) | (' with ' in x)) else None)

    # Let's clean up the primary performer column
    df['Primary_Performer'] = df['Primary_Performer'].apply(lambda x: x.split('/')[0])
    df['Primary_Performer'] = df['Primary_Performer'].apply(lambda x: x.split(',')[0])
    df['Primary_Performer'] = df['Primary_Performer'].apply(lambda x: x.rstrip())
    return df


def manipulate_data(df):
    """
    Updates the billboard top 100 data to show the first top 40 hit for an artist
    :param df: Dataframe with all the billboard top 100 data
    :return: A dataframe of potential one hit wonders
    """

    # Clean-up the names of the performers, create new primary and featured artist columns
    df = performer_cleanup(df)

    # Looking at the first date the song was on the chart, to avoid data leakage when factoring in year-end information
    df['datetime'] = pd.to_datetime(df['WeekID'])
    df['first_week'] = df.groupby('SongID')['datetime'].transform('min')
    df['year'] = df['first_week'].dt.year
    df['Weeks on Chart'] = df.groupby('SongID')['Weeks on Chart'].transform('max')
    df = df.drop(columns=['WeekID'])

    # Sort so the row representing a particular song will be its best (lowest) position and the earliest date
    # it reached that position. Then drop duplicates, so we only keep the first row.
    df = df.sort_values(by=['SongID', 'Week Position', 'datetime'])
    df = df.drop_duplicates(subset=['SongID'])

    # Sort by performer and datetime for the below cumulative count, where we want to number the songs of an artist
    # in sequence
    df = df.sort_values(by=['Primary_Performer', 'datetime'])
    # Defining a one-hit wonder as an artist/song that only reached the billboard top 40 once
    df = df[df['Week Position'] <= 40]
    df['num_single_chart'] = df.groupby('Primary_Performer').cumcount() + 1
    df['total_songs_on_chart'] = df.groupby('Primary_Performer')['SongID'].transform('nunique')

    # Only looking at the first song of an artist that reached the top 40.
    one_hit = (df['total_songs_on_chart'] == 1)
    not_one_hit = (df['total_songs_on_chart'] != 1) & (df['num_single_chart'] == 1)
    df = df[not_one_hit | one_hit]
    df = df.reset_index(drop=True)
    return df


def clean_data(df):
    """
    Removes songs with bad API data from the dataset
    :param df: The potential one-hit wonders with information from our APIs
    :return: The cleaned dataset
    """
    # Expand features from Spotify API so each has their own column
    api_filter = (df['lyrics'].notnull()) & (df['features'].notnull())
    df = df[api_filter]
    df = df.reset_index(drop=True)
    list_features = [x[0] for x in df['features']]
    features = pd.DataFrame(list_features)
    df = df.merge(features, left_index=True, right_index=True)

    # Use Fuzzy/Levenshtein Distance to compare the artist on the billboard chart to the artist returned by Spotify
    # and Genius. If a low score (they're not similar), filter these results out, since it is likely the API returned
    # the wrong results
    df['spot_ratio'] = df.apply(fuzzy, axis=1, column='spot_artist')
    df['gen_ratio'] = df.apply(fuzzy, axis=1, column='gen_artist')
    mask_api = (df['spot_ratio'] > 50) & (df['gen_ratio'] > 50)
    df = df[mask_api]

    # Filter entries where the length of lyrics is too long/the API likely returned the wrong result
    lyric_len = [len(x) for x in df['lyrics']]
    mask_len = np.array(lyric_len) < 10000
    df = df[mask_len]
    df = df.reset_index(drop=True)
    return df


def import_and_clean_data(filepath):
    """
    Imports the weekly billboard top 100, manipulates it to show potential one hit wonders (songs that hit the top 40)
    and adds data from Spotify and Genius
    :param filepath: The filepath of the billboard top 100 file
    :return: A list of potential one hit wonders with additional information from APIs
    """
    # Import weekly hot 100 files
    df = pd.read_csv(filepath)
    df = df.drop(columns=['url', 'Instance', 'Previous Week Position', 'Peak Position'])
    df = manipulate_data(df)

    # Get, label, and merge data from Genius API
    genius_data = df.apply(genius_api, axis=1, result_type='expand')
    genius_data.columns = ['lyrics', 'writers', 'producers', 'gen_artist']
    df = df.merge(genius_data, left_index=True, right_index=True)

    # Get, label, and merge data from Spotify API
    spotify_data = df.apply(spotify_api, axis=1, result_type='expand')
    spotify_data.columns = ['features', 'spot_artist', 'genres']
    df = df.merge(spotify_data, left_index=True, right_index=True)
    df = clean_data(df)
    return df
