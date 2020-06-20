import en_core_web_sm
import numpy as np
import pandas as pd
import readability
from cleaning_text_data import cleaning_text_data
from langdetect import detect
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from topic_model import topic_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from vectorize_data import vectorize_data


def time_sensitive_features(df, hit_track_minimum, genre_minimum):
    """
    Calculates features that need all of the potential one-hit wonders. The final dataset will only be one-hit wonders
    from 1980 to 2016, but to accurately calculate some metrics, we need to use all of the data.
    :param df: Dataframe with potential one-hit wonders
    :param hit_track_minimum: The minimum number of tracks a producer or writer needs to be associated with to be
    considered a hit
    :param genre_minimum: The minimum number of times a genre needs to appear on the weekly charts to be it's own metric
    :return: Dataframe with potential one-hit wonders and new features
    """
    # Sort dataframe by chart entry date- will be useful for the cumulative sum for hit producers/writers
    df = df.sort_values(by=['first_week'])
    df = df.reset_index(drop=True)
    df.loc[df['writers'].isnull(), 'writers'] = '[]'

    # MultiLabelBinarizer will one-hot encode a column that is a list of values
    one_hot_writer = MultiLabelBinarizer()
    dummy_writer = one_hot_writer.fit_transform(df['writers'])

    # Counts how many tracks a write has worked on as they are accumulated
    dummy_writer_sum = dummy_writer.cumsum(axis=0)

    # Element-wise multiplication of the cumulative and the one hot encoded function gets the number of top 40 songs
    # the writer was involved with at the time of the song's release, with a 0 in the column if the writer was not
    # involved in the track
    dummy_writer = np.multiply(dummy_writer, dummy_writer_sum)

    # Check if the writer had at least hit_track_minimum number of top 40 first singles before the release of the
    # single they worked on. If there's at least one writer that meets this condition, hit_writers should be 1.
    # Otherwise it will be 0.
    dummy_writer = ((dummy_writer >= hit_track_minimum) & (one_hot_writer.classes_ != '[')
                    & (one_hot_writer.classes_ != ']')) * 1
    df['hit_writers'] = (np.sum(dummy_writer, axis=1) >= 1) * 1

    # Create column to show whether a hit producer worked on the production of the song. Logic is the same as the logic
    # for hit writer.
    df.loc[df['producers'].isnull(), 'producers'] = '[]'
    one_hot_producer = MultiLabelBinarizer()
    dummy_producer = one_hot_producer.fit_transform(df['producers'])
    dummy_sum_producer = dummy_producer.cumsum(axis=0)
    dummy_producer = np.multiply(dummy_producer, dummy_sum_producer)
    dummy_producer = ((dummy_producer >= hit_track_minimum) & (one_hot_producer.classes_ != '[')
                      & (one_hot_producer.classes_ != ']')) * 1
    df['hit_producers'] = (np.sum(dummy_producer, axis=1) >= 1) * 1

    # Let's encode genres that appear genre_minimum or more times on the chart
    df.loc[df['genres'].isnull(), 'genres'] = '[]'
    one_hot_genre = MultiLabelBinarizer()
    dummy_genre = one_hot_genre.fit_transform(df['genres'])
    dummy_sum_genre = np.sum(dummy_genre, axis=0)
    mask_genre = (dummy_sum_genre >= genre_minimum)

    # Filter the genre one hot encoded matrix to only show genres that meet genre_minimum
    dummy_genre = dummy_genre[:, mask_genre]
    genre = pd.DataFrame(dummy_genre, columns=one_hot_genre.classes_[mask_genre])
    df = df.merge(genre, left_index=True, right_index=True)
    return df


def difficulty(df):
    """
    Calculates the difficulty of the song's lyrics. Intended to be used with an apply function.
    :param df: Dataframe of the potential one-hit wonders
    :return: The readability score
    """
    if df['foreign_language'] == 1:
        return np.nan
    else:
        text = df['lyric_difficulty']
        results = readability.getmeasures(text, lang='en')
        return results['readability grades']['FleschReadingEase']


def entity(df):
    """
    Returns how many named entities there are in a song's lyrics. Intended to be used with an apply function.
    :param df: The potential one-hit wonder's lyrics
    :return: The number of named entities in the song's lyrics
    """
    # Returns number of unique named entities that aren't people (people weren't accurate, from my sampling)
    nlp = en_core_web_sm.load()
    doc = nlp(df)
    return len(set([result.text for result in doc.ents if result.label_ != 'PERSON']))


def nlp_features(df):
    """
    Adds a number of NLP features to each potential one-hit wonder
    :param df: Dataframe of the one-hit wonders
    :return: Dataframe of the one-hit wonders plus new NLP metrics
    """
    # Don't remove stop words, let's only look at songs with lyrics (sorry instrumentals!)
    df['full_lyrics'] = cleaning_text_data(df['lyrics'], remove_words=False)
    df['lyric_length'] = df['full_lyrics'].apply(lambda x: len(x.split(' ')))
    df = df[df['lyric_length'] > 20]
    df = df.reset_index(drop=True)

    # Look at the lyrical variety (# of unique words/total word count)
    df['lyric_variety'] = df['full_lyrics'].apply(lambda x: len(set(x.split(' '))) / len(x.split(' ')))

    # Check the dominant language of the lyrics- if it is not english, 'foreign language' should be 1
    df['dominant_language'] = df['full_lyrics'].apply(detect)
    df['foreign_language'] = 0
    df.loc[df['dominant_language'] != 'en', 'foreign_language'] = 1

    # Calculate the difficulty of the lyrics using the Flesch Reading Ease score
    df['lyric_difficulty'] = cleaning_text_data(df['lyrics'], keep_breaks=True)
    df['lyric_difficulty'] = df.apply(difficulty, axis=1)

    # For songs in foreign languages, use the average difficulty value of songs in English
    df.loc[df['lyric_difficulty'].isnull(), 'lyric_difficulty'] = np.mean(df['lyric_difficulty'])

    # Use named entities as a proxy for how specific the lyrics are
    df['num_entities'] = 0
    df['num_entities'] = df['full_lyrics'].apply(entity)

    # Use VADER for sentiment analysis
    analyser = SentimentIntensityAnalyzer()
    df['sentiment'] = df['full_lyrics'].apply(lambda x: analyser.polarity_scores(x)['compound'])
    return df


def year_end_nlp(df, df_lyrics):
    """
    Using tfidf and PCA to quantify the lyrics of popular music from the year before the one-hit wonder.
    For each one hit wonder in the following year find the distance from the average/center (since we're using PCA)
    This represents how much the song differs, lyrically, from the most popular music from the prior year.
    :param df: The dataframe of potential one-hit wonders
    :param df_lyrics: The dataframe of the lyrics from the year-end billboard charts
    :return: The dataframe of potential one-hit wonders with a score comparing the one-hit lyrics to the average song
    lyrics at the time
    """
    df_lyrics['full_lyrics'] = cleaning_text_data(df_lyrics['Lyrics'], remove_words=False)
    df_lyrics['lyric_length'] = df_lyrics['full_lyrics'].apply(lambda x: len(x.split(' ')))
    df_lyrics = df_lyrics[df_lyrics['lyric_length'] > 20]
    df_lyrics['dominant_language'] = df_lyrics['full_lyrics'].apply(detect)
    df_lyrics = df_lyrics[df_lyrics['dominant_language'] == 'en']
    df_lyrics = df_lyrics[df_lyrics['Year'] >= 1979]
    df_lyrics = df_lyrics.reset_index(drop=True)
    df_lyrics = df_lyrics[['Year', 'full_lyrics', 'year_end_100']]
    df_lyrics.columns = ['year', 'full_lyrics', 'SongID']
    
    one_hit_lyrics = df[df['dominant_language'] == 'en']
    one_hit_lyrics = one_hit_lyrics[['year', 'full_lyrics', 'SongID']]
    year_list = sorted(one_hit_lyrics['year'].unique())
    df_distance_lyrics = pd.DataFrame(columns=['SongID', 'lyrical_distance'])
    
    for year in year_list:
        one_hit_temp_df = one_hit_lyrics[one_hit_lyrics['year'] == year]
        year_end_temp_df = df_lyrics[df_lyrics['year'] == year - 1]
        clean_lyrics_one_hit = cleaning_text_data(one_hit_temp_df['full_lyrics'])
        clean_lyrics_year_end = cleaning_text_data(year_end_temp_df['full_lyrics'])

        # Fit on the lyrics from the hits from the  prior year, transform the lyrics of the one hit wonder
        X_train, X_test = vectorize_data(clean_lyrics_year_end, clean_lyrics_one_hit, min_df=3, max_df=.2)
        pca = PCA(7, random_state=4)
        doc_topic = topic_model(pca, X_train, X_test, terms_per_topic=5)

        # Just need to calculate the distance from the center (0,0,0), since PCA centers the data
        one_hit_temp_df['lyrical_distance'] = np.sqrt(np.sum(np.square(doc_topic[0]), axis=1))
        one_hit_temp_df = one_hit_temp_df[['SongID', 'lyrical_distance']]
        df_distance_lyrics = pd.concat([df_distance_lyrics, one_hit_temp_df])
        
    df = pd.merge(df, df_distance_lyrics, how='left', on='SongID')
    # If missing a lyrical distance, use the mean
    df.loc[df['lyrical_distance'].isnull(), 'lyrical_distance'] = df['lyrical_distance'].mean()
    return df


def genre_score(df, genre_dict):
    """
    Calculate the genre score of a song by seeing how many songs in the prior year's billboard top 100 chart had the
    same genre, counting up the number of songs, and returning the score of the most popular genre. Meant to be used
    with an apply function.
    :param df: Dataframe with potential one-hit wonders.
    :param genre_dict: A dictionary of genres, the year, and the genre's popularity. Created by year_end_genres.
    :return: The score of the genre with the maxium popularity
    """
    year = str(df['year']-1)
    list_scores = [0]
    for genre in df['genres']:
        try:
            key = genre+year
            list_scores.append(genre_dict[key])

        # If key error, the genre of the one hit wonder wasn't in any songs from the prior year, so just continue
        except KeyError:
            continue
    return max(list_scores)


def year_end_genres(df, df_genre_summary):
    """
    Compares the genres of the one hit wonders to genres of popular songs from the prior year and returns a score
    :param df: Dataframe with the potential one-hit wonders
    :param df_genre_summary: The genre summary from import_yearly_data
    :return: Dataframe of potential one-hit wonders with a genre score
    """
    key = df_genre_summary['genre'] + df_genre_summary['year'].astype(str)
    genre_dict = dict(zip(key, df_genre_summary['frequency']))
    df['genre_score'] = df.apply(genre_score, axis=1, genre_dict=genre_dict)
    return df


def year_end_features(df, df_features):
    """
    Compares the features of one-hit wonders to the features of the top songs from the prior year. Uses scaling and
    euclidean distance to measure this.
    :param df: Dataframe with all of the one-hit wonders
    :param df_features: Dataframe with the songs from the year end chart and their lyrics
    :return: Dataframe of one-hit wonders with score comparing their features to popular songs at the time
    """
    df_features = df_features[df_features['year'] >= 1979]
    df_features = df_features.reset_index(drop=True)
    columns = list(df_features.columns)
    columns.remove('year_end_100')
    columns.append('SongID')
    df_features.columns = columns

    one_hit_features = df[columns]
    year_list = sorted(one_hit_features['year'].unique())
    df_distance_features = pd.DataFrame(columns=['SongID', 'feature_distance'])

    for year in year_list:
        one_hit_temp_df = one_hit_features[one_hit_features['year'] == year]
        year_end_temp_df = df_features[df_features['year'] == year - 1]
        one_hit_temp_features = one_hit_temp_df.drop(columns=['SongID', 'year'])
        year_end_temp_features = year_end_temp_df.drop(columns=['SongID', 'year'])
        std = StandardScaler()
        std.fit(year_end_temp_features)
        temp_features = std.transform(one_hit_temp_features)
        one_hit_temp_df.loc[:, 'feature_distance'] = np.sqrt(np.sum(np.square(temp_features), axis=1))
        one_hit_temp_df = one_hit_temp_df[['SongID', 'feature_distance']]
        df_distance_features = pd.concat([df_distance_features, one_hit_temp_df])

    df = df.merge(df_distance_features, on='SongID')
    return df


def general_features(df):
    """
    Creates some features not specific to a category
    :param df: Dataframe of potential one-hit wonders
    :return: Dataframe of one-hit wonders with some new columns
    """
    # Create target column, change featured performer and mode to binomial features
    df['Featured_Performer'] = df['Featured_Performer'].apply(lambda x: (x is not None) * 1)
    df['mode_1'] = 1
    df.loc[df['mode'] != 1, 'mode_1'] = 0

    # Miscellaneous cleaning
    df = df.rename(columns={'Week Position': 'peak_position'})
    df = df[df['genres'].apply(lambda x: x != [])]
    df = df.reset_index(drop=True)

    # Our best interaction term
    df['peak_pos_chart_div'] = df['Weeks on Chart'] / df['peak_position']

    # Change years to decades
    df['decade'] = 0
    mask80 = (df['year'] < 1990) & (df['year'] >= 1980)
    mask90 = (df['year'] < 2000) & (df['year'] >= 1990)
    mask00 = (df['year'] < 2010) & (df['year'] >= 2000)
    mask10 = df['year'] >= 2010
    df.loc[mask80, 'decade'] = '80s'
    df.loc[mask90, 'decade'] = '90s'
    df.loc[mask00, 'decade'] = '00s'
    df.loc[mask10, 'decade'] = '10s'

    # Create the target
    df['one_hit'] = df['total_songs_on_chart'].where(df['total_songs_on_chart'] == 1, 0)
    return df


def genre_consolidation(df):
    """
    Consolidates all the genres returned by Spotify into 6 categories: dance, country, rock, hip hop, and pop
    :param df: Dataframe of the potential one hit wonders
    :return: Dataframe of the potential one hit wonders with genres consolidated
    """
    # Genre consolidation
    df['dance'] = ((df['dance pop'] == 1) | (df['hip house'] == 1) | (df['disco'] == 1) | (
                df['post-disco'] == 1)) * 1
    df = df.drop(columns=['disco', 'post-disco'])

    df['country'] = ((df['country'] == 1) | (df['country road'] == 1) | (
                df['contemporary country'] == 1) | (df['country rock'] == 1) | (
                                       df['modern country rock'] == 1) | (df['folk'] == 1) | (
                                       df['traditional folk'] == 1)) * 1
    df = df.drop(
        columns=['country road', 'contemporary country', 'country rock', 'modern country rock', 'folk',
                 'traditional folk'])

    df['rock'] = ((df['rock'] == 1) | (df['glam metal'] == 1) | (
                df['alternative metal'] == 1) | (df['post-grunge'] == 1) | (
                                    df['alternative rock'] == 1) | (df['hard rock'] == 1) | (
                                    df['mellow gold'] == 1) | (df['neo mellow'] == 1) | (
                                    df['pop rock'] == 1) | (df['album rock'] == 1) | (
                                    df['soft rock'] == 1) | (df['permanent wave'] == 1) | (
                                    df['dance rock'] == 1) | (df['new romantic'] == 1) | (
                                    df['new wave'] == 1) | (df['art rock'] == 1) | (
                                    df['blues rock'] == 1) | (df['british invasion'] == 1) | (
                                    df['classic garage rock'] == 1) | (df['classic rock'] == 1) | (
                                    df['folk rock'] == 1) | (df['heartland rock'] == 1) | (
                                    df['modern rock'] == 1) | (df['merseybeat'] == 1) | (
                                    df['psychedelic rock'] == 1) | (df['rock-and-roll'] == 1) | (
                                    df['rockabilly'] == 1) | (df['roots rock'] == 1) | (
                        df['yacht rock'])) * 1
    df = df.drop(
        columns=['glam metal', 'alternative metal', 'post-grunge', 'alternative rock', 'art rock', 'hard rock',
                 'mellow gold', 'neo mellow', 'album rock', 'soft rock', 'permanent wave', 'dance rock', 'new romantic',
                 'new wave', 'classic garage rock', 'classic rock', 'folk rock', 'heartland rock', 'modern rock',
                 'psychedelic rock', 'rock-and-roll', 'rockabilly', 'roots rock', 'yacht rock'])

    df['pop'] = ((df['pop'] == 1) | (df['dance pop'] == 1) | (df['europop'] == 1) | (
                df['pop rap'] == 1) | (df['post-teen pop'] == 1) | (df['pop rock'] == 1) | (
                                   df['new wave pop'] == 1) | (df['hip pop'] == 1) | (
                                   df['synthpop'] == 1) | (df['brill building pop'] == 1) | (
                                   df['british invasion'] == 1) | (df['bubblegum pop'] == 1) | (
                                   df['merseybeat'] == 1)) * 1
    df = df.drop(
        columns=['europop', 'post-teen pop', 'pop rock', 'new wave pop', 'synthpop', 'dance pop', 'brill building pop',
                 'british invasion', 'bubblegum pop', 'merseybeat'])

    df['r&b'] = ((df['r&b'] == 1) | (df['urban contemporary'] == 1) | (
                df['new jack swing'] == 1) | (df['quiet storm'] == 1) | (df['neo soul'] == 1) | (
                                   df['funk'] == 1) | (df['blues rock'] == 1) | (
                                   df['classic soul'] == 1) | (df['doo-wop'] == 1) | (
                                   df['motown'] == 1) | (df['rhythm and blues'] == 1) | (
                                   df['soul'] == 1) | (df['southern soul'] == 1) | (df['jazz funk'] == 1)) * 1
    df = df.drop(
        columns=['urban contemporary', 'new jack swing', 'quiet storm', 'neo soul', 'funk', 'blues rock',
                 'classic soul', 'doo-wop', 'motown', 'rhythm and blues', 'soul', 'southern soul', 'jazz funk'])

    df['hip hop'] = ((df['hip hop'] == 1) | (df['atl hip hop'] == 1) | (df['rap'] == 1) | (
                df['gangster rap'] == 1) | (df['southern hip hop'] == 1) | (
                                       df['dirty south rap'] == 1) | (df['east coast hip hop'] == 1) | (
                                       df['trap'] == 1) | (df['hardcore hip hop'] == 1) | (
                                       df['pop rap'] == 1) | (df['hip pop'] == 1)) * 1
    df = df.drop(
        columns=['hip house', 'rap', 'gangster rap', 'hip pop', 'atl hip hop', 'southern hip hop', 'dirty south rap',
                 'east coast hip hop', 'trap', 'hardcore hip hop', 'pop rap'])
    df = df.drop(columns=['adult standards', 'lilith', 'easy listening', 'lounge', 'singer-songwriter'])
    return df


def feature_engineering(df, df_lyrics, df_genre_summary, df_features):
    """

    :param df: Dataframe of the  potential one-hit wonders
    :param df_lyrics: Dataframe of the lyrics from the billboard year-end charts
    :param df_genre_summary: Dataframe of the genres from the billboard year-end charts
    :param df_features: Dataframe of the features from the billboard year-end charts
    :return: Dataframe that is ready for modeling
    """
    # Let's first work on features that require the entire dataset
    df = time_sensitive_features(df, 4, 40)

    # Since we no longer need songs prior to 1980, let's filter the data by date
    mask_year = (df['year'] <= 2016) & (df['year'] >= 1980)
    df = df[mask_year]
    df = df.reset_index(drop=True)

    # NLP for the song lyrics
    df = nlp_features(df)

    # Compare one-hit wonder lyrics to previous year's lyrics
    df = year_end_nlp(df, df_lyrics)

    # Compare one-hit wonder genres to previous year's genres
    df = year_end_genres(df, df_genre_summary)

    # Engineer and clean up some miscellaneous features
    df = general_features(df)

    # Compare one-hit wonder audio features to previous year's audio features
    df = year_end_features(df, df_features)

    # Consolidate the genres returned by Spotify into a handful of genres
    df = genre_consolidation(df)

    # Remove columns we don't need for modeling
    df = df.drop(
        columns=['writers', 'producers', 'track_href', 'gen_artist', 'spot_artist', 'num_single_chart', 'uri', 'id',
                 'type', 'features', 'spot_ratio', 'gen_ratio', 'analysis_url', 'mode', 'time_signature', 'energy',
                 'Song', 'Performer', 'SongID', 'Primary_Performer', 'datetime', 'first_week', 'lyrics', 'genres',
                 'full_lyrics', 'dominant_language', 'key', 'lyric_length', 'total_songs_on_chart','year'])
    return df


