import spotipy
import time
import re
import string
from spotipy.oauth2 import SpotifyClientCredentials
client_id = '227b08bbd5644a7992a211d68f175200'
client_secret = 'd21ae4c965ad49ddbeccb830b9c40990'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def spotify_api(df):
    """
    Uses the Spotify API to get the audio features for a song and the genre(s) of the song's artist
    :param df: A row of the dataframe with the 'Song' and 'Primary Performer' columns. Typically used with apply
    function
    :return: A list containing the features, artist, and genre(s) of the artist
    """
    time.sleep(.05)
    try:
        # First try to look-up the song by the primary performer and the song title
        song = df['Song'].split(' (')[0]
        song = song.split('/')[0]
        song = re.sub('[%s]' % re.escape(string.punctuation), '', song)
        performer = df['Primary_Performer']
        query = 'track:' + song + ' artist:' + performer
        result = sp.search(query, type='track')

        # If we don't get results, try looking up the song title
        if not result['tracks']['items']:
            query = 'track:' + song
            result = sp.search(query, type='track')

        # Get the features from the track, the genres and the artist name from the artist page. Will use artist name
        # later on for verification purposes
        track_uri = result['tracks']['items'][0]['id']
        artist_uri = result['tracks']['items'][0]['artists'][0]['id']
        features = sp.audio_features(track_uri)
        artist = sp.artist(artist_uri)['name']
        genres = sp.artist(artist_uri)['genres']
        return [features, artist, genres]

    except:
        return [None, None, None]