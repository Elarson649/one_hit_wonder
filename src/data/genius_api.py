import lyricsgenius
import re
import string
genius = lyricsgenius.Genius("BsO-T-uHHHn6Gv0a7fKebCHpaES4au-tzMiNv8pXwQX9rXn8rsEGNXgQUj7oMchA")


def genius_api(df):
    """
    Uses the Genius API to get the song lyrics, names of producers and writers, and the artist of the song.
    :param df: A row of the dataframe with the 'Song' and 'Primary Performer' columns. Typically used with apply
    function
    :return: A list containing the lyrics, a list of the writers, a list of producers, and the name of the artist
    """
    try:
        # First try to look-up the song by the primary performer and the song title
        song = df['Song'].split(' (')[0]
        song = song.split('/')[0]
        song = re.sub('[%s]' % re.escape(string.punctuation), '', song)
        performer = df['Primary_Performer']
        song_result = genius.search_song(song, performer)

        # If we don't get results, try looking up the song title
        if not song_result:
            song_result = genius.search_song(song)

        lyrics = song_result.lyrics
        writers = [writer['name'] for writer in song_result.writer_artists]
        producers = [producer['name'] for producer in song_result.producer_artists]
        artist = song_result.artist
        return [lyrics, writers, producers, artist]
    except:
        return [None, [], [], None]