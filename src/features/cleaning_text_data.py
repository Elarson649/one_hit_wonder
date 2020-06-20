import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string


def part_of_speech(pos):
    """
    Updates the pos to wordnet so we can use the lemmatizer.
    :param pos: The part of speech via nltk.pos_tag.
    :return: Returns the wordnet part of speech for adjectives, verbs, nouns, and adverbs. Default is noun.
    """

    if pos.startswith('J'):
        return wordnet.ADJ
    elif pos.startswith('V'):
        return wordnet.VERB
    elif pos.startswith('N'):
        return wordnet.NOUN
    elif pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def cleaning_text_data(corpus, remove_words=True, keep_breaks=False, allowed_pos=False, words_to_remove=[]):
    """
    Returns the cleaned corpus. Removes html tags, websites, punctuation, numbers, stop words, and lemmatizes the
    corpus.
    :param corpus: The corpus to be cleaned.
    :param remove_words: Determines if we want to remove stop words from the corpus. The default is yes
    :param keep_breaks: Determines if we want to keep line breaks- the default is to get rid of them.
    :param allowed_pos: Which parts of speech to exclude from the corpus. Enter false to include everything (the
     default is false).
    :param words_to_remove: Words to be removed from the corpus, outside of the stopwords in nltk.corpus.
    :return: The cleaned corpus.
    """
    cleaned_corpus = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english') + words_to_remove)
    if not keep_breaks:
        corpus = corpus.replace({'\n': ' '}, regex=True)
    corpus = list(corpus)
    for document in corpus:
        document_token = []
        document = re.sub('\w*\d\w*', ' ', document)
        document = document.lower()
        document = re.sub('<.*?>', '', document)
        document = re.sub('\[.*?\]', '', document)
        document = re.sub('\{.*?\}', '', document)
        document = re.sub('[%s]' % re.escape(string.punctuation), '', document)
        document = word_tokenize(document)
        for word in document:
            if (word in stop_words) & remove_words:
                continue
            else:
                pos = nltk.pos_tag([word])[0][1]
                # POS of interest for filtering are NNP, NN*, V*, and J*. Allowed_pos should be a tuple containing these
                # types. If the word doesn't meet any of these, we should exclude it from the vector.
                if allowed_pos:
                    if not pos.startswith(allowed_pos):
                        continue
                wordnet_pos = part_of_speech(pos)
                word = lemmatizer.lemmatize(word, wordnet_pos)
                document_token.append(word)
        cleaned_corpus.append(' '.join(document_token))
    return cleaned_corpus
