The Science of One-Hit Wonders
==============================

Analyzing what makes a song a one-hit wonder! Defined a one-hit wonder as an artist who only had one top 40 song during their career. Looked at songs from 1980-2016, but considered data from 1960 to 2019.
Used Spotify and Genius APIs, Billboard Weekly Hot 100 chart data, and Billboard Year End chart data to build a dataset of 1500 potential one-hit wonders, 48% of which were identified as one-hit wonders. Analyzed lyrics using a variety of NLP and unsupervised learning techniques including dimensionality reduction, clustering, sentiment analysis, part of speech tagging, TDIDF count vectorizers, and a variety of custom metrics to generate features based on lyrics. Also performed extensive feature engineering to compare one-hit wonders to the music landscape at the singles' time of release by comparing their lyrical, audio, and genre properties to the most popular music from the year prior.

Tried a variety of classification models, but focused on two: a neural network and logistic regression. The latter was used to dig into particular features via feature importance. For results, see [my blog post](https://elarson649.github.io/2020/06/17/final-project/) or the presentation featured in the repo.

Project Organization
------------

**Data**
  * External: 
    * billboard_lyrics_1964_2015.csv: Billboard year end charts
    * Hot Stuff.csv: Billboard weekly hot 100 charts
  * Interim: 
    * 0617_pre_engineering.pickle: Data after cleaning and manipulation to find one-hit wonders, but before feature engineering
  * Processed:
    * 0617_modeling_data.pickle: The data used for modeling

**Models**
  * logistic.pickle: Fitted logistic regression
  * neural.pickle: Fitted neural network

**Notebooks**
  * 0617_one_hit_wonder_final.ipynb: Notebook that walks through the data collection, cleaning, engineering, and modeling process. References functions in /src.

**Src**
  * data:
    * genius_api.py: Gets data from the Genius API.
    * import_and_clean_data.py: Imports and cleans data from the Billboard weekly hot 100 charts.
    * import_yearly_data.py: Imports and cleans data from the Billboard year-end charts.
    * spotify_api.py: Gets data from the Spotify API.
  * features:
    * cleaning_text_data.py: Cleans text data (e.g. lyrics)
    * feature_engineering.py : Performs feature engineering
    * topic_model.py: Performs dimenseionality reduction (e.g. PCA) and displays the top topics
    * vectorize_data.py: Vectorizes the data into a TFIDF vectorizer
  * models:
    * class_model.py: For a given trainval set, performs cross-validation for a particular model, shows key metrics, and returns the fitted model. Can also perform threshold tuning, dimensionality reduction, scaling, and more.
  * visualization:
    * moving_average_graph.py: Creates a moving average line graph

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
