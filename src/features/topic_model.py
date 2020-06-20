def display_topics(model, feature_names, no_top_words, topic_names=None):
    """
    Displays the top n terms in each topic
    :param model: Fitted model
    :param feature_names: The names of the features in the corpus
    :param no_top_words: The number of words to show for each topic
    :param topic_names: A list of the topic names, if you want the names to appear on the print out (default no)
    :return: Nothing
    """
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix + 1)
        else:
            print("\nTopic: '", topic_names[ix], "'")
        top_features = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        print(top_features)


def topic_model(model, X_train, X_test, columns=False, terms_per_topic=5):
    """
    Performs topic modeling on the given model.
    :param model: A dimensionality reduction model with a fit_transform function (e.g. NMF)
    :param X_train: A vectorized corpus that we want to train the dimensionality reduction model on
    :param X_test: A vectorized corpus that we want to perform dimensionality reduction model on
    :param columns: The names of the columns in the corpus (default the columns from X_train)
    :param terms_per_topic: How many terms we want to show per topic (default 5)
    :return: Returns the doc_topic matrix and the fitted model.
    """
    model.fit(X_train)
    doc_topic = model.transform(X_test)
    if type(columns) != bool:
        display_topics(model, columns, terms_per_topic)
    else:
        display_topics(model, X_train.columns, terms_per_topic)
    return doc_topic, model
