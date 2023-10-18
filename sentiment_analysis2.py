import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

# data 

data = [
    ["Neutral","This was ok"],
    ["Neutral", "This was fine"],
    ["Neutral", "This was not bad and not good"],
    ["Neutral", "I found this to be all right"],
    ["Negative", "This was bad"],
    ["Negative", "I didnt like this"],
    ["Negative", "This was the worst I've ever had"],
    ["Negative", "I hated this so much!"],
    ["Positive", "This was good"],
    ["Positive", "This was the best thing I've ever had"],
    ["Positive", "This was amazing! Wow"],
    ["Positive", "I loved this"]
]

dataframe = pd.DataFrame(data, columns=["Label","Review"])

labels = dataframe["Label"]
number_of_clusters = np.unique(labels).shape[0]

def vectorize_text(sentences, vocab=None, min_df=0.0, max_df=1.0, ngram_range=(1,1)):
    count_vectorizer = CountVectorizer(max_df = max_df,
                                        min_df=min_df,
                                        vocabulary=vocab,
                                        ngram_range=(1,1))
    sentences_vectorized = count_vectorizer.fit(sentences)
    feature_names = sentences_vectorized.get_feature_names_out()
    bag_of_words = sentences_vectorized.transform(sentences)
    dataframe_bag_of_words = pd.DataFrame(bag_of_words.todense(), columns=feature_names)

    transformer = TfidfTransformer()
    words_matrix = transformer.fit_transform(bag_of_words)
    word_counts = np.asarray(bag_of_words.sum(axis=0)).ravel().tolist()
    dataframe_counts = pd.DataFrame({"word": feature_names, "count": word_counts})

    dataframe_counts = dataframe_counts.sort_values("count", ascending=False)
    weights = np.asarray(words_matrix.mean(axis=0)).ravel().tolist()
    dataframe_weights = pd.DataFrame({"word":feature_names, "weight":weights})
    dataframe_weights = dataframe_weights.sort_values("weight", ascending=True)
    dataframe_weights = dataframe_weights.merge(dataframe_counts, on ="word", how="left")
    word_similarity = cosine_similarity(words_matrix, words_matrix)
    distance_matrix = 1 - word_similarity
    return sentences_vectorized, feature_names, dataframe_bag_of_words, words_matrix, dataframe_weights, word_similarity, distance_matrix

sentences = dataframe["Review"].values.tolist()
sentences_vectorized, feature_names, dataframe_bag_of_words, words_matrix, dataframe_weights, word_similarity, distance_matrix = vectorize_text(sentences)

dataframe_words_matrix = pd.DataFrame(words_matrix.todense(), columns=feature_names)
print(dataframe_words_matrix.to_string())