import pandas as pd
import nltk
import unicodedata, string, re
import matplotlib.pyplot as plt

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

authorandquote = pd.read_csv('./author_and_quote.csv')

quotes_list = authorandquote["Quote"].tolist()
author_list = authorandquote["Author"].tolist()

# tokenization 

def remove_accents(token):
    return "".join(x for x in unicodedata.normalize("NFKD", token) if x in string.ascii_letters or x ==' ')

stopwords = nltk.corpus.stopwords.words("english")
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

RE_VALID = "\w"
MIN_STRING_LEN = 3
ALLOWED_PART_OF_SPEECH_TYPES = {"NN": "n", "JJ": "a", "VB":"v", "RB":"r"}
PART_OF_SPEECH_TYPES_KEYS = list(ALLOWED_PART_OF_SPEECH_TYPES.keys())

tokens_list = []
all_tokens_lists = []
all_lemmatized_tokens = []

for index,text in enumerate(quotes_list):
    tokens = [word.lower() for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]
    list_sentence_tokens = []
    non_lemmatized_tokens = []
    for token in tokens:
        result = remove_accents(token)
        result = str(result).translate(string.punctuation)
        list_sentence_tokens.append(result)
        non_lemmatized_tokens.append("-")

        if result not in stopwords:
            if re.search(RE_VALID, result):
                if len(result) >= MIN_STRING_LEN:
                    part_of_speech = nltk.pos_tag([result])[0][1][:2]
                    default_part_speech = "n"
                    if part_of_speech in ALLOWED_PART_OF_SPEECH_TYPES :
                        default_part_speech = ALLOWED_PART_OF_SPEECH_TYPES[part_of_speech]
                    
                    stem = stemmer.stem(result)
                    lemmatization = lemmatizer.lemmatize(result,
                                                          pos= default_part_speech)
                    if part_of_speech in PART_OF_SPEECH_TYPES_KEYS:
                        tokens_list.append((result, stem, lemmatization, part_of_speech))
                        non_lemmatized_tokens = non_lemmatized_tokens[:-1]
                        non_lemmatized_tokens.append(lemmatization)
    
    all_tokens_lists.append(list_sentence_tokens)
    lemmatized_tokens_list = " ".join(non_lemmatized_tokens)
    all_lemmatized_tokens.append(lemmatized_tokens_list)

dataframe_tokens = pd.DataFrame(all_tokens_lists)

for entry in dataframe_tokens:
    if str(dataframe_tokens[entry].dtype) in ("object", "string_", "unicode_"):
        dataframe_tokens[entry].fillna(value="", inplace=True)

dataframe_all_words = pd.DataFrame(tokens_list, columns=["token",
                                    "stem",
                                    "lemmatization",
                                    "part_of_speech"])

dataframe_all_words["counts"] = dataframe_all_words.groupby(["lemmatization"])["lemmatization"].transform("count")

dataframe_all_words = dataframe_all_words.sort_values(by=["counts", "lemmatization"],
                                ascending=[False,True]).reset_index()

# only one occurence for one lemmatization
dataframe_grouped = dataframe_all_words.groupby("lemmatization").first().sort_values(by="counts", ascending=False).reset_index()

# One dataframe per part_of_speech types
dataframe_grouped = dataframe_grouped[["lemmatization","part_of_speech", "counts"]]
for part_of_speech_type in PART_OF_SPEECH_TYPES_KEYS:
    dataframe_part_of_speech = dataframe_grouped[dataframe_grouped["part_of_speech"] == part_of_speech_type]
    print(dataframe_part_of_speech.to_string())

# plot word frequency
flatten_tokens_list = [y for x in all_tokens_lists for y in x]

token_frequency = nltk.FreqDist(flatten_tokens_list)
del token_frequency[""]
sorted_token_frequency = sorted(token_frequency.items(), 
                                key=lambda x:x[1],
                                reverse=True)
token_frequency.plot(30, cumulative=False)

# Removing the stopword then print graph
lemmatized_words = dataframe_all_words["lemmatization"].tolist()
lemmtatized_frequency = nltk.FreqDist(lemmatized_words)
sorted( lemmtatized_frequency.items(), 
       key= lambda x: x[:1],
       reverse=True)

lemmtatized_frequency.plot(30, cumulative=False)