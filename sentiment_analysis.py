import pandas as pd
import nltk
import unicodedata, string, re

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

print(dataframe_tokens)