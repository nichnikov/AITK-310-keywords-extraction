# freq:
# https://www.inf.ed.ac.uk/teaching/courses/fnlp/lectures/8/tutorial.html

# n-grams:
# https://www.milindsoorya.com/blog/introduction-to-word-frequencies-in-nlp
# https://towardsdatascience.com/feature-engineering-with-nltk-for-nlp-and-python-82f493a937a0
import os
import re
import nltk
import pandas as pd
from texts_processing import TextsTokenizer
from nltk.collocations import (BigramCollocationFinder,
                               BigramAssocMeasures)


def words_frequence(splited_text: []) -> [()]:
    """
    :param splited_text: список токенов: ['token1', 'token2', ...]
    :return: список кортежей: [(слово, частота)]
    """
    freq_words = nltk.FreqDist(splited_text)
    return [(w, freq_words[w]) for w in freq_words]


def frequence_filter(words_freq: [], min_frequence: int, max_words: int) -> []:
    """
    Функция возвращает список либо по минимальной частоте либо по максимальному требуемому количеству частотных слов
    :param words_freq: список кортежей: [(слово, частота)]
    :param min_frequence: частота ниже которой не возвращается
    :param max_words: максимальное количество частотных слов, которые должны быть возвращены
    """
    words_freq_ = [(w, fr) for w, fr in words_freq if fr >= min_frequence]
    if len(words_freq_) >= max_words:
        return words_freq_
    else:
        return words_freq[:max_words]


tokenizer = TextsTokenizer()

stopwords = []
stopwords_roots = [os.path.join("data", "greetings.csv"),
                   os.path.join("data", "stopwords.csv")]

for root in stopwords_roots:
    stopwords_df = pd.read_csv(root, sep="\t")
    stopwords += list(stopwords_df["stopwords"])

tokenizer.add_stopwords(stopwords)
bigram_measures = BigramAssocMeasures()

data_df = pd.read_csv(os.path.join("data", "etalons.csv"), sep="\t")
fields = ["ID", "Topic", "ShortAnswerText"]
results = []
for num, fa_id in enumerate(list(set(data_df['ID']))):
    freq_words_dict = {"ID": fa_id}
    temp_df = data_df[data_df['ID'] == fa_id]
    freq_words_dict["GroupName"] = list(temp_df["Cluster"])[0]
    freq_words_dict["Topic"] = list(temp_df["Topic"])[0]
    freq_words_dict["ShortAnswerText"] = list(temp_df["ShortAnswerText"])[0]

    texts = " ".join(list(temp_df["Cluster"]))
    tokens = tokenizer([texts])[0]
    bigrams_finder = BigramCollocationFinder.from_words(tokens)
    bigrams_scored = bigrams_finder.score_ngrams(bigram_measures.raw_freq)
    bigrams = [" ".join(tx) for tx, sc in bigrams_scored if sc >= 0.1]

    freq_words_dict["Bigrams"] = bigrams
    txt = " ".join(tokens)

    if bigrams:
        for bg in bigrams:
            pttrn = re.compile(r"\b" + str(bg) + r"\b")
            txt = pttrn.sub("_".join(bg.split()), txt)

    words_freq = words_frequence(txt.split())
    freq_words_dict["FreqWordsBirams"] = frequence_filter(words_freq, 5, 5)
    words_freq_tk = words_frequence(tokens)
    freq_words_dict["FreqWords"] = frequence_filter(words_freq_tk, 5, 5)
    results.append(freq_words_dict)

    print(num, freq_words_dict)

results_df = pd.DataFrame(results)
for cl_name in ["Bigrams", "FreqWords", "FreqWordsBirams"]:
    results_df[cl_name] = results_df[cl_name].apply(lambda x: re.sub(r'[\[\]\']', "", str(x)))
    results_df[cl_name] = results_df[cl_name].apply(lambda x: re.sub(r'_', " ", str(x)))

print(results_df)
results_df.to_csv(os.path.join("data", "freq_words3.csv"), sep="\t", index=False)
