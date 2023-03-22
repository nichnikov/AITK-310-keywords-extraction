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

tokenizer = TextsTokenizer()
# syns_df = pd.read_csv(os.path.join("data", "synonyms.csv"), sep="\t")

stopwords = []
stopwords_roots = [os.path.join("data", "greetings.csv"),
                   os.path.join("data", "stopwords.csv")]

for root in stopwords_roots:
    stopwords_df = pd.read_csv(root, sep="\t")
    stopwords += list(stopwords_df["stopwords"])

tokenizer.add_stopwords(stopwords)
bigram_measures = BigramAssocMeasures()

data_df = pd.read_csv(os.path.join("data", "etalons.csv"), sep="\t")
# freq_words = nltk.FreqDist(texts.split())
print(list(set(data_df['ID'])))
print(len(list(set(data_df['ID']))))
fields = ["ID", "Topic", "ShortAnswerText"]
results = []
for fa_id in list(set(data_df['ID'])):
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

    txt = " ".join(tokens)
    if bigrams:
        for bg in bigrams:
            txt = re.sub(bg, "_".join(bg.split()), txt)

    freq_words = nltk.FreqDist(txt.split())
    # freq_words.plot(10)
    words_freq = [(w, freq_words[w]) for w in freq_words]
    words_freq5 = [(w, fr) for w, fr in words_freq if fr >= 5]

    if len(words_freq5) >= 5:
        freq_words_dict["FreqWords"] = words_freq5
    else:
        freq_words_dict["FreqWords"] = words_freq[:5]

    results.append(freq_words_dict)
    print(freq_words_dict)

results_df = pd.DataFrame(results)
results_df["FreqWords"] = results_df["FreqWords"].apply(lambda x: re.sub(r'[\[\]\']', "", str(x)))

print(results_df)
results_df.to_csv(os.path.join("data", "freq_words.csv"), sep="\t", index=False)