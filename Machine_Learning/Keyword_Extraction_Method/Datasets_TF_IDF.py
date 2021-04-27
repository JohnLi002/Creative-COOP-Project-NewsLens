import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer

All_The_Words = []
titles = []
with open('Web_Scraping/Keyword_Extraction/Datasets/NewYorkTimes_set.csv', mode = 'r', encoding = 'utf-8', errors='ignore') as news_set:
    csv_reader = csv.reader(news_set, delimiter = ',')
    first = True
    for row in csv_reader:
        if first:
            first = False
            continue
        All_The_Words.append(row[8])
        titles.append(row[1])

# settings that you use for count vectorizer will go here 
tfidf_vectorizer=TfidfVectorizer(use_idf=True, stop_words=["the", "of", "in", "to", "is"]) 
 
# just send in all your docs here 
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(All_The_Words)


first = True
with open('Web_Scraping/Keyword_Extraction/Keyword_Extraction_Method/Keyword_Results/NewYorkTimes_result.csv', mode = 'w', encoding = 'utf-8', newline='') as news_set:
    copier = csv.writer(news_set, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    for row in range(len(titles)):
        string = []
        vector_tfidfvectorizer=tfidf_vectorizer_vectors[0] 
        if first:
            copier.writerow(["Title", "Keyword1", "Keyword2", "Keyword3", "Keyword4", "Keyword5", "Keyword6", "Keyword7", "Keyword8", "Keyword9", "Keyword10"])
            first = False
        else:
            string.append(titles[row])
            vector_tfidfvectorizer=tfidf_vectorizer_vectors[row-1]
            df = pd.DataFrame(vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=['TF-IDF'])
            df = df.sort_values('TF-IDF', ascending = False).index.tolist()
            for words in range(10):
                string.append(str(df[words]))
        copier.writerow(string)
                