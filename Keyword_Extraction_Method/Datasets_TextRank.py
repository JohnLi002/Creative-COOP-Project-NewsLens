from gensim.summarization import keywords
import csv

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

file = open("Web_Scraping/Keyword_Extraction/test.txt", 'r', encoding="utf-8")
string = file.read()
print(keywords(string))

