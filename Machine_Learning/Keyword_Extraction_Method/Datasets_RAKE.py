from rake_nltk import Rake
import csv

# Uses stopwords for english from NLTK, and all puntuation characters by
# default
r = Rake()

file = open("Web_Scraping/Keyword_Extraction/test.txt", 'r', encoding="utf-8")

All_The_Words = file.read()

# Extraction given the text.
r.extract_keywords_from_text(All_The_Words)

amount = 0
# To get keyword phrases ranked highest to lowest with scores.
with open('Web_Scraping/Keyword_Extraction/Keyword_Extraction_Method/Keyword_Results/NewYorkTimes_result.csv', mode = 'w', encoding = 'utf-8', newline='') as news_set:
    copier = csv.writer(news_set, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    for row in range(len(titles)):
        string = []
        if first:
            copier.writerow(["Title", "Keyword1", "Keyword2", "Keyword3", "Keyword4", "Keyword5", "Keyword6", "Keyword7", "Keyword8", "Keyword9", "Keyword10"])
            first = False
        else:
            string.append(titles[row])
            amount = 0
            for phrase in r.get_ranked_phrases():
                if amount < 10:
                    copier.append(phrase)
                    amount+=1
                else:
                    break
        
        copier.writerow(string)
