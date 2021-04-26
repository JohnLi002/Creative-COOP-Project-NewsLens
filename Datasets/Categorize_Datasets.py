import csv

with open('Web_Scraping/Keyword_Extraction/Datasets/Reuters_set.csv', mode = 'w', encoding = 'utf-8', newline='') as news_set:
    copier = csv.writer(news_set, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    with open('Web_Scraping/Keyword_Extraction/articles3.csv', mode = 'r', encoding = 'utf-8', newline='', errors='ignore') as data:
        reader = csv.reader(data, delimiter = ',')
        cols = [1,2,3,4,5,6,7,8,9]
        first = True
        for row in reader:
            new_row = []
            if first:
                first = False
            elif row[3] != "Reuters":
                continue
            for i in cols:
                new_row.append(row[i])
            
            copier.writerow(new_row)