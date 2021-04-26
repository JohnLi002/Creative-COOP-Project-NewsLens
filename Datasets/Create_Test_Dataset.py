import csv

news = ['Web_Scraping/Keyword_Extraction/Datasets/CNN_set.csv',
        'Web_Scraping/Keyword_Extraction/Datasets/NewYorkTimes_set.csv',
        'Web_Scraping/Keyword_Extraction/Datasets/BusinessInsider_set.csv',
        'Web_Scraping/Keyword_Extraction/Datasets/Breitbart_set.csv',
        'Web_Scraping/Keyword_Extraction/Datasets/Vox_set.csv', #
        'Web_Scraping/Keyword_Extraction/Datasets/NPR_set.csv',
        'Web_Scraping/Keyword_Extraction/Datasets/NewYorkPost_set.csv', #
        'Web_Scraping/Keyword_Extraction/Datasets/TalkingPointsMemo_set.csv', #
        'Web_Scraping/Keyword_Extraction/Datasets/Guardian_set.csv',
        'Web_Scraping/Keyword_Extraction/Datasets/FoxNews_set.csv',
        'Web_Scraping/Keyword_Extraction/Datasets/Atlantic_set.csv',
        'Web_Scraping/Keyword_Extraction/Datasets/WashingtonPost_set.csv',
        'Web_Scraping/Keyword_Extraction/Datasets/NationalReview_set.csv',
        'Web_Scraping/Keyword_Extraction/Datasets/Reuters_set.csv'] #
with open('Web_Scraping/test.csv', mode = 'w', encoding = 'utf-8', newline='') as news_set:
    first = True
    copier = csv.writer(news_set, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    for news_outlet in news:
        with open(news_outlet, encoding = 'utf-8', errors ='ignore') as csv_file:
            total = 0
            first_row = True
            reader = csv.reader(csv_file, delimiter = ',')
            for row in reader:
                if first:
                    new_row = ["publication", "content"]
                    copier.writerow(new_row)
                    first = False
                    first_row = False
                elif first_row:
                    first_row = False
                    continue
                else:#if total < 1000:
                    new_row = []
                    new_row.append(row[2])
                    new_row.append(row[8])
                    copier.writerow(new_row)
                    total += 1
                #else:
                #    break

