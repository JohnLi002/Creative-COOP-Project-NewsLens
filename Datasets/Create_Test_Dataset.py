import csv

news = ['Datasets/publication_Dataset/CNN_set.csv',
        'Datasets/publication_Dataset/NewYorkTimes_set.csv',
        'Datasets/publication_Dataset/BusinessInsider_set.csv',
        'Datasets/publication_Dataset/Breitbart_set.csv',
        'Datasets/publication_Dataset/Vox_set.csv', #
        'Datasets/publication_Dataset/NPR_set.csv',
        'Datasets/publication_Dataset/NewYorkPost_set.csv', #
        'Datasets/publication_Dataset/TalkingPointsMemo_set.csv', #
        'Datasets/publication_Dataset/Guardian_set.csv',
        'Datasets/publication_Dataset/FoxNews_set.csv',
        'Datasets/publication_Dataset/Atlantic_set.csv',
        'Datasets/publication_Dataset/WashingtonPost_set.csv',
        'Datasets/publication_Dataset/NationalReview_set.csv',
        'Datasets/publication_Dataset/Reuters_set.csv'] #
with open('Datasets/test.csv', mode = 'w', encoding = 'utf-8', newline='') as news_set:
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

