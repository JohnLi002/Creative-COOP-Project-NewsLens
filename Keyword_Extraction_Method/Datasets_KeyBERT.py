from keybert import KeyBERT

file = open("Web_Scraping/Keyword_Extraction/test.txt", 'r', encoding="utf-8")
text = file.read()

model = KeyBERT('distilbert-base-nli-mean-tokens')
keywords = model.extract_keywords(text)

for word in keywords:
    print(word)