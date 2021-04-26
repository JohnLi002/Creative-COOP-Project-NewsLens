from yake import KeywordExtractor

file = open("Web_Scraping/Keyword_Extraction/test.txt", 'r', encoding="utf-8")
text = file.read()

kw_extractor = KeywordExtractor(lan="en", n=1, top=5)
keywords = kw_extractor.extract_keywords(text)
for word in keywords:
    print(word)