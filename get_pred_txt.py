import pandas as pd 

csv = pd.read_csv("./data/news_articles_ner.csv")
articles = csv.iloc[:,4]
sentences = []
for i in range(len(articles)):
    sentences.append(articles[i].split('.'))
sentences = sum(sentences, [])
new = []

for i, str in enumerate(sentences):
    if str == "":
        sentences.pop(i)


for i in sentences:
    if len(i) > 0 and i[0] == " ":
        new.append(i[1:])
    else:
        new.append(i)

output = "./data/articles_sentence.txt" 
with open(output, 'w+') as file: 
    file.write('\n'.join(new))
