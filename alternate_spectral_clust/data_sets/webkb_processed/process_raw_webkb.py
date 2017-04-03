from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join
import autograd.numpy as np
import re
from html2text import html2text
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import csv

# list of university names
univ = ['cornell', 'texas', 'washington', 'wisconsin']
# list of topic names
topic = ['course', 'faculty', 'project', 'student']

file_names = []
label_univ = np.zeros(0)
label_topic = np.zeros(0)

# real all the html files in a directory
for idx_univ in np.arange(len(univ)):
    for idx_topic in np.arange(len(topic)):
        dir_name = univ[idx_univ] + '_' + topic[idx_topic]
        files = [join(dir_name,f) for f in listdir(dir_name) if isfile(join(dir_name, f))]
        file_names = file_names + files
        label_univ = np.concatenate((label_univ, \
                idx_univ * np.ones(len(files))))
        label_topic = np.concatenate((label_topic, \
                idx_topic * np.ones(len(files))))

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element)):
        return False
    return True

words = []
# translate html to text
for filename in file_names:
    html_file = open(filename)
    html = html_file.read()

    flag_method = 1
    if flag_method == 1:
        soup = BeautifulSoup(html)
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        texts = '\n'.join(chunk for chunk in chunks if chunk)

        #texts_all = soup.findAll(text = True)
        #texts = filter(visible, texts_all)
    elif flag_method == 2:
        texts = html2text(html)
    elif flag_method == 3:
        texts = nltk.clean_html(html)
    words.append(texts)

cv = CountVectorizer(stop_words='english', lowercase=True)
#cv = TfidfVectorizer(stop_words='english', lowercase=True)
words_all = cv.fit_transform(words).toarray()
featname_all = cv.get_feature_names()

# choose the alpha features
idx_words = []
for j in np.arange(len(featname_all)):
    if featname_all[j].isalpha():
        idx_words.append(j)
featname_alpha = np.array(featname_all)[idx_words]
words_alpha = words_all[:, idx_words]

#words_std = np.std(words_alpha, axis=0)
words_std = np.sum(words_alpha, axis=0)
# choose the high-variance words
t1 = np.argsort(words_std)
idx_highstd = t1[-500:][::-1]

featname_highstd = featname_alpha[idx_highstd]
words_highstd = words_alpha[:, idx_highstd]

# save the data into csv files
np.savetxt("webkbRaw_word.csv", words_highstd, delimiter=',')
np.savetxt("webkbRaw_label_univ.csv", label_univ.astype(int), delimiter=',')
np.savetxt("webkbRaw_label_topic.csv", label_topic.astype(int), delimiter=',')

csvfile = open("webkbRaw_wordnames.csv", "wb")
csvwriter = csv.writer(csvfile, delimiter=',')
for i in np.arange(len(featname_highstd)):
    csvwriter.writerow([featname_highstd[i]])
csvfile.close()

