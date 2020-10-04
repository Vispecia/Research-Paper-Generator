#!/usr/bin/python

import bs4
import requests
import re
import string

from stop_words import get_stop_words
en_stop = get_stop_words('en')

from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()
words = []
def preprocessing():
    url = "https://en.wikipedia.org/wiki/Convolutional_neural_network"
    response = requests.get(url)

    if response is not None:
        page = bs4.BeautifulSoup(response.text,'html.parser')
        title = page.select("#firstHeading")[0].text
        p = page.select("p")

        #appending all paragraphs text
        data = '\n'.join([para.text for para in p])
        # data = data.encode("utf-8")
        #to lower case letter
        data = data.lower()
        #data = data.decode('utf-8')
        #to remove digits
        data = re.sub(r'\d+', '', data)
        #to remove special chars
        data = data.translate(str.maketrans("","",string.punctuation))
        #to split into words
        words = data.split()
        #removing stop words (is the a)
        words = [i for i in words if not i in en_stop]
        words = [p_stemmer.stem(i) for i in words]
        appendFile = open('cleanText.txt','a',encoding='utf-8') 
        for r in words:          
            appendFile.write(" "+r)

        appendFile.close() 
        return title    





