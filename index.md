### *Project Proposal:* **Discover Technology Trends through Patent Analysis**
 ## Introduction
 Patents and trademarks are often a very good indication of the direction in which a field of research is evolving. In this project, we aim to procure and analyze millions of patents, published academic work and trademarks. The procured documents must range over a wide period of time (~20 years) and over diverse fields in order give us a sense of how academic research has evolved over time and how inter and cross disciplinary research has grown.
 
 ## Dataset:
 The US Patent and Trademark Office has made all its patent and trademark information freely available for public use. The database consists of 9.4 million records from 1981. For the newest patents not included in the dataset, data mining techniques can be used to retrieve patents. 
 This data is available in XML and JSON formats out of which, I chose to work with data in JSON format owing to ease of working with JSON structures. Following code extracts titles of patents from json data of patents information. Since the extraction of text from huge json files is very time consuming, I am dumping the list of titles into pickle files so as to be able to load the data whenever required.
 
 ```python
 
import sys
sys.path.insert(0, 'DataCollection/')

from DataCollection import dataCollection, loadPickle

# Extract titles of all patents of the 2016 year
# Takes 2016.json and dumps into 2016.pkl
dataCollection('DataCollection/2018.json')

titlesList = loadPickle('DataCollection/2016.pkl')
```

 ## Preliminary Patent Data Analysis

Once the titles have been collected, I import the data from the pickle file which is stored in our local directory. We import the NLTK Stop Word List and remove blank spaces as well from each title. After this, I build a bag of words model from the title data. Our TFIDF function returns a sparse TFIDF matrix, a list of unique words and a doctionary of word counts.

```python
# TFIDF

from tfidf_and_cosine import tfidf
number_of_docs=5000
x=titlesList[1:number_of_docs]
x=[xx.encode('UTF8') for xx in x]

#####STOP WORD REMOVAL START########################

test_1=[]
for sentence in x:
    sentence=sentence.lower()
    text = ' '.join([word for word in sentence.split() if word not in (stopwords.words('english'))])
    test_1.append(text)
    
##############STOP WORD REMOVAL END############################

sparse_mat, word_list, dict_words = tfidf(x)
```

