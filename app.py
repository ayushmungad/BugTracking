from flask import Flask, render_template, request
app = Flask(__name__)
from urllib.request import urlopen as ur
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import logging
from nltk.corpus import stopwords
import nltk

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
stop_words = stopwords.words('english')
stop_words.extend(['mainframe','tech','lead','work','want','developer','program','olf','br'])
import fasttext
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from scipy.cluster.hierarchy import dendrogram, linkage

corpus_list = []
cluster=0
@app.route('/')
def home():
   return render_template('form.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      for key,value in result.items():
          corpus_list.append(value)
      
      return render_template("form.html")   
      
@app.route('/scrape',methods=['POST','GET'])
def scrape():
    if request.method=='POST':
        import os, ssl
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context

        # parse the html using beautiful soup and store in variable `soup`
        from urllib.request import Request, urlopen
        corpus_text=[]
        for i in corpus_list:
            links = ['p','li']
            req = Request(i, headers={'User-Agent': 'Mozilla/5.0'})
            page = urlopen(req).read()
            soup = BeautifulSoup(page)
            for link in links:
                all_data=soup.find_all(link)
                for data in all_data:
                    data = data.text
                    data = data.encode('ascii', 'ignore').decode('ascii')
                    corpus_text.append(data)
                    
        corpus_text = list(filter(None,corpus_text))

        file=open('text_corpus.txt','w')

        for sentence in corpus_text:
            sentence = sentence.split()
            for word in sentence:
                file.writelines(word)
                file.writelines(' ')

    return render_template("input.html")



@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        file = request.files['dataset']
        if not file: return render_template('input.html', label="No file")
        #give the domain specific corpus to fasttext for training word vectors
        model = fasttext.skipgram('text_corpus.txt', 'model',lr=0.1,dim=300)
        # Reading the data to be clustered can be changed depending upon the file
        new_data=pd.read_excel(file)
        y=new_data.iloc[:,0].values
        # cleaning the sentences of the data to be clustered
        corpus=[]
        for i in range(0,len(y)):
            review = re.sub('[^a-zA-Z]',' ',y[i])
            review = review.lower()
            review = review.split()
            review = [word for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            corpus.append(review)

        #making sentence vectors, directly give the sentences to the fasttext model0
        vector=[]
        for sentence in corpus:
            vector.append(model[sentence])

        if request.method == 'POST':
            result = request.form
            for key,value in result.items():
                if(key=='clusters'):
                    cluster=value
        
        


        #cluster the vector embedding
        total_clusters = int(cluster) # may vary depending on the use case
            
        agg_cluster = AgglomerativeClustering(n_clusters = total_clusters)
        assigned_clusters = agg_cluster.fit_predict(vector)
        y_pred = pd.DataFrame(data=assigned_clusters)
        
        df = pd.DataFrame(data=y)
        df['Cluster']=y_pred.values

        writer = ExcelWriter('result.xlsx')
        df.to_excel(writer,'Sheet1',index=False)
        writer.save()
        return render_template("form.html")
         
        


if __name__ == '__main__':
   app.run(debug = True)

     