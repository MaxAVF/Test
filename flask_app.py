from flask import Flask, render_template, request

from urllib.request import urlopen
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import pickle
import pandas as pd

app = Flask(__name__)
#es = Elasticsearch('10.0.1.10', port=9200)

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search/results', methods=['GET', 'POST'])
def search_request():
    search_term = request.form["input"]
    res = recommendations(search_term)
    return render_template('results.html', res=res )

#import json
#with open('data.json') as f:
#    datamath = json.load(f,encoding="UTF-8")

dataset='https://www.dropbox.com/s/q8am1p40irfrq9c/arxiv_dataset.csv?dl=1'
url_test='http://www.dropbox.com/s/v1dotm2fdnxplkd/df_test?dl=1'
datamath=pickle.load(urlopen(url_test))#pd.read_csv(dataset,sep=';') # pd.read_csv('dataset_app.csv',sep=';')

titles = datamath['Title']
summaries = datamath['Abstract']
link = datamath['Id'] #[datamath["entries"][i]["link"] for i in range(len(datamath["entries"]))]
tags= datamath['Categories']

model_url='www.dropbox.com/s/o8aiyptawyqpxmq/model.sav?dl=1'
tfmatrix_url='https://www.dropbox.com/s/bnapqsl19i8r76m/matrixtf?dl=1'
tf = pickle.load(urlopen(model_url))    #'model.sav', 'rb'))
tfmatrix = pickle.load(urlopen(tfmatrix_url))   #open('matrixtf', 'rb'))

Titles = pd.Series(titles)
indices = pd.Series(summaries, index=titles)
Links = pd.Series(link)
Abstracts = pd.Series(summaries)
Tags=pd.Series(tags)

def recommendations(search):
    #idx = titles.index(title)
    sim_scores = list(enumerate(linear_kernel(tfmatrix, tf.transform([search]))))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    paper_indices = [i[0] for i in sim_scores]
    sim_scores=[i[1] for i in sim_scores]
    df=pd.DataFrame({'Title': Titles.iloc[paper_indices],'Abstract': Abstracts.iloc[paper_indices], \
                     'Link': Links.iloc[paper_indices], 'Categories' : Tags.iloc[paper_indices],\
                     'Cosine Similiarity': sim_scores})
    #print(title)
    return df






if __name__ == '__main__':
    app.secret_key = 'mysecret'
    app.run(host='0.0.0.0', port=3000)


