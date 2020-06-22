import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from joblib import load
from PIL import Image
import matplotlib.pyplot as plt


from flask import Flask, request, render_template,url_for


#%%

ps = PorterStemmer()
pipeline = load('text_classification.joblib')


def create_corpus(text):
    review = re.sub('[^a-zA-Z]', ' ',text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    corpus = ' '.join(review)
    return corpus

def clean_text(text):
    replaced = re.sub("</?p[^>]*>", "", text)
    replaced = re.sub("\n","",replaced)
    replaced = re.sub("â","",replaced)
    final = re.sub('\W+',' ', replaced)
    return final

#%%

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        title = request.form['title']
        text = request.form['text']
        ip1 = clean_text(title + ' ' + text)
        ip2 = create_corpus(ip1)
        result = pipeline.predict([ip2])

        
    return render_template('result.html', prediction=result)

# %%

if __name__ == "__main__":
    app.run(debug=True)