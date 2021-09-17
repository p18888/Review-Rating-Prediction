from django.shortcuts import render,HttpResponse

# import nltk
# from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import numpy as np # linear algebra

import re # for regex
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
import pickle
import string
from sklearn.ensemble import RandomForestClassifier
# nltk.download(all)

# Data cleaaning Function

def removing_email_address(text):

    return text.replace(r'^.+@[^\.].*\.[a-z]{2,}$','')

def removing_emojis(text):
    return text.encode('ascii', 'ignore').decode('ascii')

def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text) # substring replace with ''(space)

def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

def to_lower(text):
    return text.lower()

def removing_phone_number(text):
    return text.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','')

def remove_numbers(text):
    return re.sub(r'\d','',text)

def remove_word_less_than_2(text):
    return [w for w in text.split() if len(w)>2]

def rem_stopwords(text):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 
    'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', 
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
    'won', "won't", 'wouldn', "wouldn't"]
    return [w for w in text if w not in stop_words]

def stem_txt(text):
    lemma_words=[]
    for word in text:
        # word = WordNetLemmatizer().lemmatize(word,pos='v')
        lemma_words.append(word)
    return ' '.join(lemma_words)
loaded_model = pickle.load(open('rating_bow.pkl', 'rb'))
loaded_model = pickle.load(open('rating_model_rf.pkl', 'rb'))
def predict_rating(text):
    rem_email=removing_email_address(text)
    rem_emoji=removing_emojis(rem_email)
    cln=clean(rem_emoji)
    isspecial=is_special(cln)
    tolower=to_lower(isspecial)
    removing_phonenumber=removing_phone_number(tolower)
    removenumbers=remove_numbers(removing_phonenumber)
    remove_word_lessthan2=remove_word_less_than_2(removenumbers)
    remstopwords=rem_stopwords(remove_word_lessthan2)
    stemtxt=stem_txt(remstopwords)
    bow,words = [],stemtxt.split()
    for word in words:
        bow.append(words.count(word))
    inp = []
    for i in vocabularies:
        inp.append(stemtxt.count(i[0]))
    y_pred = loaded_model.predict(np.array(inp).reshape(1,15259))
    return y_pred[0]














def index(request):    
    return render(request,"pavan/index.html")
def result(request):
    if request.GET.get('review_text'):
        try:
            if request.GET.get('review_text'):

                num1 = request.GET.get('review_text','')
                result = predict_rating(num1)
        except Exception as e:
            print("ERROR",e)
            result = None
        return render(request,"pavan/result.html",{"name":result})
    else:
        return render(request,"pavan/result.html",{"name":''})

def home(request):
    return HttpResponse("Hello World")
# def result(request):    
#     return HttpResponse("Hello World")
