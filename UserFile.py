import pickle

import nltk
from nltk.corpus import wordnet



def prepos(sentence):

    words = nltk.word_tokenize(sentence)
    
    new_words = []
    
    temp_word = ''
    for word in words:
        antonyms = []
        if word == 'not':
            temp_word = 'not_'
        elif temp_word == 'not_':
            for syn in wordnet.synsets(word):
                for s in syn.lemmas():
                    for a in s.antonyms():
                        antonyms.append(a.name())
            if len(antonyms) >= 1:
                word = antonyms[0]
            else:
                word = temp_word + word
            temp_word = ''
        if word != 'not':
            new_words.append(word)
    
    sentence = ' '.join(new_words)
    print(sentence)
    return [sentence]



# Using our classifier
with open('TFIDF.pickle','rb') as f:
    tfidf = pickle.load(f)
    
with open('LogistiClassifier.pickle','rb') as f:
    clf = pickle.load(f)
    
    
#sample = "Hello , not beautiful person"
while True:
    sample = input("ENter:")
    if sample=="X":
        break
    sample = prepos(sample)
    sample = tfidf.transform(sample).toarray()
    sentiment = clf.predict_proba(sample)
    
    #print(sentiment)
    
    if(sentiment[0][1]>sentiment[0][0]):
        print("Positive: ",int((sentiment[0][1]+sentiment[0][2])*100),"%")
        print("Negative :",int((sentiment[0][0])*100),"%")
    else:
        print("Negative :",int((sentiment[0][0]+sentiment[0][2])*100),"%")
        print("Positive: ",int((sentiment[0][1])*100),"%")
    
