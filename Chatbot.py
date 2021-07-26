# Importing the required libraries

import numpy as np # packages for scientific computing in Python.
import nltk  #Natural language processing library
import string
import random # built-in module to generate the pseudo-random variables

# Importing and reading the corpus

f = open('chatbot.txt','r', errors='ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower() #Converts text to lowercase
nltk.download('punkt') #Using the Punkt tokenizer
nltk.download('wordnet') #Using the WordNet dictionary
sent_tokens = nltk.sent_tokenize(raw_doc) #Convert doc list of sentences
word_tokens = nltk.word_tokenize(raw_doc) # Converts doc to list of words

# Example of sentence tokens

sent_tokens[:2]

# Example of word tokens

word_tokens[:2]

# Text preprocessing

lemmer = nltk.stem.WordNetLemmatizer() # WordNet is a semantically-oriented dictionary of English included in NLTK
def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict  = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Defining the greeting function

GREET_INPUTS = ("hello","hi","greetings","sup","what's up","hey")
GREET_RESPONSES = ["hi" ,"hey" , "*nods*" , "hi there" , "hello" , " I am glad! you are talking to me"]
def greet(sentence):

  for word in sentence.split():
    if word.lower() in GREET_INPUTS:
      return random.choice(GREET_RESPONSES)
   
 # Responses generation

from sk.learn.feature_extraction.text import TfidfVectorizer # Tfid- Term frequency(frequecy of occurence of words) and inward document frequency(rare words)
from sklearn.metrics.pairwise import cosine_similarity # word in binary

def response(user_response):
  robo1_response= '  '
  TfidVec = TfidVectorizer(tokenizer = LemNormalize, stop_words='english')
  tfidf = TfidVec.fit_transform(sent_tokens)
  vals = cosine_similarity(tfidf[-1],tfidf)
  idx = vals.argssort()[0][-2]
  flat = vals.flatten()
  flat.sort()
  req_tfidf = flat[-2]
  if (req_tfidf==0):
   robo1_response=robo1_response+" I am sorry! I don't understand you"
   return robo1_response 
  else:
    robo1_response = robo1_response+sent_tokens[idx]
    return robo1_response

# Defining conversation start/end protocals

flag=True
print("BOT : My name is Ivory. Let's have a converstion! Also, if you want to exit any time, just type Bye!")
while(flag==True):
  user_response = input()
  user_response = user_response.lower()
  if(user_response!='bye'):
    if(user_response=='thanks' or user_response =='thank you'):
      flag = False
      print("BOT: You are welcome..")
    else:
      if(greet(user_response)!= None):
        print("BOT: "+greet(user_response))
      else:
        sent_tokens.append(user_response)
        word_tokens = word_tokens+nltk.word_tokenize(user_response)
        final_words = list(set(word_tokens))
        print("BOT: ", end="")
        print(response(user_response))
        sent_tokens.remove(user_response)
  else:
    flag==False
    print("BOT: Goodbye ! Take care <3")
























































