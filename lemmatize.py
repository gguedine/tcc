import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

wordnet_lemmatizer = WordNetLemmatizer()


def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:          
    return None


def lemmatize_sentence(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
  res_words = []
  for word, tag in wn_tagged:
    if tag is None:            
      res_words.append(word)
    else:
      res_words.append(wordnet_lemmatizer.lemmatize(word, tag))
  return " ".join(res_words)


def perform_lemmatize_dataset(data):
    i = 0
    for setting_value in data['setting_value']:
        data.at[i, 'setting_value'] = lemmatize_sentence(setting_value)
        i += 1
    return data