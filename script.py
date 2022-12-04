from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def get_email_classifier_accuracy(categories):
  train_emails= fetch_20newsgroups(categories=categories, subset='train', shuffle=True, random_state=108)

  # SEE TARGET NAMES (CATEGORIES)
  # print(train_emails.target_names)

  # SEE AN INDIVIDUAL EMAIL
  # print(train_emails.data[5])
  # SEE THE EMAIL'S CLASSIFICATION
  # print(train_emails.target[5]) # => 1 (hockey)

  test_emails= fetch_20newsgroups(categories=categories, subset='test', shuffle=True, random_state=108)

  # CREATE CountVectorizer OBJECT
  counter= CountVectorizer()
  # FIT THE COUNTER WITH A LIST OF ALL THE DATA (WORDS)
  counter.fit(test_emails.data + train_emails.data)

  # MAKE LISTS OF THE COUNTS OF THE WORDS IN EACH SET
  train_counts= counter.transform(train_emails.data)
  test_counts= counter.transform(test_emails.data)

  # CREATE NAIVE BAYES CLASSIFIER OBJECT
  classifier= MultinomialNB()
  # FIT THE MODEL WITH TRAINING SET AND LABELS
  classifier.fit(train_counts, train_emails.target)
  # SEE THE MODEL SCORE
  return classifier.score(test_counts, test_emails.target) 

categories1= ['rec.sport.baseball', 'rec.sport.hockey']
categories2= ['comp.sys.ibm.pc.hardware', 'rec.sport.hockey']
categories3= ['talk.politics.mideast', 'talk.politics.misc']

print(get_email_classifier_accuracy(categories1)) # => 97.22% accuracy
print(get_email_classifier_accuracy(categories2)) # => 99.75% accuracy
print(get_email_classifier_accuracy(categories3)) # => 92.57% accuracy

# as topics get more related, the accuracy decreases.
