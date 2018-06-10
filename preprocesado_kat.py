import csv
import re

from sklearn import svm   #SVM algorithm
from sklearn.feature_extraction.text import TfidfVectorizer #TF IDF
from sklearn.feature_extraction.text import CountVectorizer #Bag of words
from sklearn.naive_bayes import GaussianNB                  #Naive Bayes


stopWords = []
def processTweet(tweet):
	tweet = tweet.lower()
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
	tweet = re.sub('@[^\s]+','AT_USER',tweet)
	tweet = re.sub('[\s]+', ' ', tweet)
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	tweet = tweet.strip('\'"')
	return tweet


def readFile(fp,file): 
	    reader = csv.reader(file)
	    for (label, tweet) in reader:
	    	if label != None:
	    		fp.write(label+' '+processTweet(tweet)+'\n')
#-----------------------------------------------------------
def getStopWordList(fileName):
	stopWords = []
	stopWords.append('AT_USER')
	stopWords.append('URL')
	fp = open(fileName, 'r')
	line = fp.readline()
	while line:
		word=line.strip()
		stopWords.append(word)
		line = fp.readline()
	fp.close()
	return stopWords

def TwoOrMoreWords(s):#--->?
	pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
	return pattern.sub(r"\1\1",s)
		
def getFeatureVector(tw):
	featureVector = []
	label = tw.split()[0]
	words = tw.split()[1:]
	for w in words:
		w = TwoOrMoreWords(w)
		w = w.strip('\'"?,.')
		val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
		if(w in stopWords or val is None):
			continue
		else:
			featureVector.append(w.lower())
	return featureVector



#--------------------Preprocess-----------------------------
stopWords = getStopWordList('stopwords.txt')
with open('example.csv','r',newline='') as Rfile: 
#example.csv archivo con los tweets
	with open('example1.csv', 'w',newline='') as Wfile:
	#example1.csv archivo donde se guardan los tweet sin  url y stopwords
		reader = csv.reader(Rfile)
		writer = csv.writer(Wfile)
		for (label, tweet) in reader:
			processedTweet = processTweet(tweet)
			featureVector = getFeatureVector(processedTweet)
			if len(featureVector)!= 0:
				writer.writerow([label]+[' '.join(featureVector)])


# ------------------------- BAG OF WORDS ----------------------#

print("Vector caracteristico")

corpus = []
labels = []

with open('example1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        tweet = row[1]
        label = row[0]
        corpus.append(tweet)
        labels.append(label)

    print(corpus)



############# BAG OF WORDS #############

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
bag = X.toarray()

print("Bag of words de entrenamiento:")
print(bag)
print(labels)



################ TF IDF ################

TF_IDF = TfidfVectorizer(min_df=1)
sklearn_representation = TF_IDF.fit_transform(corpus)
tfidf = sklearn_representation.toarray()

print("TF_IDF de entrenamiento:")
print(tfidf)
print(labels)


# ---------------- VARIABLES ------------------- #
positivos = []
negativos = []
Vector_pos = []
Vector_neg = []

# --------------------
bag_Pos= []
bag_Neg= []



with open('positivos.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        pos = row[0]
        positivos.append(pos)

        processedTweet = processTweet(pos)
        featureVector = getFeatureVector(processedTweet)
        Vector_pos.append(" ".join(featureVector))
        frase = " ".join(featureVector)

        #---------------- BoW
        #bag_Pos.extend(vectorizer.transform([frase]).toarray())
        #---------------- Tf Idf
        bag_Pos.extend(TF_IDF.transform([frase]).toarray())

    print("Vector de positivos:")
    print(positivos)
    print("Vector de feature vectors:")
    print(Vector_pos)

# ----------------- vector of pos ---------#
print(bag_Pos)




with open('negativos.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        neg = row[0]
        negativos.append(neg)

        processedTweet = processTweet(neg)
        featureVector = getFeatureVector(processedTweet)
        Vector_neg.append(" ".join(featureVector))
        frase = " ".join(featureVector)

        #---------------- BoW
        #bag_Neg.extend(vectorizer.transform([frase]).toarray())
        #---------------- Tf Idf
        bag_Neg.extend(TF_IDF.transform([frase]).toarray())

    print("Vector de negativos:")
    print(negativos)
    print("Vector de feature vectors:")
    print(Vector_neg)

# ----------------- vector of neg ---------#
print(bag_Neg)


"""
# ------------------ Predict -----------------#

# ------------------------------------- SVM ----------------------------------------#
clf = svm.SVC()
clf.fit(bag, labels)

print("Negativos 0 - Positivos 1")

print("predict negativos")
print(clf.predict(bag_Neg))

print("predict positivos")
print(clf.predict(bag_Pos))


# ------------------------------------ NAIVE BAYES ----------------------------------#
NB = GaussianNB()
NB.fit(bag, labels)

print("Negativos 0 - Positivos 1")

print("predict negativos")
print(NB.predict(bag_Neg))

print("predict positivos")
print(NB.predict(bag_Pos))
"""


# ------------------ Predict with TF IDF -----------------#

# ------------------------------------- SVM ----------------------------------------#
clf = svm.SVC()
clf.fit(tfidf, labels)

print("Negativos 0 - Positivos 1")

print("predict negativos")
print(clf.predict(bag_Neg))

print("predict positivos")
print(clf.predict(bag_Pos))


# ------------------------------------ NAIVE BAYES ----------------------------------#
NB = GaussianNB()
NB.fit(tfidf, labels)

print("Negativos 0 - Positivos 1")

print("predict negativos")
print(NB.predict(bag_Neg))

print("predict positivos")
print(NB.predict(bag_Pos))

