import csv
import re

from sklearn import svm   #SVM algorithm
from sklearn.naive_bayes import GaussianNB                  #Naive Bayes
from sklearn.feature_extraction.text import TfidfVectorizer #TF IDF
from sklearn.feature_extraction.text import CountVectorizer #Bag of words

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
	if(len(tw) == 0):
		return featureVector
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


# ----------------------------------------------------------------------------------------------------#

# ---------------- Read File (Archivo preprocesado) : Devuelve el corpus, y los labels --------------- #

print("Creando vectores característicos: ")

def Read_File(nombre_archivo):
        corpus = []
        labels = []
        with open(nombre_archivo) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                tweet = row[1]
                label = row[0]
                corpus.append(tweet)
                labels.append(label)
        return (corpus, labels)


# ------------------------------ BAG OF WORDS -----------------------------#
# Creamos un bag of words del corpus, se guarda en la variable bag (lista de listas)   //  bow

#---------------------------------- TF IDF --------------------------------#
# Usamos TF IDF para crear los vectores caracteristicos del corpus     //  tfidf

def TecnicaParaVector( tecnica,corpus ):
        if(tecnica == "bow"):
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(corpus)
                bag = X.toarray()
                return (vectorizer,bag)
        elif(tecnica == "tfidf"):
                TF_IDF = TfidfVectorizer(min_df=1)
                sklearn_representation = TF_IDF.fit_transform(corpus)
                tfidf = sklearn_representation.toarray()
                return (TF_IDF, tfidf)


# ------------------------------- Get Vector -----------------------------#
# Preprocesa los datos para probar y retorna el vector caracteristico de los datos de prueba
# To_transform = tf idf or BoW

def Get_Vector(nombre_archivo, To_transform):
        vectoresCarac = []
        lista_vectores = []

        print("\nTweets preprocesados para probar:\n")
        
        with open(nombre_archivo) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                tweet = row[0]
                lista_vectores.append(tweet)

                processedTweet = processTweet(tweet)
                featureVector = getFeatureVector(processedTweet)

                lista_vectores.append(" ".join(featureVector))
                frase = " ".join(featureVector)

                print(frase)
                #transformar de acuerdo a la tecnica elegida
                vectoresCarac.extend(To_transform.transform([frase]).toarray())
        
        return vectoresCarac

# --------------------------------- Usar algoritmo ---------------------------#

def Use_Algorithm( algoritmo , bag, labels, to_predict):
        print("Negativos 0 - Positivos 1")
        if(algoritmo == "svm" ):
                print("Usando el SVM")
                clf = svm.SVC()
                clf.fit(bag, labels)
                print(clf.predict(to_predict))
        elif ( algoritmo == "nb" ):
                print("Usando el Naive Bayes")
                NB = GaussianNB()
                NB.fit(bag, labels)
                print(NB.predict(to_predict))
        else:
                print("Algoritmo no soportado")



######################################## MAIN #######################################

# ---------------- Read File (Archivo preprocesado) --------------- #

(corpus, labels) = Read_File('example1.csv')
print(corpus)
print(labels)


# ------------------------- Tecnica para vectores ----------------------#
#vectorizer depende de la tecnica requerida

(vectorizer, bag) = TecnicaParaVector("bow",corpus)
print("\nBag of words de entrenamiento:")
print(bag[0])

# ----------------- Vectores caracteristicos de prueba positivos ---------#
bag_Pos = Get_Vector('positivos.csv',vectorizer)
print("\nBolsa de palabras positivas")
print(bag_Pos)

# ----------------- Vectores caracteristicos de prueba negativos ---------#
bag_Neg = Get_Vector('negativos.csv',vectorizer)
print("\nBolsa de palabras negativas")
print(bag_Neg)


# ------------------ Predict with Bag of Words -----------------#

# ------------------- SVM -----------------#
print("Probando negativos")
Use_Algorithm("svm", bag, labels, bag_Neg)
print("Probando positivos")
Use_Algorithm("svm", bag, labels, bag_Pos)

# ------------------- Naive Bayes -----------------#
print("Probando negativos")
Use_Algorithm("nb", bag, labels, bag_Neg)
print("Probando positivos")
Use_Algorithm("nb", bag, labels, bag_Pos)


"""
# ------------------------- TF IDF ----------------------#
(TF_IDF, tfidf) = To_TfIdf()
print("TF_IDF de entrenamiento:")
print(tfidf)

# ----------------- Vectores caracteristicos de prueba positivos ---------#
bag_Pos = Get_Vector('positivos.csv','t',TF_IDF)
print(bag_Pos)

# ----------------- Vectores caracteristicos de prueba negativos ---------#
bag_Neg = Get_Vector('negativos.csv','t',TF_IDF)
print(bag_Neg)


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
"""
