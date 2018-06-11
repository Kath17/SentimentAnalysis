import csv
import re

from sklearn import svm   #SVM algorithm
from sklearn.naive_bayes import GaussianNB                  #Naive Bayes
from sklearn.feature_extraction.text import TfidfVectorizer #TF IDF
from sklearn.feature_extraction.text import CountVectorizer #Bag of words

# ----------------------------------------------------------------------------------------------------#

# ---------------- Read File (Archivo preprocesado) : Devuelve el corpus, y los labels --------------- #

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

print("Creando vectores característicos: ")

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

                #processedTweet = processTweet(tweet)
                #featureVector = getFeatureVector(processedTweet)

                #lista_vectores.append(" ".join(tweet))
                frase = "".join(tweet)

                print(frase)
                #transformar de acuerdo a la tecnica elegida
                vectoresCarac.extend(To_transform.transform([frase]).toarray())

        print(vectoresCarac)
        return vectoresCarac

# --------------------------------- Usar algoritmo ---------------------------#

def Use_Algorithm( algoritmo , bag, labels, to_predict):
        print("Negativos 0 - Positivos 1")
        if(algoritmo == "svm" ):
                print("Usando el SVM")
                #clf = svm.SVC(decision_function_shape='ovo')
                clf = svm.SVC()
                clf.fit(bag, labels)
                resultado = clf.predict(to_predict)
                print(resultado)
        elif ( algoritmo == "nb" ):
                print("Usando el Naive Bayes")
                NB = GaussianNB()
                NB.fit(bag, labels)
                resultado = NB.predict(to_predict)
                print(resultado)
        else:
                print("Algoritmo no soportado")



######################################## MAIN #######################################

# ---------------- Read File (Archivo preprocesado) --------------- #

(corpus, labels) = Read_File('example1.csv')
#print(corpus)
#print(labels)


# ------------------------- Tecnica para vectores ----------------------#
#vectorizer depende de la tecnica requerida

(vectorizer, bag) = TecnicaParaVector("tfidf",corpus)
print("\nBag of words de entrenamiento:")
print(bag[0])

# ----------------- Vectores caracteristicos de prueba positivos ---------#
bag_Pos = Get_Vector('negativos.csv',vectorizer)
print("\nBolsa de palabras positivas")
#print(bag_Pos)

"""
# ----------------- Vectores caracteristicos de prueba negativos ---------#
bag_Neg = Get_Vector('negativos.csv',vectorizer)
print("\nBolsa de palabras negativas")
#print(bag_Neg)
"""

# ------------------ Predict with Bag of Words -----------------#

# ------------------- SVM -----------------#
#print("Probando negativos")
#Use_Algorithm("svm", bag, labels, bag_Neg)
print("Probando positivos")
Use_Algorithm("svm", bag, labels, bag_Pos)

# ------------------- Naive Bayes -----------------#
#print("Probando negativos")
#Use_Algorithm("nb", bag, labels, bag_Neg)
print("Probando positivos")
Use_Algorithm("nb", bag, labels, bag_Pos)





