import csv

from sklearn import svm   #SVM algorithm
from sklearn.naive_bayes import GaussianNB                  #Naive Bayes
from sklearn.feature_extraction.text import TfidfVectorizer #TF IDF
from sklearn.feature_extraction.text import CountVectorizer #Bag of words

# ----------------------------------------------------------------------------------------------------#

# ----------------------- Separar datos de entrenamiento con datos de preuba ----------------#
def Separar(nombre_archivo,porcentaje_entrenar,num_elem):
        with open(nombre_archivo,'r',newline='') as Rfile:
                with open('Entrenar_'+nombre_archivo, 'w',newline='') as Wfile:
                        with open('Probar_'+nombre_archivo, 'w',newline='') as Pfile:
                                reader = csv.reader(Rfile)
                                writer = csv.writer(Wfile)
                                writerP = csv.writer(Pfile)

                                counter = 0
                                limite_E = int((porcentaje_entrenar * num_elem)/100)
                                print(limite_E)
                                for (sentiment, tweet) in reader:
                                        counter = counter+1
                                        if counter <= limite_E:
                                                writer.writerow([sentiment]+[tweet])
                                        else:
                                                writerP.writerow([sentiment]+[tweet])
        return ('Entrenar_'+nombre_archivo , 'Probar_'+nombre_archivo)

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
        lista_labels = []

        print("\nTweets preprocesados para probar:\n")
        
        with open(nombre_archivo) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                label = row[0]
                tweet = row[1]
                lista_vectores.append(tweet)
                lista_labels.append(label)

                #processedTweet = processTweet(tweet)
                #featureVector = getFeatureVector(processedTweet)

                #lista_vectores.append(" ".join(tweet))
                frase = "".join(tweet)

                #print(frase)
                #transformar de acuerdo a la tecnica elegida
                vectoresCarac.extend(To_transform.transform([frase]).toarray())

        print(vectoresCarac[0])
        return vectoresCarac , lista_labels

# --------------------------------- Usar algoritmo ---------------------------#
# Retorna porcentaje de acierto

def Use_Algorithm( algoritmo , bag, labels_e, to_predict, labels_prueba):
        #print("Negativos 0 - Positivos 1")
        acierto_svm = 0
        acierto_nb = 0
        total = 0
        
        if(algoritmo == "svm" ):
                #C = 1.0
                print("Usando el SVM")
                #clf = svm.SVC(decision_function_shape='ovo')
                clf = svm.SVC(kernel='linear')
                clf.fit(bag, labels_e)
                resultado = clf.predict(to_predict)
                for i in resultado:
                        if(str(i) == str(labels_prueba[total])):
                                acierto_svm = acierto_svm +1
                        total = total + 1
                        #print(resultado[i])
                #print(resultado)
                return (acierto_svm/total)
        elif ( algoritmo == "nb" ):
                print("Usando el Naive Bayes")
                NB = GaussianNB()
                NB.fit(bag, labels_e)
                resultado = NB.predict(to_predict)
                for i in resultado:
                        if(str(i) == str(labels_prueba[total])):
                                acierto_nb = acierto_nb + 1
                        total = total + 1
                #print(resultado)
                return (acierto_nb/total)
        else:
                print("Algoritmo no soportado")
        return 0


######################################## MAIN #######################################

# ----------------------- Separar datos en dos archivos --------------------#
(archivo_entrenar , archivo_probar) = Separar('dataTraining.csv',80,4200)
# training.8000.csv # dataTraining.csv 4200
print((archivo_entrenar , archivo_probar))

# ---------------- Read File (Archivo preprocesado) --------------- #
# Separa el corpus y los labels

(corpus, labels_entrenamiento) = Read_File(archivo_entrenar)
print("\nDatos del corpus preprocesados:")
print(corpus[0])
print("\nDatos del label:")
print(labels_entrenamiento)


# ------------------------- Tecnica para vectores ----------------------#
# vectorizer saldrá de acuerdo a la tecnica requerida

print("\nCreando vectores de los datos de entrenamiento:")
(vectorizer_bow, bag_bow) = TecnicaParaVector("bow",corpus)
(vectorizer_tfidf, bag_tfidf) = TecnicaParaVector("tfidf",corpus)

print(bag_bow[0])
print(bag_tfidf[0])

# ----------------- Vectores caracteristicos de datos de prueba ---------#
print("\nCreando vectores de los datos de prueba de acuerdo a bow o tfidf")
datos_probar_bow , lista_labels_bow = Get_Vector(archivo_probar,vectorizer_bow)
datos_probar_tfidf, lista_labels_tfidf = Get_Vector(archivo_probar,vectorizer_tfidf)

print(datos_probar_bow[0])
print(datos_probar_tfidf[0])


# --------------------------------- Predict ----------------------------------#

# ------------------- SVM -----------------#
#Use_Algorithm("svm", bag, labels, bag_Neg)s
print("\nProbando con datos de prueba en el SVM")
print("\nProbando con datos bow")
acierto_bow_svm = Use_Algorithm("svm", bag_bow, labels_entrenamiento, datos_probar_bow , lista_labels_bow)
print("Acierto de bow en SVM:", acierto_bow_svm )

print("\nProbando con datos tfidf")
acierto_tfidf_svm = Use_Algorithm("svm", bag_tfidf, labels_entrenamiento, datos_probar_tfidf , lista_labels_tfidf)
print("Acierto de tfidf en SVM:", acierto_tfidf_svm )

# ------------------- Naive Bayes -----------------#
#print("Probando negativos")
#Use_Algorithm("nb", bag, labels, bag_Neg)
print("\nProbando con datos de prueba en el Naive Bayes")
print("\nProbando con datos bow")
acierto_bow_nb = Use_Algorithm("nb", bag_bow, labels_entrenamiento, datos_probar_bow , lista_labels_bow)
print("Acierto de bow en NB:", acierto_bow_nb )

print("\nProbando con datos tfidf")
acierto_tfidf_nb = Use_Algorithm("nb", bag_tfidf, labels_entrenamiento, datos_probar_tfidf , lista_labels_tfidf)
print("Acierto de tfidf en NB:", acierto_tfidf_nb )




