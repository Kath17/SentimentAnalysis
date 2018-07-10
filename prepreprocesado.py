import pandas as pd
import numpy as np

def Preparar_data(nombre):
    data = pd.read_csv('Original/'+nombre)
    data = data[['text','sentiment']]
    data = data[data.sentiment != "Neutral"]
    data = data.sort_values('sentiment')

    texto = data.text
    labels = data.sentiment

    # ------------------ Guardamos los datos en un csv
    df_save = pd.DataFrame(texto)
    df_label = pd.DataFrame(labels)

    result = pd.concat([df_label, df_save], axis = 1)
    result.to_csv('DataBase/BD_'+nombre, index=False)

def Preparar_data2(nombre):
    data = pd.read_csv('Original/'+nombre, encoding = "ISO-8859-1")
    data = data[['Sentiment','SentimentText']]

    texto = data.SentimentText
    labels = data.Sentiment
    # ------------------ Guardamos los datos en un csv
    df_save = pd.DataFrame(texto)
    df_label = pd.DataFrame(labels)

    result = pd.concat([df_label, df_save], axis = 1)
    result.to_csv('DataBase/BD_'+nombre, index=False)

def Ordenar_data(nombre):
    data = pd.read_csv('Original/'+nombre)
    data = data.sort_values('0')

    texto = data.ix[:,1]
    labels = data.ix[:,0]

    df_save = pd.DataFrame(texto)
    df_label = pd.DataFrame(labels)

    result = pd.concat([df_label, df_save], axis = 1)
    result.to_csv('DataBase/BD_'+nombre, index=False)

def TXT_to_Csv(nombre):
    document=[]
    with open("Original/positive.txt",errors="ignore") as fp:
        line = fp.readline()
        while line:
            document.append((line,"pos"))
            line=fp.readline()

    with open("Original/negative.txt",errors="ignore") as fp:
        line=fp.readline()
        while line:
            document.append((line,"neg"))
            line=fp.readline()

    labels =['text','sentiment']
    data= pd.DataFrame.from_records(document,columns=labes)

    texto = data.ix[:,0]
    labels = data.ix[:,1]
    df_save = pd.DataFrame(texto)
    df_label = pd.DataFrame(labels)

    result = pd.concat([df_label, df_save], axis = 1)
    result.to_csv('DataBase/BD_'+nombre, index=False)

def Rating_Polarity(nombre):
    data = pd.read_excel(nombre+'.xlsx', sheet_name="COAR_Orden_Nombre",encoding='iso-8859-1')

    data = data[['rank','Comentario']]
    texto = data.Comentario
    #labels = data.rank

    data['rank'] = data['rank'].str.replace('3 de 5 estrellas','Negative')
    data['rank'] = data['rank'].str.replace('2 de 5 estrellas','Negative')
    data['rank'] = data['rank'].str.replace('1 de 5 estrellas','Negative')
    data['rank'] = data['rank'].str.replace('5 de 5 estrellas','Positive')
    data['rank'] = data['rank'].str.replace('4 de 5 estrellas','Positive')

    print(data['rank'])
    labels = data['rank']

    result = pd.concat([labels,texto], axis = 1)
    result.to_csv(nombre+'_pre.csv', index=False, encoding='utf-8-sig', header=False) #encoding='iso-8859-1')

def shuffler(filename):
  df = pd.read_csv(filename, header=0)
  # return the pandas dataframe
  return df.reindex(np.random.permutation(df.index)).to_csv('_'+filename, index=False)

#Preparar_data('Sentiment.csv')
#Ordenar_data('training.8000.csv')
#TXT_to_Csv('PosNeg.csv')
#Preparar_data2('training.1600000.processed.noemoticon.csv')
#Rating_Polarity("CorpusCOAR")

shuffler("BD_Sentiment.csv")
shuffler("BD_PosNeg.csv")
