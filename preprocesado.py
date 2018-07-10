import csv
import re
import nltk
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
	words = tw.split()[0:]
	for w in words:
		w = TwoOrMoreWords(w) #paabras repetidas
		w = w.strip('\'"?,.') #removiendo signos
		val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
		if(w in stopWords or val is None):
			continue
		else:
			featureVector.append(w.lower())
	return featureVector

def print_vector(vector):
	for line in vector:
		print(line)
		#print("\n")

def extractFeature(tweet):
	tweetWord= set(tweets)
	features = {}
	for word in featureList:
		feature['contains(%s)' % word] = (word in tweetWord)
	return features


#--------------------Preprocess-----------------------------
# Ingles
# stopWords = getStopWordList('datasets/stopwords.txt')

#Español
stopWords = getStopWordList('stop_words_español.txt')

def Preprocesar(nombre):
	with open('datasets/'+nombre,'r',newline='') as Rfile:
	#example.csv archivo con los tweets
		with open('Pre/'+nombre, 'w',newline='') as Wfile:
		#example1.csv archivo donde se guardan los tweet sin  url y stopwords
			featureList = []
			tweets = []
			reader = csv.reader(Rfile)
			writer = csv.writer(Wfile)
			for (sentiment, tweet) in reader:
				processedTweet = processTweet(tweet)
				featureVector = getFeatureVector(processedTweet)
				if len(featureVector) > 3:
					writer.writerow([sentiment]+[' '.join(featureVector)])
					featureList.extend(featureVector)
					tweets.append((sentiment,featureVector))

		##print_vector(tweets)
		##print("\n")
		##print(featureList)
		featureList = list(set(featureList))
		##print("\n")
		#print(featureList)
		#trainingSet=nltk.classify.util.apply_feature(extractFeature,tweets)


Preprocesar('CorpusCOAR.csv')
