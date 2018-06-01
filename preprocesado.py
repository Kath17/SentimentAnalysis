import csv
import re
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
stopWords = getStopWordList('datasets/stopwords.txt')
with open('datasets/example.csv','r',newline='') as Rfile: 
#example.csv archivo con los tweets
	with open('datasets/example1.csv', 'w',newline='') as Wfile:
	#example1.csv archivo donde se guardan los tweet sin  url y stopwords
		reader = csv.reader(Rfile)
		writer = csv.writer(Wfile)
		for (label, tweet) in reader:
			processedTweet = processTweet(tweet)
			featureVector = getFeatureVector(processedTweet)
			if len(featureVector)!= 0:
				writer.writerow([label]+[' '.join(featureVector)])


