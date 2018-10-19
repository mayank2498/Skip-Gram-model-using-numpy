import numpy as np
import string
from nltk.corpus import stopwords 

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class word2vec(object):
    def __init__(self):
        self.N = 10
        self.X_train = []
        self.y_train = []
        self.window_size = 2
        self.alpha = 0.001
        self.words = []
        self.word_index = {}

    def initialize(self,V,data):
        self.V = V
        self.W = np.random.uniform(-0.8, 0.8, (self.V, self.N))
        self.W1 = np.random.uniform(-0.8, 0.8, (self.N, self.V))
        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i
#        self.W = np.random.randn(self.V,self.N)
#        self.W1 = np.random.randn(self.N,self.V)
    
    def feed_forward(self,X):
        self.h = np.dot(self.W.T,X).reshape(self.N,1)
        self.u = np.dot(self.W1.T,self.h)
        #print(self.u)
        self.y = softmax(self.u)  
        return self.y
        
    def backpropagate(self,x,t):
        e = self.y - np.asarray(t).reshape(self.V,1)
        # e.shape is V x 1
      
        dLdW1 = np.dot(self.h,e.T)
        X = np.array(x).reshape(self.V,1)
        dLdW = np.dot(X, np.dot(self.W1,e).T)
        self.W1 = self.W1 - self.alpha*dLdW1
        self.W = self.W - self.alpha*dLdW
        
    def train(self,epochs):
        for x in range(1,epochs):  
              
            self.loss = 0
            for j in range(len(self.X_train)):
                self.feed_forward(self.X_train[j])
                self.backpropagate(self.X_train[j],self.y_train[j])
                C = 0
                for m in range(self.V):
                    if(self.y_train[j][m]):
                        self.loss += -1*self.u[m][0]
                        C += 1
                self.loss += C*np.log(np.sum(np.exp(self.u)))
            print("epoch ",x, " loss = ",self.loss)
            self.alpha *= 1/( (1+self.alpha*x) )
    def predict(self,word,number_of_predictions):
        if word in self.words:
            index = self.word_index[word]
            X = [0 for i in range(self.V)]
            X[index] = 1
            prediction = self.feed_forward(X)
            output = {}
            for i in range(self.V):
                output[prediction[i][0]] = i
            
            top_context_words = []
            for k in sorted(output,reverse=True):
                top_context_words.append(self.words[output[k]])
                if(len(top_context_words)>=number_of_predictions):
                    break
    
            return top_context_words
        else:
            print("Word not found in dicitonary")


def preprocessing(corpus):
    stop_words = set(stopwords.words('english'))    
    training_data = []
    sentences = corpus.split(".")
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
        sentence = sentences[i].split()
        x = [word.strip(string.punctuation) for word in sentence if word not in stop_words]
        x = [word.lower() for word in x]
        training_data.append(x)
    return training_data
    

def prepare_data_for_training(sentences,w2v):
    data = {}
    for sentence in sentences:
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1
    V = len(data)
    data = sorted(list(data.keys()))
    vocab = {}
    for i in range(len(data)):
        vocab[data[i]] = i
    
    #for i in range(len(words)):
    for sentence in sentences:
        for i in range(len(sentence)):
            center_word = [0 for x in range(V)]
            center_word[vocab[sentence[i]]] = 1
            context = [0 for x in range(V)]
            for j in range(i-w2v.window_size,i+w2v.window_size):
                if i!=j and j>=0 and j<len(sentence):
                    context[vocab[sentence[j]]] += 1
            w2v.X_train.append(center_word)
            w2v.y_train.append(context)
    w2v.initialize(V,data)

    return w2v.X_train,w2v.y_train    

corpus = ""
corpus += "The World Wide Web (WWW), also called the Web, is an information space where documents and other web resources are identified by Uniform Resource Locators (URLs), interlinked by hypertext links, and accessible via the Internet.[1] English scientist Tim Berners-Lee invented the World Wide Web in 1989. He wrote the first web browser in 1990 while employed at CERN in Switzerland.[2][3] The browser was released outside CERN in 1991, first to other research institutions starting in January 1991 and to the general public on the Internet in August 1991. The World Wide Web has been central to the development of the Information Age and is the primary tool billions of people use to interact on the Internet.[4][5][6] Web pages are primarily text documents formatted and annotated with Hypertext Markup Language (HTML).[7] In addition to formatted text, web pages may contain images, video, audio, and software components that are rendered in the user's web browser as coherent pages of multimedia content. Embedded hyperlinks permit users to navigate between web pages. Multiple web pages with a common theme, a common domain name, or both, make up a website. Website content can largely be provided by the publisher, or interactively where users contribute content or the content depends upon the users or their actions. Websites may be mostly informative, primarily for entertainment, or largely for commercial, governmental, or non-governmental organisational purpose "
corpus += "The World Wide Web had a number of differences from other hypertext systems available at the time. The Web required only unidirectional links rather than bidirectional ones, making it possible for someone to link to another resource without action by the owner of that resource. It also significantly reduced the difficulty of implementing web servers and browsers (in comparison to earlier systems), but in turn presented the chronic problem of link rot. Unlike predecessors such as HyperCard, the World Wide Web was non-proprietary, making it possible to develop servers and clients independently and to add extensions without licensing restrictions. On 30 April 1993, CERN announced that the World Wide Web would be free to anyone, with no fees due.[25] Coming two months after the announcement that the server implementation of the Gopher protocol was no longer free to use, this produced a rapid shift away from Gopher and towards the Web. An early popular web browser was ViolaWWW for Unix and the X Windowing System"
corpus += "Connected by the Internet, other websites were created around the world. This motivated international standards development for protocols and formatting. Berners-Lee continued to stay involved in guiding the development of web standards, such as the markup languages to compose web pages and he advocated his vision of a Semantic Web. The World Wide Web enabled the spread of information over the Internet through an easy-to-use and flexible format. It thus played an important role in popularising use of the Internet.[29] Although the two terms are sometimes conflated in popular use, World Wide Web is not synonymous with Internet.[30] The Web is an information space containing hyperlinked documents and other resources, identified by their URIs.[31] It is implemented as both client and server software using Internet protocols such as TCP/IP and HTTP. Berners-Lee was knighted in 2004 by Queen Elizabeth II for services to the global development of the Internet"
corpus += "The terms Internet and World Wide Web are often used without much distinction. However, the two are not the same. The Internet is a global system of interconnected computer networks. In contrast, the World Wide Web is a global collection of documents and other resources, linked by hyperlinks and URIs. Web resources are usually accessed using HTTP, which is one of many Internet communication protocols"
corpus += "Viewing a web page on the World Wide Web normally begins either by typing the URL of the page into a web browser, or by following a hyperlink to that page or resource. The web browser then initiates a series of background communication messages to fetch and display the requested page. In the 1990s, using a browser to view web pages—and to move from one web page to another through hyperlinks—came to be known as 'browsing,' 'web surfing' (after channel surfing), or 'navigating the Web'. Early studies of this new behaviour investigated user patterns in using web browsers. One study, for example, found five user patterns: exploratory surfing, window surfing, evolved surfing, bounded navigation and targeted navigation"

training_data = preprocessing(corpus)
w2v = word2vec()
X,y = prepare_data_for_training(training_data,w2v)
w2v.train(1000)    

#print(w2v.predict("results",5))    
    
    

#def generate_training_data(words,w2v):
#    data = {}
#    for word in words:
#        if word not in data:
#            data[word] = 1
#        else:
#            data[word] += 1
#    V = len(data)
#    data = sorted(list(data.keys()))
#    vocab = {}
#    for i in range(len(data)):
#        vocab[data[i]] = i
#    
#    #for i in range(len(words)):
#    for i in range(len(words)):
#        center_word = [0 for i in range(V)]
#        center_word[vocab[words[i]]] = 1
#        context = [0 for i in range(V)]
#        for j in range(i-w2v.window_size,i+w2v.window_size):
#            if i!=j and j>=0 and j<len(words):
#                
#                context[vocab[words[j]]] += 1
#        w2v.X_train.append(center_word)
#        w2v.y_train.append(context)
#    print(data)
#    w2v.initialize(V,data)
#
#    return w2v.X_train,w2v.y_train
    
    