# pip install --upgrade gensim
import os
import re
import pandas as pd
import numpy as np
import networkx as nx
import random as rd
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from gensim.models import KeyedVectors
import gensim as gs
from sklearn.decomposition import PCA

# Repertory in which we read and write every file :
os.chdir("C:/Users/Lilian/Documents/3A et +/554/Projet")


### Precompute some graph features and open files

# read training data
df_train = pd.read_csv('train.csv', dtype={'author': np.int64, 'hindex': np.float32})
n_train = df_train.shape[0]

# read test data
df_test = pd.read_csv('test.csv', dtype={'author': np.int64})
n_test = df_test.shape[0]

# load the graph
G = nx.read_edgelist('coauthorship.edgelist', delimiter=' ', nodetype=int)
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()

print("Proportion de train/data:",174242/217801*100," %")
print("Proportion de test/train:",43561/174242*100," %")
print('Number of nodes:', n_nodes)
print('Number of edges:', n_edges)

DEG = nx.degree(G)
print("DEG ok")
CN = nx.core_number(G)
print("CN ok")
EC = nx.eigenvector_centrality(G)
print("EC ok")
PR = nx.pagerank(G)
print("PR ok")
print("precomputing ok")


### Load the list of papers + some functions
def train_sample(df, ratio, seed=False):
    if seed :
        rd.seed(17)
    df['Random'] = [int(rd.random()<ratio) for _,_ in df.iterrows()]
    df_traintrain = df.loc[df['Random'] == 1].drop("Random",1)
    df_traintest = df.loc[df['Random'] == 0].drop("Random",1)
    df.drop("Random",1)
    return df_traintrain, df_traintest

def MSE(df_traintest,h_calc):
    h_column = df_traintest.loc[:,'hindex']
    h_exact = h_column.values
    MSE = (np.square(h_exact-h_calc)).mean()
    return MSE

def splitx(string):
    return (len(string)==0)*([])+(len(string)!=0)*(string.split('-'))

def in_df(df,id,index) :
    if df.iat[index,0]==id :
        return df.iat[index,1]
    return -1
# load the list of papers
df_papers = pd.read_csv('author_papers.txt',sep=":",header=None)
df_papers.columns = ["author_id", "paper_ids"]
df_papers["paper_ids"] = [splitx(ids) for ids in df_papers["paper_ids"]]


### Normalizing many features into [0,100]
def transfo_logaffine(xmin,xmax,ymin,ymax,t):
    return (np.log(t)-np.log(xmin))*(ymax-ymin)/(np.log(xmax)-np.log(xmin))+ymin

def transfo_affine(xmin,xmax,ymin,ymax,t):
    return (t-xmin)*(ymax-ymin)/(xmax-xmin)+ymin

def normalizing(D,printing=False, log=False):
    D_values = np.array(list(D.values()))
    d_max = D_values.max()
    d_min = D_values.min()
    if printing :
        print("max is : ",d_max)
        print("min is : ",d_min)
    D_norm = {}
    phi = transfo_affine
    if log:
        phi = transfo_logaffine

    for k in D.keys():
        e = D[k]
        f = phi(d_min,d_max,0,100,e)
        D_norm[k] = f
    return D_norm

print("normalizing for DEG : ")
DEG_norm = normalizing(dict(DEG),True,False)
print("normalizing for CN : ")
CN_norm = normalizing(CN,True,False)
print("normalizing for EC : ")
EC_log = normalizing(EC,True,True)
print("normalizing for PR : ")
PR_norm = normalizing(PR,True,False)
#print("normalizing for HT : ")
#HT_1_norm = normalizing(HT_1,True,False)
#HT_2_norm = normalizing(HT_2,True,False)
print("normalisations OK")
### Read Abstracts Dataset_full
df_asbtracts = pd.read_csv('Abstracts_Dataset.csv', dtype={'row':np.int64,'authorID': np.int64, 'abstracts': str,'hindex' : np.float64})
nc = 5
pca = PCA(n_components = nc)
print("abstracts_dataset loaded")

### After that use this instead :
# load the Stanford GloVe model
filename = 'glove.6B.300d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
print("glove300 loaded")

### (Pre)processing
df_abstracts = df_asbtracts

def sentence_preprocessing(s):
    sentence = re.sub('[^a-zA-Z]', ' ', s) #remove special characters and numbers
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence) #remove single characters (" a ")
    sentence = re.sub(r'\s+', ' ', sentence) #remove multiple spaces
    #word = word.lower() #get all lower cases
    liste = sentence.split(" ") #split a string into its list of words
    liste = [l.lower() for l in liste]
    return liste


def list_processing(liste, ponderated=False):
    U = np.zeros_like(model.get_vector('machine'))
    S = 0
    for word in liste :
        if word !='':
            try :
                wv = model.get_vector(word)
                U+=wv
                S+=1
            except :
                1+2
    if S==0:
        return model.get_vector('a'), 1
    if ponderated :
        wpop = model.most_similar(U,topn=1)[0][0]
        V = np.zeros_like(model.get_vector('machine'))
        I = 0
        for word in liste :
            if word !='':
                try :
                    coeff = model.similarity(word,wpop)
                    wv = model.get_vector(word)
                    V+=(1-coeff)*wv
                    I+=(1-coeff)
                except :
                    1+2
        return V,I
    return U,S

def from_dataframe_to_vector(df):
    dic = {}
    table = []
    j=0
    for i,row in df.iterrows():
        node = row['authorID']
        sentence_of_node = row['abstracts']
        #print(node,sentence_of_node)
        v,S = list_processing(sentence_preprocessing(sentence_of_node))
        #print(S)
        dic [node] = j
        table.append(v/S)
        #print(model.most_similar(v,topn=4))
        j+=1
    return dic, np.array(table)

print("ATAV started")
ATAV_dic, ATAV_array = from_dataframe_to_vector(df_abstracts)
print("ATAV ok")
#PCA-ing
ATAV_arraylite =  pca.fit_transform(ATAV_array)
print("ATAV_lite ok")
### Normalizing to [0,100] the gensim-features
def ATAV_like_normalizing(a):
    nb_authors, nb_features = a.shape
    a_norm = np.zeros_like(a)
    print(nb_authors,nb_features)
    for i in range(nb_features) :
        print(i)
        i_feat = a[:,i]
        #print(i_feat.shape)
        i_min, i_max = np.min(i_feat),np.max(i_feat)
        i_norm = lambda x : transfo_affine(i_min,i_max,0,100,x)
        a_norm[:,i] = i_norm(i_feat)
    return a_norm

ATAV_array_norm = ATAV_like_normalizing(ATAV_array)
print("ATAV array normalised ok ")
ATAV_arraylite_norm = ATAV_like_normalizing(ATAV_arraylite)
print("ATAV-lite array normalised ok ")

### MAIN : Computing on the train+test sets
print("train start")

graph_features = True
gensim_features = True
gensim_lite = False
nber_of_papers_feature = True


X_train = np.zeros((n_train,4*graph_features+1*nber_of_papers_feature+300*gensim_features+nc*gensim_lite))
y_train = np.zeros(n_train)
for i,row in df_train.iterrows():
    node = row['author']
    one_row_df = df_papers.loc[df_papers["author_id"]==node]
    list_papers_id = one_row_df.iat[0,1]
    assert(node==one_row_df.iat[0,0])
    # j, list_papers_id = (df_papers.loc[df_papers["author_id"]==node])["paper_ids"]
    if graph_features :
        X_train[i,0] = DEG_norm[node]
        X_train[i,1] = CN_norm[node]
        X_train[i,2] = EC_log[node]
        X_train[i,3] = PR_norm[node]
    if nber_of_papers_feature :
        X_train[i,4*graph_features] = transfo_affine(1,5,0,100,len(list_papers_id))
    if gensim_lite :
        Vect = ATAV_arraylite_norm[ATAV_dic[node]]
        for k in range(nc):
            X_train[i,4*graph_features+1*nber_of_papers_feature+k] = Vect[k]
    if gensim_features :
        Vect = ATAV_array_norm[ATAV_dic[node]]
        for k in range(300):
            X_train[i,4*graph_features+1*nber_of_papers_feature+nc*gensim_lite+k] = Vect[k]
    #X_train[i,5] = transfo_affine(0,189,0,100,df_deeptrain.loc[df_deeptrain["author"]==node].values[0,1])
    #X_train[i,5] = HT_1_norm[node]
    #X_train[i,6] = HT_2_norm[node]
    y_train[i] = row['hindex']
    #print(i)
print("train OK")
X_test = np.zeros((n_test, 4*graph_features+1*nber_of_papers_feature+nc*gensim_lite+300*gensim_features))
L_to_post_process = []
K = 0
for i,row in df_test.iterrows():
    node = row['author']
    one_row_df = df_papers.loc[df_papers["author_id"]==node]
    list_papers_id = one_row_df.iat[0,1]
    assert(node==one_row_df.iat[0,0])
    # j, list_papers_id = (df_papers.loc[df_papers["author_id"]==node])["paper_ids"]
    # if X_test[i,2] < 5 :
    #     L_to_post_process.append((i,X_test[i,2]))
    if graph_features :
        X_test[i,0] = DEG_norm[node]
        X_test[i,1] = CN_norm[node]
        X_test[i,2] = EC_log[node]
        X_test[i,3] = PR_norm[node]
    if nber_of_papers_feature :
        X_test[i,4*graph_features] = transfo_affine(1,5,0,100,len(list_papers_id))
    if gensim_lite :
        Vect = ATAV_arraylite_norm[ATAV_dic[node]]
        for k in range(nc):
            X_test[i,4*graph_features+1*nber_of_papers_feature+k] = Vect[k]
    if gensim_features :
        Vect = ATAV_array_norm[ATAV_dic[node]]
        for k in range(300):
            X_test[i,4*graph_features+1*nber_of_papers_feature+nc*gensim_lite+k] = Vect[k]
    # X_test[i,0] = DEG_norm[node]
    # X_test[i,1] = CN_norm[node]
    # X_test[i,2] = transfo_affine(1,5,0,100,len(list_papers_id))
    # X_test[i,3] = EC_log[node]
    # X_test[i,4] = PR_norm[node]
    #X_train[i,5] = transfo_affine(0,189,0,100,df_deeptest.loc[df_deeptest["author"]==node].values[0,1])
    #X_test[i,5] = HT_1_norm[node]
    #X_test[i,6] = HT_2_norm[node]
    K+=1
print(K)
print("test OK")
## train a regression model and make predictions
#reg = Lasso(alpha=0.1)
reg = GradientBoostingRegressor()
reg.fit(X_train, y_train)
print("fit OK")
y_pred = reg.predict(X_test)
print(y_pred.shape)



## Write the predictions into the file
df_test['hindex'] = pd.Series(np.round_(y_pred, decimals=3))

df_test.loc[:,["author","hindex"]].to_csv('submission.csv', index=False)


## * Do this once to precompute the word2vec model
glove_input_file = 'glove.6B.300d.txt'
word2vec_output_file = 'glove.6B.300d.txt.word2vec'
glove_wiki = KeyedVectors.load_word2vec_format(glove_input_file, binary=False, no_header=True)
KeyedVectors.save_word2vec_format(glove_wiki,word2vec_output_file)

### * Precompute Abstract Dataset_full
import json
from bisect import bisect_left
import csv
import numpy as np

purcent = 100

# functions

abstracts_path = "abstracts.txt"
train_path = "train.csv"
author_papers_path = "author_papers.txt"
test_path = "submission.csv"

def interpret(abstracts,N,purcent=100):

    def aux(i):
        if i% purcent*200 == 0 :
            print(str(int(100*100*i//(purcent*N)))+" %",end="\r")
        return json.loads(abstracts[i][int(np.log10(paper_ids[i]))+5:])

    itp_v = np.vectorize(aux)
    content = itp_v(np.arange(0,purcent*N//100,1))
    print("100 %")
    return content

def get_h_indexes():
    h_indexes = {}
    file_h_index = open(train_path,'r')
    reader = csv.reader(file_h_index)
    i = 0
    for row in reader :
        if i != 0 :
            h_indexes[row[0]]=row[1]
        if i == 0 :
            i = 1
    file_h_index = open(test_path,'r')
    reader = csv.reader(file_h_index)
    i = 0
    for row in reader :
        if i != 0 :
            h_indexes[row[0]]=row[1]
        if i == 0 :
            i = 1

    return h_indexes

def get_papers_authors():
    file_author_papers = open(author_papers_path,'r',encoding="utf-8")
    papers_authors = file_author_papers.readlines() # one line per author
    papers = {}
    for row in papers_authors :
        row = row.rstrip('\n')
        temp = row.split(":")
        if len(temp)!=2 :
            print("error")
        author = temp[0]
        papers_ = temp[1].split("-")
        papers[author]=papers_
    return papers


# getting everything in memory

file = open(abstracts_path,'r',encoding='utf-8')

abstracts = file.readlines() # list of all the "abstracts"
N = len(abstracts) # number of papers

paper_ids = np.array([int(abstracts[i].split("----")[0]) for i in range(N)])
content = interpret(abstracts,N,purcent)
h_indexes = get_h_indexes()
papers = get_papers_authors()


# reformatting

list_of_authors = list(h_indexes.keys())
list_of_authors.sort()

column1 = [ "" for _ in range(len(list_of_authors))]
column2 = np.zeros(len(list_of_authors))

for i in range(len(list_of_authors)):

    if i % 200 == 0 :
        print(str(round(i*100/len(list_of_authors),2))+" %", end = "\r")

    author_id = list_of_authors[i]

    concatenated_abstracts = ""
    author_paper_ids = papers[str(author_id)]

    for j in range(len(author_paper_ids)):

        paper_id = int(author_paper_ids[j])

        index = bisect_left(paper_ids, paper_id)

        if index == paper_ids.size :
            continue

        abstract_dico = content[index]["InvertedIndex"]

        numbers = [e for m in abstract_dico.values() for e in m]
        list_of_words = ["" for _ in range(max(numbers)+1)]

        for word in abstract_dico.keys():
            for rank in abstract_dico[word]:
                list_of_words[rank] = word+" "

        abstract = "".join(list_of_words)

        concatenated_abstracts = concatenated_abstracts + " "+abstract

    column1[i] = concatenated_abstracts
    column2[i] = h_indexes[str(author_id)]

print("100 %")



# saving everything on harddrive

results_path = "Abstracts_Dataset.csv"

def serialize(column0,column1,column2,file_path):
    agg = [[column0[i],column1[i],column2[i]] for i in range(len(column1))]
    df = pd.DataFrame(agg,columns=["authorID","abstracts","hindex"])
    df.loc[:,["authorID","abstracts","hindex"]].to_csv(file_path)

serialize(list_of_authors,column1,column2,results_path)


### *** Little study of the way the train set's h-index behaves for authors with <5 papers
u = []
M = np.zeros((5,5))
print(M)
S = 0
for i in  range(len(X_train)) :
    x = X_train[i]
    if x[2]<5:
        M[int(x[2]),int(y_train[i])]+=1
        S+=1
print(M/S * 100)
print(S)

## *** Abandonned features :

CC = nx.communicability_exp(G)
print("CC ok")

HT = nx.hits(G)
HT_1, HT_2 = HT
print("HT ok")


LC = nx.load_centrality(G)
print("LC ok")


# read trained data
df_deeptrain = pd.read_csv('for_feature.csv', dtype={'author': np.int64, 'hindex': np.float32})
df_deeptest = pd.read_csv('submission50.csv', dtype={'author': np.int64, 'hindex': np.float32})

## *** Eventual Post-processing on the authors with <5 papers referenced
for i,N in L_to_post_process :
    #mean prediction using M (next block)
    N = int(N)
    m = (1*M[N,1]+2*M[N,2]+3*M[N,3]+4*M[N,4])/(M[N,1]+M[N,2]+M[N,3]+M[N,4])
    if N < y_pred[i]:
        print(N, m,  y_pred[i])
    y_pred[i] = m
    #naive prediction : 1 for N=1 and 2 for the rest
    #y_pred[i] = 1*(N==1)+2*(N!=1)
    #print (y_pred[i],N)

print("post-processing OK")

### *** Little test on Gensim : Queen = (King - Man)
# calculate: (king - man) + woman = queen
result = model.most_similar(positive=['death'], negative=['man'], topn=1)
print(result)


