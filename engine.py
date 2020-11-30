# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import re
import string
import numpy as np
import nltk
import unidecode
from collections import Counter
from nltk import tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import spatial
import gensim 
from gensim.models import Word2Vec
from nltk.stem import SnowballStemmer
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity

# From Utilities
from utilities.pre_process_helpers import * 

import warnings
warnings.filterwarnings('ignore')

# +
stemmer = SnowballStemmer("english")

def preprocess(text, use_stemming=True):
    
    # TODO: Perform Name entity analysis first
    
    # Converting to Lower
    text = str(text)
    text = text.lower()
    text = expand_contrations_driver(text)
    text = remove_noise(text)
    tokens = tokenize_word_text(text)
    tokens = remove_accent(tokens)
    tokens = remove_stopwords(tokens)
    
    # Stemming is used for Decriptions and not for the Name features
    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)
# +
def get_tfidf_model(documents, threshold=0.1):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_df=0.4, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    # Tuning the tfidf model as per zipf law. 
    tf_df = pd.DataFrame(tfidf_matrix.toarray().T, index=tfidf_vectorizer.get_feature_names())
    tf_df['tfidf'] = tf_df.max(axis=1)
    temp = tf_df.drop(index=tf_df[tf_df.tfidf < threshold].index)
    new_vocab = list(temp.index)
    
    # New tf model
    tfidf_vectorizer_n = TfidfVectorizer(use_idf=True, vocabulary=new_vocab)
    tfidf_matrix_n = tfidf_vectorizer_n.fit_transform(documents)
    
    return [tfidf_vectorizer_n, tfidf_matrix_n]


WORD_VECTOR_SIZE = 50
MIN_COUNT = 1
WINDOW = 7
def get_word2vec_model(documents):
    documents = [document.split() for document in documents]
    word2vec_model = gensim.models.Word2Vec(documents, min_count = MIN_COUNT, \
                                             size = WORD_VECTOR_SIZE, window = WINDOW, iter=3)
    document_matrix = []
    for document in documents:
        document_matrix.append(get_doc2vec_vector(word2vec_model, " ".join(document)))
    return word2vec_model, np.array(document_matrix)
    
def get_doc2vec_vector(model, doc):
    doc_vec = [0]*WORD_VECTOR_SIZE
    num_skip = 0
    for word in doc:
        try:
            doc_vec = np.add(doc_vec, model[word])
        except (KeyError):
            # If word does not exist in dictionary then give 0.01 weight to it for all dimension
            num_skip += 1
    return np.asarray(doc_vec)/(len(doc)-num_skip)


def get_name_features(names):
    name_vectorizer = CountVectorizer()
    transformed = name_vectorizer.fit_transform(names)
    name_features = transformed.toarray()
    return name_vectorizer, name_features


# +
def get_total_score(tf_score, embed_score):
    return tf_score+embed_score

def cosine_score(doc, query):
    return 1 - spatial.distance.cosine(doc, query)
#     return np.sum(np.multiply(doc, query))

def get_cosine_nearest_documents(embed_q_vec, tf_q_vec, tfidf_features, embed_features, TOTAL_DOCS):
    cos_scores = []
    for doc_index in range(TOTAL_DOCS):
        score1 = cosine_score(embed_features[doc_index], embed_q_vec)
        score2 = cosine_score(tfidf_features[doc_index].reshape(1, tfidf_features.shape[1]), tf_q_vec)
        cos_scores.append([score1, score2])
    return pd.DataFrame(cos_scores, columns=['string_match', 'embedd_match'])

def get_query_related_tool(name_features, query_vector):
    similar_score = -1
    similar_index = -1

    for index in range(len(name_features)):
        sim_score = cosine_similarity(name_features[index].reshape((1, -1)), query_vector)[0][0]
        if similar_score <= sim_score:
            similar_index = index
            similar_score = sim_score
            
    return [similar_index, similar_score]

def query_synthesizer(query, tfidf_vectorizer, embed_vectorizer):
    # Applying same pre-processing on query
    query = preprocess(query)
    # print("Query: "+str(query.split()))
    
    # Creating feature vector of query. Feature vector creation should be same as of Documents
    tfidf_query_vector = tfidf_vectorizer.transform([query])
    embedd_query_vector = get_doc2vec_vector(embed_vectorizer, query)
    
    return [embedd_query_vector, tfidf_query_vector.toarray()]


# -

# # Making data better

def category_filter(category):
    category = category.lower()
    category = category.split(",")[0]
    category = category.strip()
    category = str(category)
    
    if 'crm' in category or 'market' in category:
        category = 'crm software'
    elif 'erp' in category:
        category = 'erp software'
    elif 'business' in category:
        category = 'business intelligence software'
    elif 'dashboard' in category:
        category = 'dashboard'
    elif ('project') in category:
        category = 'project management software'
    elif ('product') in category or ('inventory') in category or ('plm') in category:  
        category = 'inventory or product management software'
    elif ('analys') in category or ('analytic') in category or ('visual') in category:
        category = 'predictive analytics software'
    elif ('supply') in category or ('scm') in category:
        category = 'supply chain management software'
    elif ('procurement') in category:
        category = 'procurement management software'
    elif ('task') in category or ('tms soft') in category:
        category = 'task management software'
        
    return category


# # Tuning Tf-idf model

# + active=""
# # Removing terms which appears more than 50% of the documents
# tfidf_vectorizer_x = TfidfVectorizer(use_idf=True, max_df=0.5)
# tfidf_matrix_x = tfidf_vectorizer_x.fit_transform(df.desc_preprocess)
# feature_2 = tfidf_vectorizer_x.get_feature_names()
#
# print("Shape: "+ str(tfidf_matrix_x.shape))
# # Tuning the tfidf model as per zipf law. 
# tf_df = pd.DataFrame(tfidf_matrix_x.toarray().T, index=tfidf_vectorizer_x.get_feature_names())
# tf_df['tfidf'] = tf_df.max(axis=1)
# # min_threshold = 0.04
# # print(tf_df[tf_df.tfidf < min_threshold])
# max_threshold = 0.75
# tf_df[tf_df.tfidf > max_threshold]
#
#
# # threshold = 0.00
# # for i in range(20):
# #     print("Threshold: %.3f -> %d"%(threshold, tf_df[tf_df.tfidf < threshold].shape[0]))
# #     threshold += 0.01
# # temp = tf_df.drop(index=tf_df[tf_df.tfidf < 0.05].index)
# # new_vocab = list(temp.index)

# +
import pickle

def dump_vectors(tfidf_matrix, wordvec_matrix):
    '''
    Exports the calculated matrices in pickle dataset for further use. 
    It saves a lot of learning time. Just load then and calculate proximity with query with the vectors
    To load learnt features use below code:
    data = pickle.load(open("data/features_v1.pkl", "rb"))
    
    @input
    tfidf_matrix -> Learnt Tf-Idf matrix based on dataset given
    word2vec_matrix -> Learnt Word2Vec matrix based on dataset given
    '''
    
    PIK_FILE = "data/features_v1.pkl"
    data = {}
    data['tfidf'] = tfidf_matrix
    data['word2vec'] = wordvec_matrix
    
    pickle.dump(data, open(PIK_FILE, 'wb+'))


# -

# # Execution

# Script for Reading CSV data and dropping duplicate name tools 
temp_df = pd.read_csv("data/Alter.csv")
temp_df = temp_df.drop_duplicates(subset=['name'], keep='first')
temp_df.to_csv("data/new_data_v1.csv")

# +
df = pd.read_csv("data/new_data_v1.csv", index_col=0)
df.reset_index(inplace=True)
df.drop(['index'], axis=1, inplace=True)
df_copy = df.copy()
 
# Performing pre-process. Output -> tokens separated by space
df.description = df.description + " " + df.license
df.desc_preprocess = df.description.apply(preprocess)

# Not using new categories
# df['cat_new'] = df.category.apply(category_filter)

# Getting feature vector
## Getting Tf-Idf feature vector
tfidf_vectorizer, tfidf_matrix = get_tfidf_model(df.desc_preprocess)
tfidf_features = tfidf_matrix.toarray()

## Getting word-2-vec feature vector
embed_vectorizer, embed_features = get_word2vec_model(df.desc_preprocess)

TOTAL_DOCS = 0
if(embed_features.shape[0] == tfidf_features.shape[0]):
    TOTAL_DOCS = embed_features.shape[0]
else:
    print("Tf-Idf and Embedding feature shapes are different")

# Getting Name features
df['name_preprocess'] = df.apply(lambda row: preprocess(row['name'], False), axis=1)
name_vectorizer, name_features = get_name_features(df.name_preprocess)
# -

# # Handling Query

# +
query_string = "SAP BW"

# Check if Query has exact match with database
name_query_vector = name_vectorizer.transform([preprocess(query_string, False)])
q_sim_index, q_sim_score = get_query_related_tool(name_features, name_query_vector)
# If cosine similarity score is high then change query to that document
if q_sim_score > 0.8:
    query = df.loc[q_sim_index]['description']
else:
    query = query_string
    
print("Query match with tool: '{}', with score: {}".format(df.loc[q_sim_index]['name'], q_sim_score))
    
embed_q_vec, tf_q_vec = query_synthesizer(query, tfidf_vectorizer, embed_vectorizer)

# Finding Cosine similarity between Documents and Query
cos_score = get_cosine_nearest_documents(embed_q_vec, tf_q_vec, tfidf_features, embed_features, TOTAL_DOCS)
result = df.join(cos_score)
result = result.sort_values(by=['embedd_match', 'string_match'], ascending=False)

# Printing Similar documents. Top 20
total_results = 20
# Filter out SAP products
result = result[result.name.str.contains("SAP") == False]
result = result[result.name != df.loc[q_sim_index]['name']]
result['total_score'] = result['embedd_match'] + result['string_match']
result[['name', 'embedd_match', 'string_match', 'total_score', 'license']][:total_results]\
    .sort_values(by=['total_score'], ascending=[False])
# -



# ## Future work

# ### (Not Using) KD Tree
# ##### Substitute of knn

from sklearn.neighbors import KDTree

a = tfidf_features.tolist()
b = embed_features.tolist()

kdt_tf = KDTree(a, leaf_size=10)
kdt_embed = KDTree(b, leaf_size=20)

# +
# For Tf-Idf 
dist, idx = kdt_tf.query(tf_q_vec, k=10)

# For Word2Vec
dist_1, idx_1 = kdt_embed.query(embed_q_vec.reshape([1, embed_q_vec.shape[0]]), k=10)
# -



# # Abbreviation extraction

# +
def get_abbreviated_terms(text):
    # matches = re.findall(r'(?:(?<=^)|(?<=[^.]))\s+([A-Z]{2,})', text)
    matches = re.findall(r'\b[A-Z]{2,}\b', text)
    return list(set(matches))

# df_copy['abbr'] = df_copy.description.apply(get_abbreviated_terms)
