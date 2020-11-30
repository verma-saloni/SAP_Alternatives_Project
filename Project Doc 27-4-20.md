# Preliminary Documentation for Decision Support System

This is a preliminary documentation for the Decision Support System, outlining the progress and current activities of the project. We have collected the basic data to start modelling the SAP products and their alternatives. The early models have low accuracy but show promising results. The pre-processed data through Information retrieval methods was used in the basic machine learning models resulting in different observations. A mockup of the User's first interaction page is also ready. Further work is being done on creating an apt model, the user interface and how the results will be modelled. Challenges with the data and its adaptability to the model means that data collection and updating is a constant process as no each model is fit for the type and range of data we gather. 



## Data Gathering

The data process started with understanding the range of products that SAP offers. Under different umbrella terms, we started creating an exhaustive list of SAP products and each with its market counter-parts. Their features like ranking, information, etc was gathered from different sources for the initial data bank that we worked with. 

Link: https://docs.google.com/spreadsheets/d/1yfEgHNgCRZ4pOXio3fDqTfSVLOCpJTOihkW3_Zp1Pbg/edit#gid=697868244

- Other Docs:

https://docs.google.com/spreadsheets/d/13ITi2y5f02IgQG7lTBJdEwLbgsRBDuZfh4SrS4JUvAI/edit#gid=0

- After initial preprocessing and Bag of Words, stemma and lematisation, we gathered more data for our models since the data was not ideal for the models being used. 
- Updated Docs: https://drive.google.com/file/d/1WOxcvAPCLCIt_1Wu-55ZFT3u_hGJ0zvN/view and https://drive.google.com/file/d/12xP8L-KLVZO_0zt4lFe09chuKlPVWZpe/view which include extensive information about SAP products from their official documentation and more alternatives that we were able to find. 
- Another set of alternatives and categories we gathered was: https://docs.google.com/spreadsheets/d/12GcIHlRyKYWG0hPyDiE6Qjv8K-dmbI3hfhSpN804R7M/edit?usp=sharing

- Currently the data that we are working with has 500 data rows, extensive product information which was preprocessed and is being used in our models now. 

Sources: https://www.predictiveanalyticstoday.com/sap-business-bydesign/
https://www.capterra.com/p/92075/SAP-BusinessObjects/#comparisons 
https://d.dam.sap.com/m/mJUnN/62449_SB_40408_enUS.pdf
https://www.trustradius.com/products/oracle-financials-cloud/reviews

### Pre-processing the Raw Data

Different methods were employed multiple times to see how the data should be handled before it can be input to a put. Bag of words, Lemmatisation, Stemma was applied initially. Checks for spellings were also applied in the initial code. But later, the spell check was removed to see how the data performed since it had technical words that don't exist in normal vocabulary that were being ignored. Current preprocessing snippets: 

```python
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    text = expand_contrations_driver(text)
    text = remove_noise(text)
    tokens = tokenize_word_text(text)
    tokens = remove_accent(tokens)
    tokens = remove_stopwords(tokens)
    # tokens = [correct_word(token) for token in tokens]
    
    # Applying Normalization - Both are worse
    # Stemming
#     words_stem = [stemmer.stem(w) for w in tokens]
#     for i in range(len(tokens)):
#         if (tokens[i] != words_stem[i]):
#             print(tokens[i] + " - " + words_stem[i])
    
    # 2- Lemmatization
    # words_lem = [lemmatizer.lemmatize(w) for w in tokens]
    # Checking for its worth
#     for i in range(len(tokens)):
#         if (tokens[i] != words_lem[i]):
#             print(tokens[i] + " - " + words_lem[i])
    return tokens
```

Source: https://github.com/krazygaurav/ovgu-dss/blob/master/pre-process.ipynb



### Models Implemented and Preliminary Results

When we had an initial data set ready and the preprocessing was completed, we started working on models to see how the data performs and to test our initial hypothesis. We have tried to implement through Hierarchical Clustering, Wards Methods, KNN, etc, with varying results. A snippet of the Hierarchical clustering is as follows:

```python
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt


linkage_matrix = ward(dist)

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=list(dataset.name));

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
```

Source: https://github.com/krazygaurav/ovgu-dss/blob/master/hierarchical-clustering.ipynb

Image: ![ward_clusters](C:\Users\Saloni PC\Desktop\Project DSS\ward_clusters.png)

We are implementing more models and we will choose the best one based on evaluation measures and results. 

### User Interface

A very basic UI was created for taking input from the user. Working on further pages and how we will visualise the code later. 

![UI-1](C:\Users\Saloni PC\Desktop\Project DSS\UI-1.jpg)

------------------

### Understanding and Interpreting the Model

We will work on some evaluation measures based on how the data is modelled and if the results are accurate, precise. We will also move on to the interpretability of the project in the next few weeks so it is easy to use and understand the logic behind our models and how aptly the alternatives are suggested. 



mention literarure : word to vec

tsne 

and fir iske alawa humne kya choose kiya h.

