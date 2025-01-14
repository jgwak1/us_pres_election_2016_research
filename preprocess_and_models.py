import pandas as pd
import numpy as np
import re
from nltk import *


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


''' Basic Preprocessing '''

#lower-case
#punctuation removal
#stop-word removal
#Stemming

def stop_word_remover(Input, stopword_ls):

    x = ' '.join([word for word in Input.split() if ( word not in stopword_ls )]) 
    return x 

def stem_by_row(Input):
    stemmer = PorterStemmer()
    x = ' '.join([stemmer.stem(word) for word in Input.split()])
    return x 

def custom_remover(Input):

    # Remove all numbers except for years
    x = ' '.join([word for word in Input.split() if ((word<='2100') & (word>='1900') & (len(word)==4)) or word.isalpha() ])   
 
    # Remove words combined with both alphabets and numbers 
    exclude_ls = []
    for word in x.split(): 
        if any(chr.isalpha() for chr in word) and any(chr.isdigit() for chr in word):
            exclude_ls.append(word)
    x = ' '.join([word for word in x.split() if ( word not in exclude_ls )]) 

    return x 
    
def transcript_preprocessor(data, trasncript_colname = 'transcript'):

    from nltk.corpus import stopwords    
    import string 
    punctuations = re.escape(string.punctuation)
    punctuations+='—' ; punctuations+='–'; punctuations+='•'


    stop = stopwords.words('english')
    stop.append('\n')
    stop.append('n')


    data[trasncript_colname] = data.apply(lambda row: row[trasncript_colname].lower(), axis=1)
    data[trasncript_colname] = data.apply(lambda row: re.sub(r'['+punctuations+']', '', row[trasncript_colname]), axis= 1)
    data[trasncript_colname] = data.apply(lambda row: stop_word_remover(row[trasncript_colname],stop), axis= 1)
    data[trasncript_colname] = data.apply(lambda row: custom_remover(row[trasncript_colname]), axis= 1)
    data[trasncript_colname] = data.apply(lambda row: stem_by_row(row[trasncript_colname]), axis= 1)

    return data


''' Modeling '''

# LDA /w

def lda_DocWordScore_generator(transcript_data, vectorizer_object, lda_ojbect = LatentDirichletAllocation(n_components=10, random_state= 1) ):

    vectorizer_fit = vectorizer_object.fit_transform(transcript_data) #can extract feature after this
    feature = vectorizer_object.get_feature_names() # words

    ldamod = lda_ojbect.fit(vectorizer_fit)

    score_document_topic = pd.DataFrame(lda_ojbect.fit_transform(vectorizer_fit)) # Row: document, Column: topic    
    score_topic_words = pd.DataFrame(ldamod.components_) # Row: topic, Column: words(feature)
    score_topic_words.columns = feature  
    score_topic_words = score_topic_words.div(score_topic_words.sum(axis=1), axis=0) # normalized socre_topic_words
    
    score_document_words = score_document_topic.dot(score_topic_words)

    return score_document_words


def DocWordScorebyVar_ColGenerator(data, values_of_variable, variable_colname = 'CANDIDATES', transcript_colname = 'TRANSCRIPT', vectorizer_obj = TfidfVectorizer( max_df=0.5, min_df=3 )  ):

    '''
    CountVectorizer(max_df=0.5, min_df=3 )     # "max_df==1.0(default)" means ignore terms that appear in more than 100% of the documents (default setting doesn't ignore any terms)
                                                     # 'min_df==1 (default)' means ignore terms that appear in less than 1 document. (default setting does not ignore any terms)
                                                     #   When these parameters are changed, it can lead to cases where too much pruning occurs.
            
    TfidfVectorizer( max_df=0.5, min_df=3 )  
    '''

    data = data[ (data[transcript_colname]!="longer public video") & (data[transcript_colname]!="noneyoutub") ]
    
    result_df = pd.DataFrame()
    
    for value in values_of_variable:
        
        value_corresponding_data = data[data[variable_colname]==value]
    
        score_document_words = lda_DocWordScore_generator( transcript_data = value_corresponding_data[transcript_colname], vectorizer_object = vectorizer_obj )

        value_corresponding_data = value_corresponding_data.reset_index()
        
        value_data_w_score_doc_words = pd.concat([value_corresponding_data, score_document_words], axis = 1 )
        
        result_df = pd.concat([result_df, value_data_w_score_doc_words], axis = 0, join = "outer", ignore_index=False, sort=False)
        
        
    return result_df  





def TopXWords_fromEntire(data, X, vocab_start_col_idx = False):
    
    if (vocab_start_col_idx == False):
        raise ValueError("Error: Carefully check the column index where which the vocabularies start, and specify it.")


    M = list(zip(list(data.iloc[:,vocab_start_col_idx:].columns), list(data.iloc[:,vocab_start_col_idx:].sum())))
    dt = pd.DataFrame(sorted(M, key = lambda x: x[1], reverse=True))

    return dt[0:X]





def TopXTopics_TopYDocuments_Generator(dt, X, Y, trasncript_colname = "TRANSCRIPT" ):
    
    data = dt.copy()
    
    # First apply LDA
    cv = TfidfVectorizer(max_df=0.5, min_df= 3 ) 
    
    cv_fit_transformed = cv.fit_transform(data[trasncript_colname])
    feature = cv.get_feature_names()
    lda = LatentDirichletAllocation(n_components = X, random_state =1 )
    ldamod = lda.fit(cv_fit_transformed)
    score_document_topic = pd.DataFrame(ldamod.transform(cv_fit_transformed)) # Row: document, Column: topic
    
    result_df = pd.DataFrame([])
    
    
    # For Top X Topics, get Top Y Documents , and make it into dataframe
    for i in range(X):
    
        x= pd.DataFrame(score_document_topic[i]).sort_values(by=i, ascending=False)[0:Y]
        x = pd.DataFrame(data.loc[data.index.intersection(x.index),:][trasncript_colname])
        x.columns = ["topic_"+str(i)]
        x = x.reset_index() #.drop(columns=["index"])
        
        result_df = pd.concat([result_df, x], axis = 1) 
        
    return result_df




def TopXTopics_TopYWords_Generator(dt, X, Y, trasncript_colname = "TRANSCRIPT" ):
    
    data = dt.copy()
    
    # First apply LDA
    cv = TfidfVectorizer(max_df=0.5, min_df= 3 ) 
    
    cv_fit_transformed = cv.fit_transform(data[trasncript_colname])
    feature = cv.get_feature_names()
    lda = LatentDirichletAllocation(n_components = X, random_state =1 )
    ldamod = lda.fit(cv_fit_transformed)
    score_topic_words = pd.DataFrame(ldamod.components_)  # Row: topic, Column: words(feature)
    score_topic_words.columns = feature 
    
    result_df = pd.DataFrame([])

    # For Top X Topics, get Top Y Words , and make it into dataframe
    for i in range(X):
    
        x= pd.DataFrame(score_topic_words.iloc[i,:]).sort_values(by=[i], ascending=False)[0:Y]
        x= pd.DataFrame(x.T.columns)
        x.columns = ["topic_"+str(i)]
        
        result_df = pd.concat([result_df, x], axis = 1) 
        
    return result_df
    



