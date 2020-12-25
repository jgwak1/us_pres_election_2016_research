import pandas as pd
import numpy as np
import re
import nltk

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
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    x = ' '.join([stemmer.stem(word) for word in Input.split()])
    return x 

def transcript_preprocessor(data):
    from nltk.corpus import stopwords    
    import string 
    punctuations = re.escape(string.punctuation)

    stop = stopwords.words('english')
    stop.append('im')
    
    data['transcript'] = data.apply(lambda row: row['transcript'].lower(), axis=1)
    data['transcript'] = data.apply(lambda row: re.sub(r'['+punctuations+']', '', row['transcript']), axis= 1)
    data['transcript'] = data.apply(lambda row:  stop_word_remover(row['transcript'],stop), axis= 1)
    data['transcript'] = data.apply(lambda row: stem_by_row(row['transcript']), axis= 1)

    return data


''' Modeling '''

# LDA /w


def topic_modeling_by_candidate(data, candidate_list):

    data = data[ (data.TRANSCRIPT!="longer public video") & (data.TRANSCRIPT!="noneyoutub") ]
    
    
    result_df = pd.DataFrame()
    
    #cv = CountVectorizer(max_df=0.5, min_df=3, stop_words='english') # max_df and min_f are hyperparatmeters dealwith this part 
    
    #cv = CountVectorizer(max_df=0.5, min_df=3 )     # "max_df==1.0(default)" means ignore terms that appear in more than 100% of the documents (default setting doesn't ignore any terms)
                                                     # 'min_df==1 (default)' means ignore terms that appear in less than 1 document. (default setting does not ignore any terms)
                                                     #   When these parameters are changed, it can lead to cases where too much pruning occurs.
            
    cv = TfidfVectorizer( max_df=0.5, min_df=3 )  
    
    
    for candidate in candidate_list:
        
        cand_data = data[data.CANDIDATES==candidate]
    
        cv_fit = cv.fit_transform( cand_data["TRANSCRIPT"] )  #can extracture feature from cv after this
        feature = cv.get_feature_names() # words 
        
        lda = LatentDirichletAllocation(n_components=10, random_state= 1)
        ldamod = lda.fit(cv_fit)

        prob_document_topic = pd.DataFrame(lda.fit_transform(cv_fit)) # Row: document, Column: topic
          
        score_topic_words = pd.DataFrame(ldamod.components_)  # Row: topic, Column: words(feature)
        score_topic_words.columns = feature 
        prob_topic_words = score_topic_words.div(score_topic_words.sum(axis=1), axis=0) # normalized
        
        prob_document_words = prob_document_topic.dot(prob_topic_words)

        cand_data = cand_data.reset_index()
        
        cand_data_prob_document_words = pd.concat([cand_data, prob_document_words], axis = 1 )
        
        result_df = pd.concat([result_df, cand_data_prob_document_words], axis = 0, join = "outer", ignore_index=False, sort=False)
        
        
    return result_df  