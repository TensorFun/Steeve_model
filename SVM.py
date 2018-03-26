
# coding: utf-8

# In[2]:

import numpy as np
from sklearn import feature_extraction, svm, metrics
from pyfasttext import FastText

class SVM:
    def __init__(self, TFIDF): # TODO: plz give me the model or using global singleton
        self.model = FastText('wiki.en.bin')
        self.TFIDF = TFIDF
        self.tfidf_dict, self.words = TFIDF.get_tfidf()
    
    def set_tfidf(self, TFIDF):
        self.TFIDF = TFIDF
        self.tfidf_dict, self.words = TFIDF.get_tfidf()
    
    def to_feature(self, X):
        new_X = [vec(self.TFIDF.predict_field(post), post) for post in X]
        
    def vec(self, field_index, post):
        v = np.zeros(300)
        post = set(post) # make unique
        for pl in post:
            if pl != '' and pl in self.words:
                v += model.get_numpy_vector(pl) * tfidf_score[field_index][pl]
        return v
    
    def train(self, X, y):
        # 建立 SVC 模型
        self.svc = svm.SVC()
        svc_fit = self.svc.fit(to_feature(X), y)

    def predict(self, post):
        return self.svc.predict(to_feature([post]))
        


# In[ ]:




# In[ ]:



