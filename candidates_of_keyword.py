
# coding: utf-8

# # Candidates of keyword

# ## Load model

# In[1]:


import regex as re
import spacy
import json
import os
# from gensim.models.word2vec import Word2Vec

from rake import Rake
from pl_module import get_pl_keywords
from nltk.corpus import stopwords

nlp = spacy.load('en',disable=['ner', 'tagger'])
PATH = '../Steeve_data/no_filter_Dice/raw_data_post/'
eng_stop = stopwords.words('english')


# In[2]:


def Data_loading(PATH):
    
    # load json files and get title name of each json file
    file_data=[]
    feild_names=[]
    key = []
    for path, dirs, files in os.walk(PATH):
        for i,file in enumerate(files):
            file_data.append( json.load(open(PATH+file)))
            for k in file_data[i].keys():
                key.append(k)
            
    return file_data,key


# ## Get key candidates 
# #### Get NC(spaCy), NC(rake) and programming languages
# 

# In[3]:


def getN_Crunk(doc):
    
    candidates = []
    doc = nlp(doc)
    chunk = list(doc.noun_chunks)
    candidates = list(map(lambda el: str(el).lower(), chunk))
    
#     for token in chunk:
#         candidates.append(token.lower())
        
    return candidates


def getoffStopWord(n_chunks):
    
    for n, n_str in enumerate(n_chunks):

        n_s = n_str.split(" ")
        for word in n_s:
            if word in eng_stop:
                n_s.remove(word)

        n_chunks[n] = " ".join(n_s)
    
    return n_chunks
    
def getCleanWord(n_chunks):
#     exclude = "(\w+)( \-|\- | \- )(\w+)"
    exclude="[\w ]|[ \w][\w\-\w]"
#     exclude = "[(\w+)-(\w+)]"
    
    for n, n_str in enumerate(n_chunks):
        n_str = n_str.replace("\n", " ")
        matches = re.findall(exclude,n_str)
        cl_words = "".join(matches)
        n_chunks[n] = cl_words
                
    return n_chunks

def get_NC(n_chunks):
    
    n_chunks = getCleanWord(n_chunks)
    n_chunks = getoffStopWord(n_chunks)
    
    return n_chunks

def get_Rake_NC(n_chunks):
    
    rake_object = Rake("SmartStoplist.txt")
    keywords = rake_object.run(n_chunks)
    rake_skills = [x[0] for x in keywords]
    
    return rake_skills
    


# ## Load data

# In[4]:


if __name__ == "__main__":
    
    ori_data, field_names = Data_loading(PATH)
#     json_strucure=["jobTitle","jobEmployer","jobLocation","jobPostTime","skills",\
#                "employmentType","baseSalary","jobDescription","url"]
     
    print("load data")
    sentences = ''
    rake_object = Rake("SmartStoplist.txt")
    
    for i, f in enumerate(ori_data):  
        print(field_names[i])
        data = {} 
        data[field_names[i]] = []
    
        for num, job_num in enumerate(ori_data[i][field_names[i]]):
            if num%500 ==0:
                print(num)
            ### testing gensim
#                 sentences += job_num["jobDescription"]
            ### testing gensim
            
            nc_des = getN_Crunk(job_num["jobDescription"])
            nc_ski = getN_Crunk(job_num["skills"])

            can_description = get_NC(nc_des)
            can_skills = get_NC(nc_ski)
            
            rake_des = get_Rake_NC(job_num["jobDescription"])
            rake_ski = get_Rake_NC(job_num["skills"])        
            
            pl_des = get_pl_keywords(job_num["jobDescription"])
            pl_ski = get_pl_keywords(job_num["skills"])
            
            data[field_names[i]].append({
                "jobTitle": job_num["jobTitle"],
                "NC": can_description+ can_skills+rake_des+rake_ski,
                "PL": pl_des+ pl_ski, 
                "url": job_num["url"]})
        
        with open('../Steeve_data/no_filter_Dice/can/Keywords'+field_names[i]+'.txt', 'w') as f:
            json.dump(data, f)


    print("DONE")


# ## Testing

# In[ ]:


asd = json.load(open('../Steeve_data/candidates_keyword/KeywordsBackend.txt'))

text = asd["Backend"]

testing_data = ori_data[0][field_names[0]][1]["jobDescription"]
# print(testing_data)

# print(text[123]["NC"])
# for i in text[0]["NC"]:
#     print(i)
# print(ori_data[26]["skills"])

