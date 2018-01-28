
# coding: utf-8

# # Candidates of keyword

# ## Load model

# In[1]:


import regex as re
from spacy.tokenizer import Tokenizer
import spacy
import json
# import flashtext
from pl_module import get_pl_keywords
from nltk.corpus import stopwords
import os


nlp = spacy.load('en')
PATH = '../Steeve_data/raw_data_post/'


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


# ## Get NOUN TRUNK key candidates 
# 

# In[3]:


def getN_Crunk(doc):
    
    candidates = []
    doc = nlp(doc)
    chunk = list(doc.noun_chunks)
    chunk = map(str,chunk)
    
    for token in chunk:
        candidates.append(token.lower())
        
    return candidates


def getoffStopWord(n_chunks):
    
    for n, n_str in enumerate(n_chunks):

        n_s = n_str.split(" ")
        for word in n_s:
            if word in stopwords.words('english'):
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


# ## Load data

# In[4]:


if __name__ == "__main__":
    
    ori_data, field_names = Data_loading(PATH)
    json_strucure=["jobTitle","jobEmployer","jobLocation","jobPostTime","skills",               "employmentType","baseSalary","jobDescription","url"]
     
    print("load data")
    
    for i, f in enumerate(ori_data):  
        data = {} 
        data[field_names[i]] = []
    
        for job_num in ori_data[i][field_names[i]][:30]:
            
            nc_des = getN_Crunk(job_num["jobDescription"])
            nc_ski = getN_Crunk(job_num["skills"])
        
            can_description = get_NC(nc_des)
            can_skills = get_NC(nc_ski)
        
            pl_des = get_pl_keywords(job_num["jobDescription"])
            pl_ski = get_pl_keywords(job_num["skills"])
        
            data[field_names[i]].append({
                "jobTitle": job_num["jobTitle"],
                "NC": can_description+ can_skills,
                "PL": pl_des+ pl_ski, 
                "url": job_num["url"]})
        
        with open('../Steeve_data/candidates_keyword/Keywords'+field_names[i]+'.txt', 'w') as f:
            json.dump(data, f)
    


# In[5]:



for i, f in enumerate(ori_data):  
    
    data = {} 
    data[field_names[i]] = []
    
    for job_num in ori_data[i][field_names[i]][:30]:
        
        nc_des = getN_Crunk(job_num["jobDescription"])
        nc_ski = getN_Crunk(job_num["skills"])
        
        
        can_description = get_NC(nc_des)
        can_skills = get_NC(nc_ski)
        
        pl_des = get_pl_keywords(job_num["jobDescription"])
        pl_ski = get_pl_keywords(job_num["skills"])
        
        data[field_names[i]].append({
            "jobTitle": job_num["jobTitle"],
            "NC": can_description+ can_skills,
            "PL": pl_des+ pl_ski, 
            "url": job_num["url"]})
        
    with open('../Steeve_data/candidates_keyword/Keywords'+field_names[i]+'.txt', 'w') as f:
        json.dump(data, f)
    


# ## combine NT and PL

# ## Testing

# In[6]:


asd = json.load(open('../Steeve_data/candidates_keyword/Keywords_Back_End.txt'))

text = asd["Back_End"]

# print(text[123]["NC"])
for i in text[0]["NC"]:
    print(i)
# print(ori_data[26]["skills"])

