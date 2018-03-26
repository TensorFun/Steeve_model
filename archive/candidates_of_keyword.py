
# coding: utf-8

# # Candidates of keyword

# ## Load model

# In[10]:


import json
import os

from pl_module import get_pl_keywords

PATH = '../testing_data/raw_data_post/'


# In[6]:


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


# ## Load data

# In[23]:


if __name__ == "__main__":
    
    ori_data, field_names = Data_loading(PATH)
     
    print("load data")
    total_data = [] 
    
    for i, f in enumerate(ori_data):  
        print(field_names[i])
        data = []
#         data[field_names[i]] = []
    
        for num, job_num in enumerate(ori_data[i][field_names[i]]):
            if num%500 ==0:
                print(num)

            pl_des = get_pl_keywords(job_num["descirbe"])
            pl_ski = []
            data.append(pl_des+pl_ski)

#             data.append([
#                 "jobID":job_num['id'],
#                 "jobTitle": job_num["jobTitle"],
#                 'jobTitle': job_num['job_title'],
#                 "PL": pl_des+ pl_ski])
        total_data.append(data)


    print("DONE")


# In[7]:


def get_raw_pl():
    ori_data, field_names = Data_loading("../testing_data/raw_data_post/")
     
    print("load data")
    total_data = [] 
    
    for i, f in enumerate(ori_data):  
        print(field_names[i])
        data = []
#         data[field_names[i]] = []
    
        for num, job_num in enumerate(ori_data[i][field_names[i]]):
            if num%500 ==0:
                print(num)

            pl_des = get_pl_keywords(job_num["descirbe"])
            pl_ski = []
            data.append(pl_des+pl_ski)

#             data.append([
#                 "jobID":job_num['id'],
#                 "jobTitle": job_num["jobTitle"],
#                 'jobTitle': job_num['job_title'],
#                 "PL": pl_des+ pl_ski])
        total_data.append(data)
    return total_data


# In[12]:


from TFIDF import TFIDF


# In[8]:


def get_tfidf():
    total_data = get_raw_pl()
    tf_idf = TFIDF(total_data)
    tfidf_scores, words = tf_idf.get_tfidf()
    
    return tfidf_scores


# In[14]:


# tfidf_scores = get_tfidf()
# print(tfidf_scores[0].get("javascript"))


# In[44]:


# for i in total_data:
#     for j in i:
#         for pl in j:
#             print(tfidf_scores[0].get(pl))
# #         print(j)
# #     print(i)

