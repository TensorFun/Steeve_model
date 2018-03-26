
# coding: utf-8

# # Count PL value in each field and total

# In[1]:


import json
from collections import Counter
import os

PATH = "../Steeve_data/no_filter_Dice/can/"
fields_posts_PL =[] # PL of each post in every fileds
total_PL = []


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
                print(key)
                key.append(k)
            
    return file_data,key

def Count_TF(field,field_name):
    
    # input js file, counting ["PL"] for each post in js file
    total_cnt = Counter()
    posts_PL = {}  
    posts_PL[field_name] = []
    
    field_post = field[field_name]
    
    # for post_id in field_post[:300]:
    for post_id in field_post:
        pl = map(lambda el: el.lower(), post_id["PL"])
        post_cnt = Counter(pl)
        
        ################  TESTING -- regards repeat pl as 1 time ###################
        for i in post_cnt:
            post_cnt[i] = 1
        ################  TESTING -- regards repeat pl as 1 time ###################
        
        # post_id use jobTitle as ID temporarily
        posts_PL[field_name]=write_to_js(posts_PL,field_name,post_id["jobTitle"],post_cnt,post_id["url"])

    
    return posts_PL

def write_to_js(posts_PL,field_name,post_id,cnt,url):


    posts_PL[field_name].append({  
    "jobTitle":post_id,
    "PL_value":cnt,
    "url":url})
    
    return posts_PL[field_name]

def Count_Total_TF(fields_posts_PL):
    
    Total_PL =Counter()
    Total_PL_VALUE =[]
    Total_Comparison = {}  
    ##########   TESTING -- get certain pl  
    # jobN = []
        
    for i,f in enumerate(fields_posts_PL): # each feild
 
        field_name = field_names[i]
        fpost = Counter()
        Total_Comparison[field_name] = []
        
        print(field_name)
        
        ##########   TESTING -- get certain pl  
        # jobN.append(field_name)
        ##########
        
        for f_post in f[field_name]:
            fpost += f_post["PL_value"]
            
        ##########   TESTING -- get certain pl   ##########
        #    if (f_post["PL_value"].get("css")):
        #        jobN.append((f_post["url"]))
        ###########   TESTING -- get certain pl  ##########

        Total_PL_VALUE.append(fpost)
        Total_PL += fpost

    for i, f in enumerate(Total_PL_VALUE):
        field_name = field_names[i]
        for word in f:
            f[word] = f.get(word)/ Total_PL.get(word)
#             print(f)
#         Total_Comparison.append(f)
            
        Total_Comparison[field_name]=write_to_js(Total_Comparison,field_name,Total_PL_VALUE,f,Total_PL_VALUE)
        
    return Total_Comparison

    ##########   TESTING -- get certain pl   ##########
    # return Total_Comparison,jobN


# ## Count PL value
# ##### Total_Comparison_Fields.txt for total comparison
# ##### PL_posts_in_+field_name+.txt for each fields 

# In[3]:


if __name__ == "__main__":
    
    files_keyword, field_names = Data_loading(PATH)
    for i,field in enumerate(files_keyword):
        print(i)
        field_name = field_names[i]
        posts_PL = Count_TF(field,field_name)
        fields_posts_PL.append(posts_PL) 

        with open('../Steeve_data/no_filter_Dice/pl_count_1/PL_posts_in_'+field_name+'.txt', 'w') as f:
            json.dump(posts_PL, f)
            
    Total_Fields_PL_V= Count_Total_TF(fields_posts_PL)

########### TESTING -- get certain pl ########### 
#    Total_Fields_PL_V, b = Count_Total_TF(fields_posts_PL)
#    with open('../Steeve_data/PL_value/css.txt', 'w') as f:
#        json.dump(b, f)
########### TESTING -- get certain pl ###########
    with open('../Steeve_data/no_filter_Dice/pl_count_1/Total_Comparison_Fields.txt', 'w') as f:
            json.dump(Total_Fields_PL_V, f)


# ## TESTING

# In[ ]:


asd = json.load(open('../Steeve_data/PL_value/total_testing.txt'))
a = sorted(asd['Front End Developer'][0]["PL_value"].items(), key=lambda k_v: k_v[1], reverse=True)[:90]
#     print(asd['Backend-NY'][0]["PL_value"])
print(a)


# In[ ]:


b = sorted(asd[field_names[0]][0]["PL_value"].items(), key=lambda k_v: k_v[1], reverse=True)[:90]
print(b)


# In[ ]:


print(asd['Backend'][0]["PL_value"].get("web"))
print(asd['Front End Developer'][0]["PL_value"].get("web"))

