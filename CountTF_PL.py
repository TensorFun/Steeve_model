
# coding: utf-8

# In[1]:


import json
from collections import Counter
import os

PATH = "../Steeve_data/candidates_keyword/"
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
    
#     for post_id in field_post[:10]:
    for post_id in field_post:
        pl = map(lambda el: el.lower(), post_id["PL"])
        post_cnt = Counter(pl)

        
        # post_id use jobTitle as ID temporarily
        posts_PL[field_name]=write_to_js(posts_PL,field_name,post_id["jobTitle"],post_cnt)

    
    return posts_PL

def write_to_js(posts_PL,field_name,post_id,cnt):


    posts_PL[field_name].append({  
    "jobTitle":post_id ,
    "PL_value":cnt })
    
    return posts_PL[field_name]

def Count_Total_TF(fields_posts_PL):
    
    Total_PL =Counter()
    Total_PL_VALUE =[]
    Total_Comparison = {}  
        
    for i,f in enumerate(fields_posts_PL): # each feild
 
        field_name = field_names[i]
        fpost = Counter()
        Total_Comparison[field_name] = []
        
        for f_post in f[field_name]:
            fpost += f_post["PL_value"]

        Total_PL_VALUE.append(fpost)
        Total_PL += fpost

    for i, f in enumerate(Total_PL_VALUE):
        field_name = field_names[i]
        for word in f:
            f[word] = f.get(word)/ Total_PL.get(word)
#             print(f)
#         Total_Comparison.append(f)
#         print(fpost)
            
        Total_Comparison[field_name]=write_to_js(Total_Comparison,field_name,field_name,f)
#         S.append(field_Com_PL)
        
            
    return Total_Comparison
    
# print(feild_value[0]["Back_End"][0][])


# In[3]:


if __name__ == "__main__":
    
    files_keyword, field_names = Data_loading(PATH)
    for i,field in enumerate(files_keyword):
        print(i)
        field_name = field_names[i]
        posts_PL = Count_TF(field,field_name)
        fields_posts_PL.append(posts_PL) 

        with open('../Steeve_data/PL_value/PL_posts_in_'+field_name+'.txt', 'w') as f:
            json.dump(posts_PL, f)
            
    Total_Fields_PL_V= Count_Total_TF(fields_posts_PL)
    with open('../Steeve_data/PL_value/Total_Comparison_Fields.txt', 'w') as f:
            json.dump(Total_Fields_PL_V, f)


# In[4]:


text = file_data[0][feild_names[0]]

# for i in text:
#     print(i["PL"])


# sss =  Count_TF(text)
sss = Count_TF(text,feild_names[0])
# print(sss[0])
print(sss[1])
Count_TF(field[feild_names[i]],feild_names[i][1])#TOATL

Counter(words).most_common(10)

