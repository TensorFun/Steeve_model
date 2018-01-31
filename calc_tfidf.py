
# coding: utf-8

# In[ ]:

from os import listdir
from collections import Counter, defaultdict
from functools import reduce
import json
import math


# In[ ]:




# In[ ]:

def read_files(directory):
    pass

def get_field_counter(posts):
    clean_NC = []
    for post in posts[:300]:
        clean_NC += list(map(lambda el: el.replace('\n', '').replace('\xa0', '').lower(), post['NC'])) # TODO: '\n' deprecated yet? and refactor
    return Counter(clean_NC)


# In[ ]:




# In[ ]:

if __name__ == "__main__":
    KEYWORD_PATH = '/home/fun/Atos/Steeve_data/candidates_keyword/'
#     read_files(KEYWORD_PATH)
    # print(open(KEYWORD_PATH + 'Keyword_Front_End.txt').read())
    
    ########### Get all fields counter and total counter
    fields_counter = {} 
    files = listdir(KEYWORD_PATH)
    for file in files:
        doc = json.loads(open(KEYWORD_PATH + file, 'r', encoding='utf8').read())
        for field in doc:
            cnt = get_field_counter(doc[field])
            
            for k in list(cnt):
                if cnt[k] < 25: # threshold
                    del cnt[k]
            
            fields_counter[field] = cnt # get_field_counter(doc[field])

    total_counter = reduce(lambda a, b: a + b, fields_counter.values())
    ############
    
    
    ############ Count tfidf
    tfidf = defaultdict(lambda: defaultdict(lambda: 0))
    word_num = len(total_counter.keys())
    total_count = sum(total_counter.values())
    for field in fields_counter:
        field_count = sum(fields_counter[field].values())
        other_count = total_count - field_count
        
        for key in fields_counter[field]:
            # TODO: module: FORMULA
            score = fields_counter[field][key] / total_counter[key]
            
#             p_w_f = math.log(fields_counter[field][key]+1 / field_count)
#             p_w_o = math.log((total_counter[key] - fields_counter[field][key])+1 / other_count)
#             score = p_w_f - p_w_o
            
            tfidf[field][key] = score
#     print(tfidf)
#     print(fields_counter)


    


# In[ ]:




# In[ ]:

for field in tfidf:
    a = sorted(tfidf[field].items(), key=lambda k_v: k_v[1], reverse=True) #2010
    print(a)


# In[ ]:

print(tfidf["Front_End"]['angularjs'])


# In[ ]:



