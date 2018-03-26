
# coding: utf-8

# In[120]:

import json
from get_pl_filter import get_pl_filter
from flashtext import KeywordProcessor

# from wappalyzer and augment list
pl_filter = get_pl_filter()
wap_pls = []
for pl in [pl for cate in pl_filter for pl in pl_filter[cate]]: # flatten
    if '.' in pl:
        wap_pls += [pl, pl.replace('.', '')]
    elif '_' in pl:
        wap_pls += [pl, pl.replace('_', ' ')]
    elif '-' in pl:
        wap_pls += [pl, pl.replace('-', ' '), pl.replace('-', '')]
    else:
        wap_pls += [pl]

# from stackoverflow, tags in order of popularity
stack_tags = open('stack_filter.txt', 'r', encoding='utf8').read().split(',') # total 30*200
stack_tags = stack_tags[:30*50]

# combine together and remove duplicates
all_pls = list(set(wap_pls + stack_tags))
# all_pls = list(set(wap_pls))

keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(all_pls)

# print(all_pls)


# In[121]:

def get_pl_keywords(content):
    return keyword_processor.extract_keywords(content)


# In[122]:

if __name__ == "__main__":
    # test
    with open('../Data/Front-End-Developer-NY-3000.json') as fs:
        for jobfield in json.loads(fs.read()):
            print(jobfield)
            for post in articles[jobfield]:
                keywords = get_pl_keywords(post['skills'] + " " + post['jobDescription'])
#                 print(keywords)


# In[ ]:




# In[ ]:



