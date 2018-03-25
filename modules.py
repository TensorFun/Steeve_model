
# coding: utf-8

# In[1]:

# Extract PL

from flashtext import KeywordProcessor

pls = open('Rule.txt', 'r', encoding='utf8').read().split('\n')

keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(pls)

def get_pl_keywords(content):
    return keyword_processor.extract_keywords(content)

def norm_pls(pls):
    '''
    Params: ['pl_A', 'pl_B', 'pl_C']
    
    Returns: list for normalized pls
    '''
    return [pl.lower().replace(' ', '_').replace('.js', '') for pl in pls]


# In[26]:

from sklearn.metrics.pairwise import cosine_similarity

def pick_k_jobs(user_pls, posts, k=100):
    '''
    Params:
    - user_pls: ['pl_A', 'pl_B', 'pl_C']
    - posts: every post's pls in specific field [{id, pl}, {id, pl}, {id, pl}]
    - k: return top k jobs and must match at least one pl
    
    Return: Top 100 jobs in suitable order
    '''
    matches = [len(set(user_pls).intersection(set(post['PL']))) for post in posts]

    top_jobs = sorted(zip(posts, matches), key=lambda pair: pair[1], reverse=True)
    top_k_jobs = filter(lambda pair: pair[1] > 0, top_jobs[:k])
    top_k_jobs = list(map(lambda pair: pair[0], top_k_jobs))
    return top_k_jobs


# In[28]:

# pick_k_jobs(['A','B','C'], [{'id': 1, 'PL': ['A','B']}, {'id': 12, 'PL': ['D']}, {'id': 3, 'PL': ['E']}])


# In[ ]:




# In[ ]:



