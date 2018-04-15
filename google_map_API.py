
# coding: utf-8

# In[7]:

import time
import googlemaps
# google_Key = 'AIzaSyDoQASuNxOvnMLZKPUmEk9bpplb-8e5now'
api_keys = ['AIzaSyDoQASuNxOvnMLZKPUmEk9bpplb-8e5now',\
            'AIzaSyAnV4U22OWug0zY-E5osq-gtv87TVA0xS0',\
            'AIzaSyBlPC_9n8jjRm7MNZy-y11FFBGMzGHPREI',\
            'AIzaSyCRs0DOtYIMhxQtpvqTnSwG4vwRj5wmMAI']
# gmaps = googlemaps.Client(key=google_Key)
import random 
import signal
# r_key = random.randint(0,3)
# gmaps = googlemaps.Client(key=api_keys[r_key])

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


# def get_location_range(address):
#     # Change the behavior of SIGALRM
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(5)
#     address_range = []
#     try :
#         geocode_result = gmaps.geocode(address)
#         first_filter = geocode_result[0]['address_components']
#         #[0]['long_name']
#         for a in first_filter[:2]:
#             address_range.append(a['long_name'])
        
#     except:
#         address_range.append(address)
# #     print(address_range)

#     return address_range

def get_all_location_range(address_list):
    # Change the behavior of SIGALRM
    all_address_range = []
    loc_len = len(address_list)
    r_key = 0
    gmaps = googlemaps.Client(key=api_keys[r_key])
    for n,address in enumerate(address_list):
        if r_key != int(n//2000%4):
            r_key = int(n//2000%4)
            gmaps = googlemaps.Client(key=api_keys[r_key])
        else:
            r_key = int(n//2000%4)
        

        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        address_range = []
        try :
            geocode_result = gmaps.geocode(address)
            first_filter = geocode_result[0]['address_components']
        #[0]['long_name']
            for a in first_filter[:2]:
                address_range.append(a['long_name'])
        
        except:
            address_range.append(address)
        all_address_range.append(address_range)
#     print(address_range)
    

    return all_address_range


# In[1]:


def location_filter(user_target, company_range):
    
    for c_address in company_range:
        if user_target == c_address:
            break
    
    return True
    


# In[8]:


"""get_location_range('taichung')"""
"""location_filter('taichung',['taipei','china',''usa])"""

