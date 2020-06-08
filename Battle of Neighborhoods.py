#!/usr/bin/env python
# coding: utf-8

# ## First I will download all the dependencies that we will need

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

print('Libraries imported successfully.')


# ## Download and Export Dataset
# 
# As stated the dataset has already been located as a .json file so I will download the dataset using wget command.

# In[2]:


get_ipython().system("wget -q -O 'newyork_data.json' https://cocl.us/new_york_dataset")
print('Data loaded successfully')


# Now, I will load the data and explore the data

# In[3]:


with open('newyork_data.json') as json_data:
    newyorkcity_data = json.load(json_data)


# In[4]:


newyorkcity_data


# We can notice that all data is in features key, so I will define a new key that contains the features variable only.

# In[5]:


newyork_features=newyorkcity_data['features']
newyork_features[0]


# ## Tranforming this data into a pandas dataframe

# I will tranform this data into a pandas dataframe. For that step I will first create an empty dataframe and then load the newyork_features data into it.

# In[6]:


# define the dataframe columns
column_names = ['Borough', 'Neighborhood', 'Latitude', 'Longitude'] 

# instantiate the dataframe
newyork_boroughs = pd.DataFrame(columns=column_names)
newyork_boroughs


# Lets load newyork_features data in this dataframe now

# In[7]:


for data in newyork_features:
    borough = data['properties']['borough'] 
    newyork_name = data['properties']['name']
        
    newyork_latlon = data['geometry']['coordinates']
    newyork_lat = newyork_latlon[1]
    newyork_lon = newyork_latlon[0]
    
    newyork_boroughs = newyork_boroughs.append({'Borough': borough,
                                          'Neighborhood': newyork_name,
                                          'Latitude': newyork_lat ,
                                          'Longitude': newyork_lon}, ignore_index=True)


# In[8]:


newyork_boroughs.head()


# We have to make sure that the dataframe contains five boroughs

# In[9]:


print('The dataframe has {} boroughs.'.format(
        len(newyork_boroughs['Borough'].unique())
    )
)


# ## Using Geopy Library to get Longitude and Latitude of New York City

# In[10]:


address = 'New York City, NY'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of New York City are {}, {}.'.format(latitude, longitude))


# ## Creating a map of New York using Folium library

# In[11]:


# create map of New York using latitude and longitude values
newyork_map = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(newyork_boroughs['Latitude'], newyork_boroughs['Longitude'], newyork_boroughs['Borough'], newyork_boroughs['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.6,
        parse_html=False).add_to(newyork_map)  
    
newyork_map


# The above map shows the boroughs and neighborhoods of New York City. We will use this data to compare the amount of Italian restaurants in Manhattan.

# ## I. Exploring Manhattan
# 
# I will start with exploring Manhattan and create a new dataframe manhattan_df where I will slice the original dataframe newyork_boroughs

# In[12]:


manhattan_df = newyork_boroughs[newyork_boroughs['Borough'] == 'Manhattan'].reset_index(drop=True)
manhattan_df.head()


# ### Getting geographical coordinates of Manhattan

# In[13]:


address = 'Manhattan, NY'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Manhattan are {}, {}.'.format(latitude, longitude))


# ### Visualizing the map of Manhattan 

# In[14]:


# create map of Manhattan using latitude and longitude values
map_manhattan = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(manhattan_df['Latitude'], manhattan_df['Longitude'], manhattan_df['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='red',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.6,
        parse_html=False).add_to(map_manhattan)  
    
map_manhattan


# ## Loading Foursqure Credentials

# In[15]:


CLIENT_ID = 'RBJ1ZXNLLVUWAW1F5HCZEOTNODCMA10LKVIBHUN0GOKOXZHC' # your Foursquare ID
CLIENT_SECRET = '4QKW0DHRXDOSDNL2QHYKPWTDCHLIQA5F4RMIE0RWBHS33XLI' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# She stays in this area and will use the address to conduct the analysis in Manhattan

# In[16]:


address = '33 W 55th St, New York, NY'

geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print(latitude, longitude)


# ### I will run a query for Italian restaurants in the area first

# In[17]:


search_query = 'Italian'
radius = 500
print(search_query + ' .... OK!')


# Define the corresponding URL

# In[18]:


url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query, radius)
url


# Sednding Get request to obtain and examine the result

# In[19]:


results = requests.get(url).json()
results


# In[20]:


# assign relevant part of JSON to venues
manhattan_venues = results['response']['venues']

# tranform venues into a dataframe
venues_manhattan_df = json_normalize(manhattan_venues)
venues_manhattan_df.head()


# Now I will filter the Dataframe to only show the information that is important to the client

# In[21]:


# keep only columns that include venue name, and anything that is associated with location
filtered_columns = ['name', 'categories'] + [col for col in venues_manhattan_df.columns if col.startswith('location.')] + ['id']
venues_manhattan_df_filtered = venues_manhattan_df.loc[:, filtered_columns]

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# filter the category for each row
venues_manhattan_df_filtered['categories'] = venues_manhattan_df_filtered.apply(get_category_type, axis=1)

# clean column names by keeping only last term
venues_manhattan_df_filtered.columns = [column.split('.')[-1] for column in venues_manhattan_df_filtered.columns]

venues_manhattan_df_filtered


# In[22]:


venues_manhattan_df_filtered.count()


#  ### We can observe that there are 10 Italian restaurants in this area. 
#  We will check the rating of two randomly selected restaurants to get some idea of the quality of the food as well. 

# In[23]:


venue_id = '51e7310c498e639ed27062b1' # ID of Quality Italian
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)
url


# ### Send GET request for result

# In[24]:


result = requests.get(url).json()
print(result['response']['venue'].keys())
result['response']['venue']


# In[25]:


try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')


# This rating for this restaurant is quite good. Now we will also check rating for another restaurant.

# In[92]:


venue_id = '50024e24e4b093944cc9f94a' # ID of Serifina Italian
url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(venue_id, CLIENT_ID, CLIENT_SECRET, VERSION)

result = requests.get(url).json()
try:
    print(result['response']['venue']['rating'])
except:
    print('This venue has not been rated yet.')


# In this case, Veronica can start with Quality Italian!
# 

# ## To give Veronica more ideas of the Italian options in the city, we'll also find out what area has the most Italian restaurants so that she can take her time to try them out. 

# In[93]:


def geo_location(address):
    # get geo location of address
    geolocator = Nominatim(user_agent="foursquare_agent")
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    return latitude,longitude


def get_venues(lat,lng):
    #set variables
    radius=400
    LIMIT=100
    #url to fetch data from foursquare api
    url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
    # get all the data
    results = requests.get(url).json()
    venue_data=results["response"]['groups'][0]['items']
    venue_details=[]
    for row in venue_data:
        try:
            venue_id=row['venue']['id']
            venue_name=row['venue']['name']
            venue_category=row['venue']['categories'][0]['name']
            venue_details.append([venue_id,venue_name,venue_category])
        except KeyError:
            pass
    column_names=['ID','Name','Category']
    df = pd.DataFrame(venue_details,columns=column_names)
    return df


def get_venue_details(venue_id):
    #url to fetch data from foursquare api
    url = 'https://api.foursquare.com/v2/venues/{}?&client_id={}&client_secret={}&v={}'.format(
            venue_id,
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION)
    # get all the data
    results = requests.get(url).json()
    print(results)
    venue_data=results['response']['venue']
    venue_details=[]
    try:
        venue_id=venue_data['id']
        venue_name=venue_data['name']
        venue_likes=venue_data['likes']['count']
        venue_rating=venue_data['rating']
        venue_tips=venue_data['tips']['count']
        venue_details.append([venue_id,venue_name,venue_likes,venue_rating,venue_tips])
    except KeyError:
        pass
    column_names=['ID','Name','Likes','Rating','Tips']
    df = pd.DataFrame(venue_details,columns=column_names)
    return df


def get_new_york_data():
    url='https://cocl.us/new_york_dataset'
    resp=requests.get(url).json()
    # all data is present in features label
    features=resp['features']
    # define the dataframe columns
    column_names = ['Borough', 'Neighborhood', 'Latitude', 'Longitude'] 
    # instantiate the dataframe
    new_york_data = pd.DataFrame(columns=column_names)
    for data in features:
        borough = data['properties']['borough'] 
        neighborhood_name = data['properties']['name']
        neighborhood_latlon = data['geometry']['coordinates']
        neighborhood_lat = neighborhood_latlon[1]
        neighborhood_lon = neighborhood_latlon[0]
        new_york_data = new_york_data.append({'Borough': borough,
                                          'Neighborhood': neighborhood_name,
                                          'Latitude': neighborhood_lat,
                                          'Longitude': neighborhood_lon}, ignore_index=True)
    return new_york_data


# In[28]:


ny_data = get_new_york_data()
ny_data.head()


# In[29]:


ny_data.shape


# In[31]:


from matplotlib import pyplot as plt
clr = "dodgerblue"
ny_data.groupby('Borough')['Neighborhood'].count().plot.bar(figsize=(10,5), color=clr)
plt.title('Neighborhoods per Borough: NYC', fontsize = 20)
plt.xlabel('Borough', fontsize = 15)
plt.ylabel('No. Neighborhoods',fontsize = 15)
plt.xticks(rotation = 'horizontal')
plt.show()


# In[32]:


# queens has most neighborhoods
# prepare neighborhood list that contains Italian resturants
column_names=['Borough', 'Neighborhood', 'ID','Name']
italian_rest_ny=pd.DataFrame(columns=column_names)
count=1
for row in ny_data.values.tolist():
    Borough, Neighborhood, Latitude, Longitude=row
    venues = get_venues(Latitude,Longitude)
    italian_resturants=venues[venues['Category']=='Italian Restaurant']   
    print('(',count,'/',len(ny_data),')','Italian Resturants in '+Neighborhood+', '+Borough+':'+str(len(italian_resturants)))
    print(row)
    for resturant_detail in italian_resturants.values.tolist():
        id, name , category=resturant_detail
        italian_rest_ny = italian_rest_ny.append({'Borough': Borough,
                                                'Neighborhood': Neighborhood, 
                                                'ID': id,
                                                'Name' : name
                                               }, ignore_index=True)
    count+=1


# In[33]:


italian_rest_ny.to_csv('italian_rest_ny_tocsv1.csv') # Save the information so far to a .csv file due to limited calls on FourSquare


# In[34]:


italian_rest_ny = pd.read_csv('italian_rest_ny_tocsv1.csv')
italian_rest_ny.tail()


# In[35]:


italian_rest_ny.shape


# In[36]:


italian_rest_ny.groupby('Borough')['ID'].count().plot.bar(figsize=(10,5), color = clr)
plt.title('Italian Resturants per Borough: NYC', fontsize = 20)
plt.xlabel('Borough', fontsize = 15)
plt.ylabel('No.of Italian Resturants', fontsize=15)
plt.xticks(rotation = 'horizontal')
plt.show()


# In[37]:


NOofNeigh = 6 # top number for graphing all the same past 6
italian_rest_ny.groupby('Neighborhood')['ID'].count().nlargest(NOofNeigh).plot.bar(figsize=(10,5), color=clr)
plt.title('Italian Resturants per Neighborhood: NYC', fontsize = 20)
plt.xlabel('Neighborhood', fontsize = 15)
plt.ylabel('Italian Resturants', fontsize=15)
plt.xticks(rotation = 'horizontal')
plt.show()


# So Belmont has the most Italian restaurants while Lenox Hill the least, so Veronica can save the hassel of going there at all. Veronica can also explore this area once she's done with the first Italian restaurant and continue her food journey!

# In[ ]:




