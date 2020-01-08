import requests
import json
import pandas as pd
from bs4 import BeautifulSoup


# ZILLOW API
# Pull all LA property value data for each residential address in LA from Zillow's API, if available
# GET Calls return xml data -> use beautiful soup

# ENDPOINT PARAMETERS
ZWSID = "**-**************_*****"  # API Key
citystatezip = 'Los+Angeles%2C+CA'

# ENDPOINTS
base = "http://www.zillow.com/webservice/"
region_children = "GetRegionChildren.htm?zws-id={ZWSID}&state={state}&city={city}&childtype=neighborhood"
address_search = "GetSearchResults.htm?zws-id={ZWSID}&address={address}&citystatezip={citystatezip}"

# Retrieve City of LA Address Data (> 1 million addresses)
data = pd.read_csv("Addresses_in_the_City_of_Los_Angeles.csv")
data = data.astype({'HSE_NBR':'str'}) # Change the col of address #'s from int to str

# Add a property value column to the dataframe
value = [0] * data.shape[0]
data['Value'] = value


i=0

print('dataset parsing complete')

for i in range(data.shape[0]):
    number = data.HSE_NBR[i]
    direction = str(data.HSE_DIR_CD[i])
    street_name = data.STR_NM[i]
    street_suffix = str(data.STR_SFX_CD[i])
    
    address = number + '+' + direction + '+' + street_name + '+' + street_suffix
    url = f"http://www.zillow.com/webservice/GetSearchResults.htm?zws-id={ZWSID}&address={address}&citystatezip={citystatezip}"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # Check if the property is registered on zillow and store its value
    if soup.find('amount') != None and len(list(soup.find('amount').children)) != 0:
        #print(soup.find('amount'))
        data.loc[i,'Value'] = list(soup.find('amount').children)[0]
    
    if i == 1:
        break
    
data.to_csv("property_values.csv")
