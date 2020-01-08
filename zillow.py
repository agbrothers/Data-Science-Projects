import requests
import json
import pandas as pd
from bs4 import BeautifulSoup


# -----------------------------------------------------------------------------
"""                        EXAMPLE API CALLS

Ex Call: a listing of neighborhoods in Seattle, WA:
• http://www.zillow.com/webservice/GetRegionChildren.htm?zws-id={ZWSID}&state=wa&city=seattle&childtype=neighborhood

Ex Call: the exact address match "2114 Bigelow Ave", "Seattle, WA":
• http://www.zillow.com/webservice/GetSearchResults.htm?zws-id={ZWSID}&address=2114+Bigelow+Ave&citystatezip=Seattle%2C+WA





                    Retrieving Zipcodes for LA County                       

zip_page = requests.get("https://www.zip-codes.com/county/ca-los-angeles.asp")
zip_soup = BeautifulSoup(zip_page.content, 'html.parser')
zip_data = zip_soup.find_all('td',attrs={'class':"label"})

zip_codes = []
for i in range(1,len(zip_data)-17):
    zip_codes.append(list(list(zip_data[i].children)[0].children)[0][9:14])
    # this is just convoluted notation for pulling the text out of 
    # some nested <a> tags within each table row
"""
# -----------------------------------------------------------------------------



# ZILLOW API
# Store and compare LA rental data by zipcode
# GET Calls return xml data -> use beautiful soup

# ENDPOINT PARAMETERS
ZWSID = "X1-ZWz1hjeqe8mbrf_8z2s9"  # API Key
citystatezip = 'Los+Angeles%2C+CA'

# ENDPOINTS
base = "http://www.zillow.com/webservice/"
region_children = "GetRegionChildren.htm?zws-id={ZWSID}&state={state}&city={city}&childtype=neighborhood"
address_search = "GetSearchResults.htm?zws-id={ZWSID}&address={address}&citystatezip={citystatezip}"

# Retrieve City of LA Address Data
data = pd.read_csv("Addresses_in_the_City_of_Los_Angeles.csv")
data = data.astype({'HSE_NBR':'str'}) # Change the col of address num's to str

# Add a property value column to the dataframe
value = [0] * data.shape[0]
data['Value'] = value


i=0

print('dataset parsing complete')

for i in range(100000,996263):
    number = data.HSE_NBR[i]
    direction = str(data.HSE_DIR_CD[i])
    street_name = data.STR_NM[i]
    street_suffix = str(data.STR_SFX_CD[i])
    
    address = number + '+' + direction + '+' + street_name + '+' + street_suffix
    url = f"http://www.zillow.com/webservice/GetSearchResults.htm?zws-id={ZWSID}&address={address}&citystatezip=Los+Angeles%2C+CA"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # Check if the property is registered on zillow and store its value
    if soup.find('amount') != None and len(list(soup.find('amount').children)) != 0:
        #print(soup.find('amount'))
        data.loc[i,'Value'] = list(soup.find('amount').children)[0]
    
    if i == 1:
        break
    
print(data.head(10))


data.to_csv("property_values3.csv")

# 162241

