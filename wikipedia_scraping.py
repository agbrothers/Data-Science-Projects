import time
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver
import matplotlib.pyplot as plt


""" TOP VIEWED PAGES ON WIKIPEDIA 2015-2019 """

Years = [2015, 2016, 2017, 2018, 2019]
Months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

base = "https://tools.wmflabs.org/topviews/?project=en.wikipedia.org&platform=all-access&date="
extension = "{year}-{date}&excludes="


""" SCRAPE PAGE DATA USING FIREFOX WEBDRIVER """

all_data = pd.DataFrame()
j = 0

for year in Years:
    for month in Months:
        
        # Data only exists beyond July 2015, this skips all earlier months in 2015
        if j < 6:
            j+=1
            continue
        
        # Run the firefox webdriver from geckodriver executable path
        driver = webdriver.Firefox(executable_path = '/usr/local/bin/geckodriver')
        driver.get(base + f"{year}-{month}&excludes=")
        
        # Execute java script cmd to scroll to the bottom of the webpage
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
        
        # Sleep for 9s for the webpage to load/execute js displaying table data
        time.sleep(9)
                
        # Find elements by xpath
        Names = driver.find_elements_by_xpath("//*[@class='chart-container col-lg-12']//*[@class='table output-table']//*[@class='topview-entries']//*[@class='topview-entry']//*[@class='topview-entry--label-wrapper']//*[@class='topview-entry--label']")
        print('Firefox Webdriver - Number of results', len(Names))
        
        Views = driver.find_elements_by_xpath("//*[@class='chart-container col-lg-12']//*[@class='table output-table']//*[@class='topview-entries']//*[@class='topview-entry']//*[@class='topview-entry--views']")
        print('Firefox Webdriver - Number of results', len(Views))
        
        # Store the data
        data = []
        # Loop over Top 10 most viewed pages for that month
        i = 0
        for name in Names:
            if i == 10:
                break
            page_name = name.text
            page_views = Views[i].text
            link = name.find_element_by_tag_name('a')
            page_link = link.get_attribute("href")
            # append dict to array
            data.append({"Year" : year,"Month" : month,"Page Name" : page_name,"Monthly Views" : page_views,"Page Link" : page_link})
            i+=1
        
        # Save the data to a dataframe and append it to the master dataframe
        df = pd.DataFrame(data)
        df['Monthly Views'] = df['Monthly Views'].str.replace(',','')
        df['Monthly Views'] = df['Monthly Views'].astype(int)
        all_data = all_data.append(df)
        
        # Close the driver
        driver.quit()
        
        
# save to pandas dataframe

all_data.to_csv("topviews.csv")