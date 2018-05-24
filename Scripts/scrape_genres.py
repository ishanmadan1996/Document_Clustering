import csv
import re
from selenium import webdriver
import time
from bs4 import BeautifulSoup


chrome = r"C:\Users\Ishant\Downloads\chromedriver.exe" #chromedriver
driver = webdriver.Chrome(chrome) #initialise chrome webdriver
urls_page = ['https://wogma.com/movies/alphabetic/basic/#items_with_'+
             str(i) for i in range(1, 10)] #pagenation links
title_list = []
review_list = []
url_extension = 'https://wogma.com'

driver.get('https://wogma.com/movies/alphabetic/basic/#items_with_1')
try:
        while True:  # used to simulate scrolling action of a webpage on selenium browser
            last_height = driver.execute_script("return document.body.scrollHeight")
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(1.5)

            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
except:
        print
        ''

page = driver.page_source
soup = BeautifulSoup(page, 'html.parser')
review = soup.findAll('div', {"class": "button related_pages review "})  # get all div tags which contain 'a' tags
genre = []
for div in review:
    try:
        links = div.find_all('a')  # extract the required links only

        for j in links:
            driver.get(url_extension + j['href'])
            # print(url_extension + j['href'])
            page = driver.page_source
            soup = BeautifulSoup(page, 'html.parser')
            gen = soup.find('div',{'class':''})
            try :
                # genre1 = driver.find_element_by_xpath('//*[@id="review-page-top"]/div[2]/div[5]/a[1]').text
                # genre2 = driver.find_element_by_xpath('//*[@id="review-page-top"]/div[2]/div[5]/a[2]').text
                # genre3 = driver.find_element_by_xpath('//*[@id="review-page-top"]/div[2]/div[5]/a[3]').text
                if re.search('Genres:',driver.find_element_by_xpath('//*[@id="review-page-top"]/div[2]/div[5]').text):
                    g = driver.find_element_by_xpath('//*[@id="review-page-top"]/div[2]/div[5]').text
                else:
                    g = driver.find_element_by_xpath('//*[@id="review-page-top"]/div[2]/div[6]').text

            except Exception as e:
                print('hi')


            genre = g
            print(str(j)+''+genre)
            with open('C:\\Users\Ishant\Desktop\Scraped_Data_Unfound\\genre.txt', 'a') as textfile:
                textfile.write(genre + '\n')

                textfile.close()
    except Exception as e:
        print('')
