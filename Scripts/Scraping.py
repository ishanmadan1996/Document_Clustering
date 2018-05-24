from selenium import webdriver
import time
from bs4 import BeautifulSoup


chrome = r"C:\Users\Ishant\Downloads\chromedriver.exe" #chromedriver
driver = webdriver.Chrome(chrome) #initialise chrome webdriver
url_extension = 'https://wogma.com' #extesnion to be used later

driver.get('https://wogma.com/movies/alphabetic/basic/#items_with_1') #open selenium driver with given url
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

page = driver.page_source #extract page source
soup = BeautifulSoup(page, 'html.parser') #parsing the page and storing it in soup
review = soup.findAll('div', {"class": "button related_pages review "})  # get all div tags which contain 'a' tags

#to extract reviews
for div in review:
    try: #for handling exceptions
        links = div.find_all('a')  # extract the required href links only
        print(div)
        for j in links: #iterate through each link and extract the required data from the link
            driver.get(url_extension + j['href'])
            print(url_extension + j['href'])
            page = driver.page_source
            soup = BeautifulSoup(page, 'html.parser')
            review = soup.find('div', {'class': 'wogma-review'}) #extract review from div tag and store in review variable
            review = review.text
            with open('C:\\Users\Ishant\Desktop\Scraped_Data_Unfound\\reviews.txt', 'a') as textfile: #saving each review in a txt file
                textfile.write(review + '\n\n' + 'Break Here' + '\n\n')
                textfile.close()
    except Exception as e:
        print('')

#to extract movie titles
for i in range(2,1405):
    try:

        title = driver.find_element_by_xpath('/html/body/div[2]/div/div[1]/div[1]/div/div/table/tbody/tr[' + str(i) + ']/td[1]') #finding movie title element using it's xpath
        title = title.text
        print(title)
        with open('C:\\Users\Ishant\Desktop\Scraped_Data_Unfound\\title.txt', 'a') as textfile:
            textfile.write(title+'\n')
            textfile.close()

    except Exception as e:
        print('')


driver.close() #close web driver