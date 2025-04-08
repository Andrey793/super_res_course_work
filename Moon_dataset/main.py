import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import cv2
import pathlib
import csv

def detect_circles(image)->bool:
    try:
        image = cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    except Exception as e:
        return False

    # Apply Gaussian Blur (Helps in Circle Detection)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=90,
                               param1=160, param2=75, minRadius=35, maxRadius=700)

    if circles is not None:
        return True
    return False

def parse_page(url, i=0):
    options = webdriver.chrome.options.Options()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--enable-cookies')

    service = webdriver.chrome.service.Service(chrome_driver_path='/usr/local/bin/chromedriver')

    driver = webdriver.Chrome(service=service, options=options)

    driver.get(url)
    ul_element = driver.find_element(By.XPATH, '//*[@id="content"]/div[6]/div/div[2]/ul')

    # Find all <li> elements with class "gallery image" inside the <ul>
    li_elements = ul_element.find_elements(By.CLASS_NAME, "gallery_image")
    csv_filename = "cloudy_nights_full/data.csv"
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for index, li in enumerate(li_elements):
            a_tag = li.find_element(By.TAG_NAME, "a")  # Find the <a> inside <li>
            link = a_tag.get_attribute("href")  # Get the href value
            file, size = parse_url(link, index, i)
            if file is not None:
                writer.writerow([file, size[0], size[1], link])



def parse_url(url, index, i=0):
    options = webdriver.chrome.options.Options()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--enable-cookies')

    service = webdriver.chrome.service.Service(chrome_driver_path='/usr/local/bin/chromedriver')

    driver = webdriver.Chrome(service=service, options=options)

    driver.get(url)


    # Find the <img> inside the <div>
    answer = [None, None]
    try:
        div_element = driver.find_element(By.ID, "theImage")
        img_element = div_element.find_element(By.TAG_NAME, "img")
        # Get the image source URL
        image_url = img_element.get_attribute("src")
        print("Image URL:", image_url)
        # Download the image
        if image_url:
            file_name = f'cloudy_nights_full/image{index}_{i}.png'
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(file_name, "wb") as file:
                    file.write(response.content)
                if not detect_circles(file_name):
                    file = pathlib.Path(file_name)
                    file.unlink(missing_ok=True)
                else:
                    image = cv2.imread(file_name)
                    height, width = image.shape[:2]
                    answer = [file_name, [height, width]]
    except Exception as e:
        print("Something went wrong:")
        print(e)
    driver.quit()
    return answer


csv_filename = "cloudy_nights_full/data.csv"
headers = ["name", "height", "width", "url"]
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
parse_page('https://www.cloudynights.com/gallery/category/322-lunar/')
for i in range(1, 41):
    parse_page(f'https://www.cloudynights.com/gallery/category/322-lunar/?st={48*i}', i)
