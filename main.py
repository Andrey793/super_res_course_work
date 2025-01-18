import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import cv2
import pathlib


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
                               param1=160, param2=75, minRadius=35, maxRadius=200)

    if circles is not None:
        return True
    return False


def parse_url(url, i=0):
    options = webdriver.chrome.options.Options()
    options.add_argument('--disable-blink-features=AutomationControlled')
    #options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--enable-cookies')

    service = webdriver.chrome.service.Service(chrome_driver_path='/usr/local/bin/chromedriver')

    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    image_elements = driver.find_elements(By.TAG_NAME, 'img')
    for index, image in enumerate(image_elements):
        src = image.get_attribute('src')
        if src and src.startswith('http'):
            responses = requests.get(src)
            with open(f'image{index}_{i}.png', 'wb') as f:
                f.write(responses.content)
            if not detect_circles(f'image{index}_{i}.png'):
                file = pathlib.Path(f'image{index}_{i}.png')
                file.unlink(missing_ok=True)
            print(f"Downloaded {src}")
    driver.quit()


parse_url('https://www.cloudynights.com/gallery/category/322-lunar/')
for i in range(1, 41):
    parse_url(f'https://www.cloudynights.com/gallery/category/322-lunar/?st={48*i}', i)


