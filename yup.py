import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from selenium import webdriver
from PIL import Image
import cv2
import os
import tkinter as tk


key = 'Z1UK6CGV0MLWS7VQ'
symbol = "AMZN"

ts = TimeSeries(key=key, output_format='pandas')
stonks, meta = ts.get_daily(symbol=symbol, outputsize='full')
stonks = stonks.head(100)
stonks = stonks['4. close']
stonks.index = np.arange(100, 0, -1)

lowpoints = []
lowindexes = []
highpoints = []
highindexes = []

previous = 0


def findLowest(param):
    lowest = param.iloc[0]
    for x in param:
        if x < lowest:
            lowest = x
    lowindexes.append(param[param == lowest].index[0])
    return lowest


for i in range(10, 101, 10):
    lowpoints.append(findLowest(stonks[previous: i]))
    previous = i

support = pd.Series(data=lowpoints, index=lowindexes)

plt.title(symbol)

previous = 0


def findHighest(param):
    highest = param.iloc[0]
    for x in param:
        if x > highest:
            highest = x
    highindexes.append(param[param == highest].index[0])
    return highest


for i in range(10, 101, 10):
    highpoints.append(findHighest(stonks[previous: i]))
    previous = i

resistance = pd.Series(data=highpoints, index=highindexes)

#resistance.plot()
#support.plot()
stonks.plot()

m, b = np.polyfit(highindexes, highpoints, 1)
# toppy line ^^
a, z = np.polyfit(lowindexes, lowpoints, 1)
# m, b = np.polyfit(stonks.index,stonks.values, 1)
# plt.plot(stonks.index, ((a + m) / 2) * stonks.index + ((b + z) / 2))
plt.plot(stonks.index, m * stonks.index + b)
plt.plot(stonks.index, a * stonks.index + z)
plt.ylabel("Price ($)")
plt.xlabel("Days")

t1 = TechIndicators(output_format='pandas', key=key)
ema200, meta = t1.get_ema(symbol=symbol, interval='daily', time_period=200, series_type='close')
ema50, meta = t1.get_ema(symbol=symbol, interval='daily', time_period=50, series_type='close')
ema200 = ema200.tail(100)
ema200.index = np.arange(100)
ema50 = ema50.tail(100)
ema50.index = np.arange(100)

ema50_m, ema50_b = np.polyfit(ema50.index, ema50, 1)
ema200_m, ema200_b = np.polyfit(ema200.index, ema200, 1)
plt.plot(ema50.index, ema50_m * ema50.index + ema50_b, label="50 Day Moving Average")
plt.plot(ema200.index, ema200_m * ema200.index + ema200_b, label="200 Day Moving Average")
plt.legend()

#print(stonks.pct_change())


# where they cross
negative50 = 0
positive50 = 0

positive200 = 0
negative200 = 0

contradictions = 0

count = 1

def calculate50(index, y):
    global count
    global negative50
    global positive50
    slope, y_intercept = np.polyfit(index, y, 1)
    #plt.plot(index, slope * index + y_intercept)

    if slope < 0:
        negative50 += 1
        print("negative at quarter: " + str(count))

    if slope > 0:
        positive50 += 1
    count = count + 1


count = 1
for x in range(25, 101, 25):
    calculate50(ema50[x - 25: x].index, ema50[x - 25: x])

print("negative50: " + str(negative50))
print("positive50: " + str(positive50))


def calculate200(index, y):
    global count
    global negative200
    global positive200
    slope, y_intercept = np.polyfit(index, y, 1)
    #plt.plot(index, slope * index + y_intercept)

    if slope < 0:
        negative200 += 1
        if count == 4:
            print("going down in final quadrant")

    if slope > 0:
        positive200 += 1
    count = count + 1


for x in range(25, 101, 25):
    calculate200(ema200[x - 25: x].index, ema200[x - 25: x])

print("negative200: " + str(negative200))
print("positive200: " + str(positive200))

ema50 = np.round(ema50, 2)
ema200 = np.round(ema200, 2)
intersection = np.intersect1d(ema50, ema200)
# take into account low percent changes
print(intersection)
if len(intersection) > 0:
    if ema50_m > 0:
        print("golden cross")
    else:
        print("death cross")

plt.show()


if ema50_m > 0 and ema200_m > 0 and (a + m) / 2 > 0:
    print("Bullish pattern: recommended buy")

driver = webdriver.Chrome(executable_path='C:\\Users\Andrew Stelmach\\Desktop\\chromedriver.exe')

url = "https://money.cnn.com/quote/forecast/forecast.html?symb=" + symbol
driver.get(url)
image = driver.find_element_by_id("wsod_forecasts").screenshot_as_png

driver.save_screenshot('C:\\Users\\Andrew Stelmach\\Desktop\\screenshot\\out.png')

element = driver.find_element_by_id("wsod_forecasts")

location = element.location
size = element.size

x = location['x'] + 10
y = location['y'] - 800
w = size['width']
h = size['height']

im = Image.open('C:\\Users\\Andrew Stelmach\\Desktop\\screenshot\\out.png')
im = im.crop((int(x) + 575, int(y + 850), int(x + w), int(y + h) - 150))
# im.show()
im.save('C:\\Users\\Andrew Stelmach\\Desktop\\screenshot\\out.png')
os.system("taskkill /im chrome.exe /f")

img = cv2.imread('C:\\Users\\Andrew Stelmach\\Desktop\\screenshot\\out.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
black = np.array([0, 0, 0])
mask1 = cv2.inRange(hsv, black, black)

offset = 0

test = mask1 > 0
for x in test:
    if x.any():
        break
    else:
        offset += 1

im = Image.open('C:\\Users\\Andrew Stelmach\\Desktop\\screenshot\\out.png')
width, height = im.size
im.crop((0, offset, width, height)).save('C:\\Users\\Andrew Stelmach\\Desktop\\screenshot\\out.png')


# do the opposite to remove white from bottom

# u can optimize this
def loop(mask):
    for q in mask:
        for n in q:
            if n != 0:
                return True
    return False


high, median, low = True, True, True


def analysis(left, top, right, bottom, x):
    global high, median, low
    im = Image.open('C:\\Users\\Andrew Stelmach\\Desktop\\screenshot\\out.png')
    im.crop((left, top, right, bottom)).save('C:\\Users\\Andrew Stelmach\\Desktop\\screenshot\\out.png')
    img = cv2.imread('C:\\Users\\Andrew Stelmach\\Desktop\\screenshot\\out.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 = mask1 + mask2

    if loop(mask1):
        if x == 3:
            high = False
        if x == 2:
            median = False
        if x == 1:
            low = False

    #cv2.imshow("Image", img)
    #cv2.imshow("Mask", mask1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


prev_height = 0
for x in range(3, 0, -1):
    if not high:
        print("not recommended buy")
        break
    temp = Image.open('C:\\Users\\Andrew Stelmach\\Desktop\\screenshot\\out.png')
    width, height = temp.size
    analysis(0, prev_height, width, height // x, x)
    prev_height = height // x
    temp.save('C:\\Users\\Andrew Stelmach\\Desktop\\screenshot\\out.png')

ema = ema50_m > 0 and ema200_m > 0
if ema and high and median and low:
    print("strong buy")

elif ema and high and median:
    print("recommended buy")

elif high and median:
    print("projections for increase in stock price; buy")

else:
    print("Not recommended buy")


