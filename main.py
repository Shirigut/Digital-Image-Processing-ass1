import cv2
import numpy as np
import math
import warnings
from numpy.polynomial import Polynomial
# "/Users/gal_private/Documents/Digital-Image-Processing-ass1//test.jpeg"
num_of_clicks = 0
(x1, y1) = (0,0)
(x2, y2) = (0,0)
clicked_x = 0
clicked_y = 0
img_name = "img"
counter = 0
points = np.zeros((3, 2))
delta = 1

def cubic_lnterpolation(copy_image, delta):
    height, width = copy_image.shape
    defmat = np.zeros((height, width, 3), np.uint8)
    for y in range(0, height):
        for newx in range(0, width):
            src_x = detransform(newx, y, delta)
            new_pixel = get_cubic_pixel(src_x, y, copy_image, height, width)

            defmat[y][newx] = new_pixel

    ret = cv2.rectangle(defmat, (x1, y1), (x2, y2),  color=color, thickness=thickness)
    cv2.imshow("new image", ret)
    cv2.waitKey(0)


def get_cubic_pixel(src_x, y, copy_image, height, width):
    new_pixel = copy_image[y][math.floor(src_x)]
    dx, dy = abs(round(src_x) - src_x), abs(round(y) - y)

    sumGrayScaler = 0
    # sumB, sumG, sumR = 0, 0, 0

    # print(str(new_pixel) + "old_pixel")
    if round(src_x) + 3 < width and round(y) + 3 < height:
        for i in range(-1, 3):
            for j in range(-1, 3):
                cax = cubicEquationSolver(j + dx, -0.5)
                cay = cubicEquationSolver(i + dy, -0.5)
                sumGrayScaler = sumGrayScaler + copy_image[round(y) + i][round(src_x) + j] * cax * cay

    if sumGrayScaler > 255:
        sumGrayScaler = 255
    elif sumGrayScaler < 0:
        sumR = 0

    new_pixel = (sumGrayScaler)
    return new_pixel


def cubicEquationSolver(d, a):
    d = abs(d)
    if 0.0 <= d <= 1.0:
        score = (a + 2.0) * pow(d, 3.0) - ((a + 3.0) * pow(d, 2.0)) + 1.0
        return score

    elif 1.0 < d <= 2.0:
        score = a * pow(d, 3.0) - 5.0 * a * pow(d, 2.0) + 8.0 * a * d - 4.0 * a
        return score

    else:
        return 0.0

def detransform(newx, y, delta):
    topleft_x, topleft_y = int(min(x1, x2)), int(min(y1, y2))
    btmright_x, btmright_y = int(max(x1, x2)), int(max(y1, y2))

    midx = int((topleft_x + btmright_x) / 2)
    x = newx

    if newx > topleft_x and newx < btmright_x and y < btmright_y and y > topleft_y:
        ex = get_elipse_x(y, delta)

        if newx < ex:
            x = ((newx - topleft_x) * (midx - topleft_x)) / (ex - topleft_x) + topleft_x
        elif newx > ex:
            x = ((btmright_x - midx) * (newx - ex)) / (btmright_x - ex) + midx
    return x


def get_elipse_x(y, delta):
    # solve equasion
    T = pow(y - elipseEqVals[2][1], 2) / pow(elipseEqVals[1], 2)
    c = pow(elipseEqVals[2][0], 2) - (1 - T) * pow(elipseEqVals[0], 2)
    b = -2 * elipseEqVals[2][0]
    a = 1

    x1 = (-b - math.sqrt(pow(b, 2) - 4 * a * c)) / (2 * a)
    x2 = (-b + math.sqrt(pow(b, 2) - 4 * a * c)) / (2 * a)

    res = x1 if delta == -1 else x2
    return abs(res)

def nn_interpolation(img, y, x):
    y = np.clip(np.floor(y+0.5), 0, img.shape[0]-1).astype(int)
    x = np.clip(np.floor(x+0.5), 0, img.shape[1]-1).astype(int)
    return img[y, x]

def click_event(event, x, y, flags, params):
    global clicked_x
    global clicked_y
    global num_of_clicks
    global x1, y1
    global x2, y2
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print('num of clicks: ', num_of_clicks)
        if num_of_clicks == 0: #click on top left rectangle
            x1 = x
            y1 = y
            points[0][0] = x1
            points[0][1] = y1
        elif num_of_clicks == 1:
            x2 = x
            y2 = y
            points[1][0] = x2
            points[1][1] = y2
        elif num_of_clicks == 2:
            clicked_x = x
            clicked_y = y
            points[2][0] = clicked_x
            points[2][1] = clicked_y
        num_of_clicks += 1

def draw_rect(usrimg):
    cv2.rectangle(img, (x1, y1), (x2, y2),  color=color, thickness=thickness)

# Press the green button in the gutter to run the script.
# --------- Read and present image -------------
# path = input("Hi, pls enter image path:\n")
img = cv2.imread("/Users/gal_private/Documents/Digital-Image-Processing-ass1//test3.jpeg", 0)
copy_image = img.copy()

while True:
    x_mid = (x2 + x1) / 2
    x_mid_round = int(x_mid)
    col, row = img.shape[:2]
    new_image = np.zeros((col, row))


    if num_of_clicks == 3: #the user picked the parabola position
        y_mid = int((y1 + y2) / 2)
        center = (x_mid_round, y_mid)

        radiusX = abs(clicked_x - x_mid_round)
        radiusY = abs(y1 - y_mid)
        axes = (radiusX, radiusY)

        startAngle = 270
        endAngle = 90
        angle = 180 if x_mid_round < clicked_x else 0
        delta = 1 if x_mid_round < clicked_x else -1

        cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (255,255,255), 2)
        elipseEqVals = (radiusX, radiusY, center)

    elif num_of_clicks == 2: #the rectangle points were chosen, the parbola isn't
        thickness = 2
        color = (255, 255, 255)
        img = cv2.rectangle(img, (x1, y1), (x2, y2),  color=color, thickness=thickness)
        cv2.line(img, (x_mid_round, y1), (x_mid_round, y2), color, thickness)

    elif num_of_clicks == 4:
        newmat = cubic_lnterpolation(copy_image, delta)
        # newmat = linear_lnterpolation(copy_image, delta)
        # newmat = deformnn(copy_image, delta)



    for x in range(0,3):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 10, (255, 255, 255), cv2.FILLED)


    cv2.imshow(img_name, img)
    cv2.setWindowProperty(img_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(img_name, click_event)
    cv2.waitKey(1)









