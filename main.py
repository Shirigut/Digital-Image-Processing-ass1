import cv2
import numpy as np
import math

num_of_clicks = 0
(x1, y1) = (0, 0)
(x2, y2) = (0, 0)
clicked_x = 0
clicked_y = 0
img_name = "img"
points = np.zeros((3, 2))
delta = 1


def nn_interpolation(copy_image, delta):
    row, col = copy_image.shape
    new_image = np.zeros((row, col, 3), np.uint8)
    for j in range(0, row):
        for i in range(0, col):
            x = int(op_transform(i, j, delta))
            new_image[j, i] = copy_image[j, x]
    return new_image


def cubic_interpolation(copy_image, delta):
    row, col = copy_image.shape
    new_image = np.zeros((row, col, 3), np.uint8)
    for j in range(0, row):
        for i in range(0, col):
            src_x = op_transform(i, j, delta)
            new_image[j,i] = get_cubic_pixel(src_x, j, copy_image, row, col)
    return new_image


def get_cubic_pixel(src_x, y, copy_image, row, col):
    dx, dy = abs(round(src_x) - src_x), abs(round(y) - y)
    sum = 0
    if round(src_x) + 3 < col and round(y) + 3 < row:
        for i in range(-1, 3):
            for j in range(-1, 3):
                cax = cubic_solver(j + dx, -0.5)
                cay = cubic_solver(i + dy, -0.5)
                sum = sum + copy_image[round(y) + i][round(src_x) + j] * cax * cay
    if sum > 255:
        sum = 255
    elif sum < 0:
        sum = 0
    return sum


def cubic_solver(d, a):
    d = abs(d)
    if 0.0 <= d <= 1.0:
        return (a + 2.0) * pow(d, 3.0) - ((a + 3.0) * pow(d, 2.0)) + 1.0
    elif 1.0 < d <= 2.0:
        return a * pow(d, 3.0) - 5.0 * a * pow(d, 2.0) + 8.0 * a * d - 4.0 * a
    else:
        return 0.0


def op_transform(newx, y, delta):
    x = newx
    if x1 < newx < x2 and y1 < y < y2:
        ex = get_ellipse(y, delta)
        if newx < ex:
            x = ((newx - x1) * (x_mid_round - x1)) / (ex - x1) + x1
        elif newx > ex:
            x = ((x2 - x_mid_round) * (newx - ex)) / (x2 - ex) + x_mid_round
    return x


def linear_interpolation(copy_image, delta):
    height, width = copy_image.shape
    new_image = np.zeros((height, width, 3), np.uint8)
    for j in range(0, height):
        for i in range(0, width):
            src_x = op_transform(i, j, delta)
            new_pixel = get_linear_pixel(src_x, j, copy_image, height, width)
            new_image[j][i] = new_pixel
    return new_image


def get_linear_pixel(src_x, y, copy_image, row, col):
    x1_neighbor1, y1_neighbor1 = math.floor(src_x), math.floor(y)
    x1_neighbor2, y1_neighbor2 = x1_neighbor1, y1_neighbor1 + 1
    x2_neighbor1, y2_neighbor1 = x1_neighbor1 + 1, y1_neighbor1
    x2_neighbor2, y2_neighbor2 = x1_neighbor1 + 1, y1_neighbor1 + 1

    gap_x, gap_y = src_x - x1_neighbor1, y + 0.2 - y1_neighbor1
    new_pixel = copy_image[y1_neighbor1][x1_neighbor1]
    if x2_neighbor2 + 1 < col and y2_neighbor2 + 1 < row:
        sum = (1 - gap_x) * (1 - gap_y) * int(copy_image[y1_neighbor1][x1_neighbor1]) + \
                     (1 - gap_x) * gap_y * int(copy_image[y2_neighbor1][x2_neighbor1]) + \
                     gap_x * (1 - gap_y) * int(copy_image[y1_neighbor2][x1_neighbor2]) + \
                     gap_x * gap_y * int(copy_image[y2_neighbor2][x2_neighbor2])
        new_pixel = sum
    return new_pixel


def get_ellipse(y, delta):
    t = pow(y - ellipse_vals[2][1], 2) / pow(ellipse_vals[1], 2)
    c = pow(ellipse_vals[2][0], 2) - (1 - t) * pow(ellipse_vals[0], 2)
    b = -2 * ellipse_vals[2][0]
    a = 1

    res1 = (-b - math.sqrt(pow(b, 2) - 4 * a * c)) / (2 * a)
    res2 = (-b + math.sqrt(pow(b, 2) - 4 * a * c)) / (2 * a)

    res = res1 if delta == -1 else res2
    return abs(res)


def click_event(event, x, y, flags, params):
    global clicked_x
    global clicked_y
    global num_of_clicks
    global x1, y1
    global x2, y2
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if num_of_clicks == 0:  # click on top left rectangle
            x1 = x
            y1 = y
            points[0][0] = x1
            points[0][1] = y1
            print('click on the image to spot the bottom right corner of the rectangle\n')
        elif num_of_clicks == 1:  # click on bottom right rectangle
            x2 = x
            y2 = y
            points[1][0] = x2
            points[1][1] = y2
            print('click on the image to spot the parabola point\n')
        elif num_of_clicks == 2:  # click on parabola point
            clicked_x = x
            clicked_y = y
            points[2][0] = clicked_x
            points[2][1] = clicked_y
            print("click again to see the interpolation\n")
        num_of_clicks += 1


# --------- Read and present image -------------
path = input("Hi, pls enter image path:\n")
img = cv2.imread(path, 0)
copy_image = img.copy()
print('click on the image to spot the top left corner of the rectangle\n')

while True:
    x_mid = (x2 + x1) / 2
    x_mid_round = int(x_mid)

    if num_of_clicks == 3:  # the user picked the parabola position
        y_mid = int((y1 + y2) / 2)
        center = (x_mid_round, y_mid)

        radiusX = abs(clicked_x - x_mid_round)
        radiusY = abs(y1 - y_mid)
        axes = (radiusX, radiusY)

        startAngle = 270
        endAngle = 90
        angle = 180 if x_mid_round < clicked_x else 0
        delta = 1 if x_mid_round < clicked_x else -1

        cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (255, 255, 255), 2)
        ellipse_vals = (radiusX, radiusY, center)

    elif num_of_clicks == 2:  # the rectangle points were chosen, the parbola isn't
        thickness = 2
        color = (255, 255, 255)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)
        cv2.line(img, (x_mid_round, y1), (x_mid_round, y2), color, thickness)

    elif num_of_clicks == 4:
        cubic_img = cubic_interpolation(copy_image, delta)
        linear_img = linear_interpolation(copy_image, delta)
        nn_img = nn_interpolation(copy_image, delta)
        cv2.imshow("cubic_img", cubic_img)
        cv2.imshow("linear_img", linear_img)
        cv2.imshow("nn_img", nn_img)


    for x in range(0, 3):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 10, (255, 255, 255), cv2.FILLED)

    cv2.imshow(img_name, img)
    cv2.setMouseCallback(img_name, click_event)
    cv2.waitKey(1)
