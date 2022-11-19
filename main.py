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

x_roi, y_roi, w_roi, h_roi = 0, 0, 0, 0
x_movement = None
a = -1/2
epsilon = 1e-6
# radiusX = 0
# radiusY = 0

def C_a(x):
    absx = np.abs(x)
    a = -0.5*absx**3 + 2.5*absx**2 - 4*absx + 2
    b = 1.5*absx**3 - 2.5*absx**2 + 1
    ret_val = np.zeros(x.shape)
    ret_val[absx < 2] = a[absx < 2]
    ret_val[absx < 1] = b[absx < 1]
    return ret_val


def nn_interpolation(img, y, x):
    y = np.clip(np.floor(y+0.5), 0, img.shape[0]-1).astype(int)
    x = np.clip(np.floor(x+0.5), 0, img.shape[1]-1).astype(int)
    return img[y, x]


def cubic_interpolation(img, y, x):
    dx = x - np.floor(x)
    dy = y - np.floor(y)
    x_neighbours = [1+dx, dx, 1-dx, 2-dx]
    y_neighbours = [1+dy, dy, 1-dy, 2-dy]
    y = y.astype(int)
    x = x.astype(int)

    new_val = 0
    for i in range(-1, 3):
        for j in range(-1, 3):
            new_val += img[np.clip(y+i, 0, img.shape[0]-1), np.clip(x+j, 0, img.shape[1]-1)] * \
                       C_a(x_neighbours[j+1]) * C_a(y_neighbours[i+1])

    return new_val


def bilinear_interpolation(img, y, x):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # limit the value of x0,x1,y0,y1 to be inside the image boundaries
    # takes care for points at the boundaries
    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    return img[y0, x0] * (x1 - x) * (y1 - y) + img[y1, x0] * (x1 - x) * (y - y0) + img[y0, x1] * (x - x0) * (y1 - y) + \
           img[y1, x1] * (x - x0) * (y - y0)


def get_parabola_points(minima, x, w, y1, y2):
    # clamp minima to rectangle borders
    minima = (max(min(minima[0], x + w), x), minima[1])
    b = minima[0] - (x + w // 2)

    y = (y2 - y1) // 2
    a = -b / (y ** 2)
    ys = np.arange(-y, y, 1)
    xs = a * ys ** 2 - b
    # shift back according to minima
    ys += y1 + (y2 - y1) // 2  # minima[1]
    xs += minima[0] + b
    points = [(int(xs[i]), int(ys[i])) for i in range(len(ys))]
    return np.array(points)


def get_parabolic_movement(point, x, w, y1, y2):
    """
    create the coefficients of the maximum parabola x = ay^2 + b.
    w.r.t to (x_roi, y_roi + h_roi//2) as the origin point of the coordinate system
    """
    point = (np.clip(point[0], x, x + w), point[1])
    b = point[0] - (x + w // 2)

    y = (y2 - y1) // 2
    a = -b / (y ** 2)

    return [a * y ** 2 + b for y in range(-h_roi // 2, h_roi // 2)]


def get_parabolic_transform(x, y, a, b, origin_point_y):
    return a * ((y - origin_point_y) ** 2) + b


def get_gaussian_transform(b):
    sd = h_roi//6
    return [b * (np.exp(-0.5 * (y / sd) ** 2)) for y in range(-h_roi//2, h_roi//2)]



def warp_image(image, interpolation=cubic_interpolation):
    global x_movement
    yy, xx = np.mgrid[y_roi:y_roi + h_roi:1., 0.:w_roi]

    for i, row in enumerate(xx):
        row_movement = np.ceil(x_movement[i]).astype(int)
        xx[i, :w_roi // 2 + row_movement] = xx[i, :w_roi // 2 + row_movement] * (w_roi // 2) / (
                w_roi // 2 + x_movement[i])
        xx[i, w_roi // 2 + row_movement:] = w_roi // 2 + (
                xx[i, w_roi // 2 + row_movement:] - (w_roi // 2 + x_movement[i])) * (w_roi // 2) / (
                                                    w_roi // 2 - x_movement[i])

    xx = xx + x_roi

    img_warpped = np.zeros((*xx.shape, 3))

    _, _, channels = image.shape
    for layer in range(channels):
        current_layer = interpolation(image[:, :, layer], yy, xx)
        img_warpped[:, :, layer] = current_layer

    image[y_roi: y_roi + h_roi, x_roi:x_roi + w_roi, :] = img_warpped
    cv2.imshow(interpolation.__name__, image)


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

# function that returns all y values of the ellipse by x in array.

def get_ellipse_arr():
    global radiusX, radiusY
    a = -radiusX / pow(radiusY, 2)
    b = radiusX
    return [a * pow(x, 2) + b for x in range(-math.floor(radiusY), math.floor(radiusY))]


def parabola(y):
    parab_pts = get_ellipse_arr()
    first = parab_pts[0]
    if y1 <= y <= y2:
        return parab_pts[y-first]
    return -1


def transform():
    for y in range(y1, y2):
        parab_x = int(get_ellipse_arr()[y])
        left_to_parab = parab_x - x1
        right_to_parab = parab_x - x_mid_round
        for x in range(x1, x2):
            if x <= x_mid_round:
                relative = x-x1
                scale = relative / x_mid
                delta = round(left_to_parab*scale)
                new_image[y-col][x1+delta-row] = img[y][x]
            else:
                relative = x - x_mid_round
                scale = 1 - (relative / x_mid)
                delta = round((x_mid - right_to_parab) * scale)
                new_image[y-col][x2-delta-row] = img[y][x]
    # cv2.imshow(img_name, new_image)
    # cv2.waitKey(0)
    print("transform end")


def op_transform():
    print("inside op")
    for j in range(row):
        parab_x = parabola(j)
        for i in range(col):
            left_to_parab = parab_x - x1
            right_to_parab = parab_x - x_mid_round
            if x1 <= i <= x2 and y1 <= j <= y2: ##make sure it's correct
                if j <= parab_x:
                    x = (((j - x1) / left_to_parab) * x_mid) + x1
                    y = i
                    new_image[i, j] = nn_interpolation(img, y, x)
                else:
                    x = ((((-j + x1) / (x_mid - right_to_parab)) - 1) * x_mid * (-1)) + x_mid_round
                    y = i
                    new_image[i, j] = nn_interpolation(img, y, x)
            else:
                new_image[i, j] = img[i, j]
    cv2.imshow(img_name, new_image)
    cv2.waitKey(0)


# Press the green button in the gutter to run the script.
# --------- Read and present image -------------
# path = input("Hi, pls enter image path:\n")
img = cv2.imread("/Users/gal_private/Documents/Digital-Image-Processing-ass1//test3.jpeg", 0)

while True:
    x_mid = (x2 + x1) / 2
    x_mid_round = int(x_mid)
    col, row = img.shape[:2]
    new_image = np.zeros((col, row)) ##image.dtype



    if num_of_clicks == 3: #the user picked the parabola position
        y_mid = int((y1 + y2) / 2)
        center = (y_mid, y_mid)

        radiusX = abs(clicked_x - x_mid_round)
        radiusY = abs(y1 - y_mid)
        axes = (radiusX, radiusY)

        startAngle = 270
        endAngle = 90
        angle = 180 if x_mid_round < clicked_x else 0
        delta = 1 if x_mid_round < clicked_x else -1

        cv2.ellipse(img, center, axes, angle, startAngle, endAngle, (255,255,255), 2)
        # img = cv2.circle(img, (clicked_x, clicked_y), 2, (255, 255, 255), -1)
        # img = cv2.circle(img, (x_mid_round, y1), 2, (255, 255, 255), -1)
        # img = cv2.circle(img, (x_mid_round, y2), 2, (255, 255, 255), -1)
        # pts = np.array([[x_mid_round, y1],
        #                 [clicked_x, clicked_y],
        #                 [x_mid_round, y2]], np.int32)
        #
        # # side parabola coeffs
        # # coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2)
        # # poly = np.poly1d(coeffs)
        # poly = Polynomial.fit(pts[:, 1], pts[:, 0], 2)
        # # poly = Polynomial(coeffs)
        #
        # yarr = np.arange(y1, y2)
        # xarr = poly(yarr)
        # parab_pts = np.array([xarr, yarr], dtype=np.int32).T
        # cv2.polylines(img, [parab_pts], False, (255, 255, 255), 2)
        # b = np.clip(x - (x_roi + w_roi // 2), -w_roi//2+1, w_roi//2-1)
        # x_movement = get_gaussian_transform(b)
        #
        # parab_pts = [(int(x_movement[i] + x_roi + w_roi/2), i+y_roi) for i in range(0, h_roi)]
        # v2.polylines(img, [np.array(parab_pts)], False, (255, 255, 255), 2)


    elif num_of_clicks == 2: #the rectangle points were chosen, the parbola isn't
        thickness = 2
        color = (255, 255, 255)
        img = cv2.rectangle(img, (x1, y1), (x2, y2),  color=color, thickness=thickness)

        cv2.line(img, (x_mid_round, y1), (x_mid_round, y2), color, thickness)

    elif num_of_clicks == 4:
        transform()
        op_transform()

    for x in range(0,3):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 10, (255, 255, 255), cv2.FILLED)

    # with warnings.catch_warnings():
    #     warnings.filterwarnings('error')
    #     try:
    #         coefficients = np.polyfit([1], [2], 2)
    #     except np.RankWarning:
    #         print("")

    cv2.imshow(img_name, img)
    cv2.setWindowProperty(img_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(img_name, click_event)
    cv2.waitKey(1)









