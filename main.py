import cv2
import numpy as np

# "/Users/gal_private/Documents/Digital-Image-Processing-ass1//test.jpeg"
num_of_clicks = 0
(x1, y1) = (0,0)
(x2, y2) = (0,0)
clicked_x = 0
clicked_y = 0
img_name = "img"
counter = 0
points = np.zeros((3, 2))
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

def parabola(y):
    first = parab_pts[0][1]
    return parab_pts[y-first][0]

def transform():
    col, row = img.shape[:2]
    new_image = np.zeros((col, row)) ##image.dtype
    for y in range(y1, y2):
        parab_x = parabola(y)
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
    cv2.imshow(img_name, new_image)
    cv2.waitKey(0)

# def op_transform():
#     for j in range(row):
#         for i in range(col):
#             parab_x = parabola(i)
#             left_to_parab = parab_x - x1
#             right_to_parab = parab_x - x_mid_round
#             if x1 <= i <= x2 and y1 <= j <= y2: ##make sure it's correct
#                 if j <= parab_x:
#                     new_image[i, j] = pixel_val(img, y, x) ##TODO: pixel_val
#                     x = (((j - x1) / left_to_parab) * x_mid) + x1
#                     y = i
#                     new_image[i, j] = pixel_val(img, y, x)
#                 else:
#                     x = ((((-j + x1) / (x_mid - right_to_parab)) - 1) * x_mid * (-1)) + x_mid_round
#                     y = i
#                     new_image[i, j] = pixel_val(img, y, x)
#             else:
#                 new_image[i, j] = img[i, j]
#         cv2.imshow("Assignment1", new_image)
#         cv2.waitKey(0)


# Press the green button in the gutter to run the script.
# --------- Read and present image -------------
# path = input("Hi, pls enter image path:\n")
img = cv2.imread("/Users/gal_private/Documents/Digital-Image-Processing-ass1//test.jpeg")

while True:
    x_mid = (x2 + x1) / 2
    x_mid_round = int(x_mid)

    if num_of_clicks == 3: #the user picked the parabola position
        img = cv2.circle(img, (clicked_x, clicked_y), 2, (255, 255, 255), -1)
        img = cv2.circle(img, (x_mid_round, y1), 2, (255, 255, 255), -1)
        img = cv2.circle(img, (x_mid_round, y2), 2, (255, 255, 255), -1)
        pts = np.array([[x_mid_round, y1],
                        [clicked_x, clicked_y],
                        [x_mid_round, y2]], np.int32)

        # side parabola coeffs
        coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2)
        poly = np.poly1d(coeffs)

        yarr = np.arange(y1, y2)
        xarr = poly(yarr)
        parab_pts = np.array([xarr, yarr], dtype=np.int32).T
        cv2.polylines(img, [parab_pts], False, (255, 255, 255), 2)
        transform()

    elif num_of_clicks == 2: #the rectangle points were chosen, the parbola isn't
        thickness = 2
        color = (255, 255, 255)
        img = cv2.rectangle(img, (x1, y1), (x2, y2),  color=color, thickness=thickness)

        cv2.line(img, (x_mid_round, y1), (x_mid_round, y2), color, thickness)

    for x in range(0,3):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 10, (255, 255, 255), cv2.FILLED)

    cv2.imshow(img_name, img)
    cv2.setWindowProperty(img_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(img_name, click_event)
    cv2.waitKey(1)









