import cv2
import numpy as np

# "/Users/gal_private/Documents/DigitalImage/Ass1/test.jpeg"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # --------- Read and present image -------------
    path = input("Hi, pls enter image path:\n")
    img = cv2.imread(path, 0)
    cv2.imshow('graycsale image', img)
    cv2.waitKey(1)

    # ----------------------------------------------
    # --------- Let the user draw rectangle with medial line--------

    x1 = input("Enter x value of the upper left corner of the rectangle: (x1, _)\n")
    x1 = int(x1)
    y1 = input("Enter y value of the upper left corner of the rectangle: (_, y1)\n")
    y1 = int(y1)
    x2 = input("Enter x value of the bottom right corner of the rectangle: (x2,_)\n")
    x2 = int(x2)
    y2 = input("Enter y value of the bottom right corner of the rectangle: (_,y2)\n")
    y2 = int(y2)

    thickness = 2
    color = (255, 255, 255)
    img = cv2.rectangle(img, (x1, y1), (x2, y2),  color=color, thickness=thickness)

    x_mid = (x2 + x1) / 2
    x_mid_round = int(x_mid)
    print(x_mid_round)
    cv2.line(img, (x_mid_round, y1), (x_mid_round, y2), color, thickness)
    cv2.imshow('graycsale image', img)
    cv2.waitKey(1)
    # ----------------------------------------------



    input("next phase")

    col, row = img.shape[:2]
    new_image = np.zeros((col, row)) ##image.dtype
    for y in range(y1, y2):
        parab_x = parabola(y) ##TODO: parabola
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
    cv2.imshow("Assignment1", new_image)
    cv2.waitKey(0)


    for j in range(row):
        for i in range(col):
            parab_x = parabola(i)
            left_to_parab = parab_x - x1
            right_to_parab = parab_x - x_mid_round
            if x1 <= i <= x2 and y1 <= j <= y2: ##make sure it's correct
                if j <= parab_x:
                    new_image[i, j] = pixel_val(img, y, x) ##TODO: pixel_val
                    x = (((j - x1) / left_to_parab) * x_mid) + x1
                    y = i
                    new_image[i, j] = pixel_val(img, y, x)
                else:
                    x = ((((-j + x1) / (x_mid - right_to_parab)) - 1) * x_mid * (-1)) + x_mid_round
                    y = i
                    new_image[i, j] = pixel_val(img, y, x)
            else:
                new_image[i, j] = img[i, j]
        cv2.imshow("Assignment1", new_image)
        cv2.waitKey(0)







