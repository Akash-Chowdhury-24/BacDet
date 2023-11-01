import cv2
import numpy as np

image = cv2.imread("img/1240.png")
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

can = cv2.Canny(img, 100, 200)

edge_x = cv2.Sobel(img,-1,1,0)
edge_y = cv2.Sobel(img,-1,0,1)
edge = cv2.addWeighted(edge_x,0.5,edge_y,0.5,0)
edit = cv2.addWeighted(edge,0.5,can,0.5,0)

a, b = cv2.threshold(edit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
c, h = cv2.findContours(b,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_NONE)
g = cv2.drawContours(edit, c, -1, (255,69,0), thickness=1)
cv2.imwrite('1240.png', g)
cv2.waitKey(0)
