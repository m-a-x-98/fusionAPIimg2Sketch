import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from tqdm import tqdm

img = cv2.imread("img2.jpg",0)
#sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#img = cv2.filter2D(img, -1, sharpen_kernel)

#Nyttige ting 
#https://stackoverflow.com/questions/57080937/increase-accuracy-of-detecting-lines-using-opencv

img = cv2.Canny(np.uint8(img),100,255)
img = cv2.bitwise_not(img) < 128
img = skeletonize(img).astype(np.uint8)*255

#plt.imshow(img, cmap=plt.cm.gray)
#plt.show()

lsd = cv2.createLineSegmentDetector(0)
lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines

tol = 5

for i in tqdm(range(len(lines))):
    for j in range(i + 1, len(lines)):
        x01, y01, x11, y11 = lines[i][0]
        x02, y02, x12, y12 = lines[j][0]

        if abs(x02 - x01) + abs(y01 - y02) < tol:
            lines[j] = [[x01, y01, x12, y12]]
            break

        elif abs(x12 - x11) + abs(y11 - y12) < tol:
            lines[j] = [[x02, y02, x11, y11]]
            break

        elif abs(x01 - x12) + abs(y01 - y12) < tol:
            lines[j] = [[x02, y02, x01, y01]]
            break

        elif abs(x11 - x02) + abs(y11 - y02) < tol:
            lines[j] = [[x11, y11, x12, y12]]
            break

tol2 = 5
for i in tqdm(range(len(lines))):
    for j in range(i + 1, len(lines)):
        x01, y01, x11, y11 = lines[i][0]
        x02, y02, x12, y12 = lines[j][0]
        if abs(x02 - x01) + abs(y01 - y02) + abs(x12 - x11) + abs(y11 - y12) < tol2\
            or abs(x01 - x12) + abs(y01 - y12) + abs(x11 - x02) + abs(y11 - y02) < tol2:
            lines[j] = [[-1, -1, -1, -1]]
        
        #må skrive kode for om noen linjer er basically samme som andre bare kortere 
        if abs(x01 - x02) + abs(x11 - x12) < tol: #and (y01 < y02 and y11 < y12):
            lines[j] = [[-1, -1, -1, -1]]
        if abs(y01 - y02) + abs(y11 - y12) < tol: #and (x01 > x02 and x11 > x12):
            lines[j] = [[-1, -1, -1, -1]]

result = []
for i in lines:
    if i[0][0]!=-1 and i[0][1]!=-1 and i[0][2]!=-1 and i[0][3]!=-1:
        result.append(i)

if result is not None:
    for line in result:
        x0, y0, x1, y1 = line[0]
        plt.plot([x0, x1], [y0, y1])
plt.show()