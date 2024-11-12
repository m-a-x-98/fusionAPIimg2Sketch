import cv2
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
import numpy as np
from tqdm import tqdm
from skimage import img_as_ubyte
from scipy.ndimage import convolve


#img = cv2.imread("imgCog.webp",0)
img = cv2.imread("imgFlower.webp",0)



#img = cv2.Canny(np.uint8(img),100,255)
img = cv2.bitwise_not(img) < 128
#img = skeletonize(img).astype(np.uint8)*255
#cv2.imwrite('canny_edges.jpg', img)

kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])

neighbor_sums = convolve(img.astype(int), kernel, mode='constant', cval=0)
mask = neighbor_sums > 4
img = np.where(mask, 0, img)


def remove(ls, img):
    kernel = np.array([
        [ls[0], ls[1], ls[2]],
        [ls[3], ls[4], ls[5]],
        [ls[6], ls[7], ls[8]]
    ])

    neighbor_sums = convolve(img.astype(int), kernel, mode='constant', cval=0)
    mask = neighbor_sums > 1
    return np.where(mask, 0, img)

def remove2(ls, img):
    kernel = np.array([
        [ls[0], ls[1], ls[2]],
        [ls[3], ls[4], ls[5]],
        [ls[6], ls[7], ls[8]]
    ])

    neighbor_sums = convolve(img.astype(int), kernel, mode='constant', cval=0)
    mask = neighbor_sums < 2
    return np.where(mask, 0, img)

img = remove([0,0,0,  1,0,0,  0,1,0], img)
img = remove([0,0,0,  0,0,1,  0,1,0], img)
img = remove([0,1,0,  1,0,0,  0,0,0], img)
img = remove([0,1,0,  0,0,1,  0,0,0], img)

# --- optional ----
#for i in range(30):
#    img = remove2([1,1,1,  1,0,1,  1,1,1], img)



'''for n in tqdm(range(1, len(img)-1)):
    for k in range(1, len(img[0])-1):
        if int(img[n-1][k-1]) + int(img[n-1][k]) + int(img[n-1][k+1])\
            + int(img[n][k-1]) + int(img[n][k+1])\
            + int(img[n+1][k-1]) + int(img[n+1][k]) + int(img[n+1][k+1]) > 3:
            img[n][k] = 0'''

# instead of skeletonize - can check if a pixel has pixels to the left and down or left and up or right...

# kan do this also instead of canny, check all the pixels which DONT have pixels on all four sides - add them to a new image 
# maybe use numpy vectorization on the last on? 



plt.imshow(img, cmap=plt.cm.gray)
plt.show()

imgZero = np.zeros_like(img)

points = list(np.argwhere(img == 1))

final = []

def addToFinal(point, n, ls):
    ls.append( list(points.pop(n)) )
    return point

def getBlob():
    blob = []
    current = addToFinal(points[0], 0, blob)
    #current = points[0]
    last = [None, None]
    for i in range(len(points)-1, -1, -1):
        if abs(points[i][0] - current[0]) <= 1 and abs(points[i][1] - current[1]) <= 1: 
            last = points[i]
            break
    first = 2
    while True:
        for n, i in enumerate(points):
            if abs(i[0] - current[0]) <= 1 and abs(i[1] - current[1]) <= 1:
                current = addToFinal(i, n, blob)
                break
        else:
            break 
        if current[0] == last[0] and current[1] == last[1]:
            #print(current, last)
            break
    return blob

print(len(points))

while len(points) > 0:
    final.append(getBlob())
    print(len(points))

#for i in ls:
#    imgZero[i[0]][i[1]] = 1

c = 0
for i in final:
    if len(i) > 5:
        for j in i:
            imgZero[j[0]][j[1]] = 1
        c+=1 
print(c)

plt.imshow(imgZero, cmap=plt.cm.gray)
plt.show()