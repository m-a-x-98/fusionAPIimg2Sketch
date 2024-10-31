import cv2
import numpy as np
import matplotlib.image as mpimg

def loadImg():
    lines = mpimg.imread(r"C:\Users\maxfo\pythonProjects\fusionAPIimg2Sketch\image.jpg")
    return cv2.Canny(np.uint8(lines),80,255)

def thinning(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    thinned = cv2.ximgproc.thinning(binary_image)
    return thinned

def detectLines():
    thinnedFig = thinning(loadImg())

    for i in range(1, len(thinnedFig)-1):
        for j in range(1, len(thinnedFig[0])-1):
            sum = thinnedFig[i-1][j-1] + thinnedFig[i][j-1] + thinnedFig[i+1][j-1] +\
                    thinnedFig[i-1][j] + thinnedFig[i][j] + thinnedFig[i+1][j] +\
                    thinnedFig[i-1][j+1] + thinnedFig[i][j+1] + thinnedFig[i+1][j+1]
            if sum > 255*3:
                thinnedFig[i][j] = 0

    img = np.uint8(thinnedFig)

    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(img)[0]
    return lines

lines = detectLines()
with open("out.txt", "w") as outFile:
    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = map(int, line[0]) 
            outFile.write(f"{x0} {y0} {x1} {y1} \n")
