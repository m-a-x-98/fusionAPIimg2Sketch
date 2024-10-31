import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.image as mpimg

lines = mpimg.imread("image.jpg")
cannyImage = cv2.Canny(np.uint8(lines),80,255)
plt.figure("Canny Image")
#plt.imshow(cannyImage)
#plt.show()

def thinning(image):
    # Convert the image to a binary format (assuming the white pixels represent lines)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Use OpenCV's thinning algorithm
    thinned = cv2.ximgproc.thinning(binary_image)
    
    return thinned

def stipple(thinned_image, n):
    # Create a new image for the dotted effect
    dotted_image = np.zeros_like(thinned_image)
    
    # Iterate over the thinned image and only keep every n-th pixel on the lines
    for i in range(thinned_image.shape[0]):
        for j in range(thinned_image.shape[1]):
            if thinned_image[i, j] == 255:  # This is a white pixel
                if (i + j) % n == 0:  # Apply stippling based on spacing n
                    dotted_image[i, j] = 255  # Keep the pixel white
                else:
                    dotted_image[i, j] = 0  # Turn others black
    
    return dotted_image

thinnedFig = thinning(cannyImage)
plt.imshow(thinnedFig)
#plt.show()

stippleFig = stipple(thinnedFig, 5)
plt.imshow(stippleFig)
#plt.show()

for i in range(1, len(thinnedFig)-1):
    for j in range(1, len(thinnedFig[0])-1):
        sum = thinnedFig[i-1][j-1] + thinnedFig[i][j-1] + thinnedFig[i+1][j-1] +\
                thinnedFig[i-1][j] + thinnedFig[i][j] + thinnedFig[i+1][j] +\
                thinnedFig[i-1][j+1] + thinnedFig[i][j+1] + thinnedFig[i+1][j+1]
        if sum > 255*3:
            thinnedFig[i][j] = 0

print("Done thinning")

img = np.uint8(thinnedFig)

lsd = cv2.createLineSegmentDetector()
lines = lsd.detect(img)[0]

output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Draw the detected lines on the output image
if lines is not None:
    for line in lines:
        x0, y0, x1, y1 = map(int, line[0])  # Get line coordinates
        cv2.line(output_img, (x0, y0), (x1, y1), (0, 0, 255), 1)

# Display the result
plt.figure(figsize=(10, 5))
plt.imshow(output_img, cmap='gray')
plt.title("Detected Lines using LSD")
plt.axis("off")
plt.show()
