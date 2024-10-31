import cv2
import numpy as np

# Load the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Define a kernel for edge detection (e.g., Sobel or Laplacian)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel X kernel for horizontal edges
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sobel Y kernel for vertical edges

# Apply the convolution
edges_x = cv2.filter2D(image, -1, sobel_x)
edges_y = cv2.filter2D(image, -1, sobel_y)

# Combine the edges
edges = cv2.addWeighted(edges_x, 1, edges_y, 1, 0)

result = np.where(edges > 70, 255, 0).astype(np.uint8)

# Save or display the result
cv2.imshow('Edges', result)
cv2.waitKey(0)
cv2.destroyAllWindows()