import numpy as np
import cv2
import cupy as cp
import matplotlib.pyplot as plt

# Load the RGB image
image_path = "/content/sample_data/Lenna_test_image.png"   # Change to your image path
image = cv2.imread(image_path)

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get image dimensions
height, width = gray_image.shape

# Transfer grayscale image to GPU
d_gray = cp.asarray(gray_image, dtype=cp.float32)

# Define Sobel Kernels
sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)

# Define CUDA Kernel
sobel_kernel = cp.RawKernel(r"""
extern "C" __global__
void sobel_filter(const float* img, float* edges, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // Sobel Kernels
        float Gx = (-1 * img[(y-1) * width + (x-1)]) + ( 0 * img[(y-1) * width + x]) + ( 1 * img[(y-1) * width + (x+1)])
                 + (-2 * img[y * width + (x-1)])     + ( 0 * img[y * width + x])     + ( 2 * img[y * width + (x+1)])
                 + (-1 * img[(y+1) * width + (x-1)]) + ( 0 * img[(y+1) * width + x]) + ( 1 * img[(y+1) * width + (x+1)]);

        float Gy = (-1 * img[(y-1) * width + (x-1)]) + (-2 * img[(y-1) * width + x]) + (-1 * img[(y-1) * width + (x+1)])
                 + ( 0 * img[y * width + (x-1)])     + ( 0 * img[y * width + x])     + ( 0 * img[y * width + (x+1)])
                 + ( 1 * img[(y+1) * width + (x-1)]) + ( 2 * img[(y+1) * width + x]) + ( 1 * img[(y+1) * width + (x+1)]);

        // Compute magnitude
        edges[y * width + x] = sqrt(Gx * Gx + Gy * Gy);
    }
}
""", "sobel_filter")

# Allocate memory for output on GPU
d_edges = cp.zeros_like(d_gray)

# Define CUDA Grid & Block sizes
block_size = (16, 16)
grid_size = (width // 16 + 1, height // 16 + 1)

# Run Sobel Filter on GPU
%time sobel_kernel(grid_size, block_size, (d_gray, d_edges, width, height))

# Transfer result back to CPU
edges = cp.asnumpy(d_edges)

# Normalize the edges for visualization (0 to 255)
edges = (edges / edges.max()) * 255
edges = edges.astype(np.uint8)

# Show original & edge-detected image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap="gray")
plt.title("Edge Detected Image (Sobel)")
plt.axis("off")

plt.show()

# Save the edge-detected image
cv2.imwrite("sobel_edges.jpg", edges)
