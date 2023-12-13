
import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_image(img, color_map=None):
    plt.imshow(img, cmap=color_map)
    plt.show()
    
def save_image(img, img_path):
    cv2.imwrite(img_path, img)
    
def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded

if __name__ == "__main__":
    img = cv2.imread(".\coins\coins.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plot_image(img)

    img = cv2.GaussianBlur(img, (5, 5), 10)

    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, coins_mask = cv2.threshold(grayscale_img, 180, 255, cv2.THRESH_BINARY_INV)
    plot_image(coins_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    coins_mask = cv2.morphologyEx(coins_mask, cv2.MORPH_CLOSE, kernel)
    
    plot_image(coins_mask)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    _, coin_marker = cv2.threshold(hsv_img[..., 1], 75, 255, cv2.THRESH_BINARY)

    plot_image(coin_marker)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    coins_marker = cv2.morphologyEx(coin_marker, cv2.MORPH_CLOSE, kernel)
    coins_marker = cv2.morphologyEx(coins_marker, cv2.MORPH_OPEN, kernel)
    
    reconstructed_img = morphological_reconstruction(coins_marker, coins_mask)
    plot_image(reconstructed_img)