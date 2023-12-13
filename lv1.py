import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft(img):
    img_fft = np.fft.fft2(img)  # pretvaranje slike u frekventni domen (FFT - Fast Fourier Transform), fft2 je jer je u 2 dimenzije
    img_fft = np.fft.fftshift(img_fft)  # pomeranje koordinatnog pocetka u centar slike
    return img_fft

def inverse_fft(fft_img):
    img= np.abs(np.fft.ifft2(fft_img)) 
    return img

def get_magnitude_spectrum(img_fft):
    img_fft_mag = np.log(np.abs(img_fft)) 
    return img_fft_mag
 
def read_image(img_path, color=cv2.COLOR_BGR2GRAY):
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, color)
 
def plot_image(img, color_map=None):
    plt.imshow(img, cmap=color_map)
    plt.show()
    
def remove_noisy_pixels(fft_img, noisy_indices):
    for xy in noisy_indices:
        fft_img[xy] = 0
    return fft_img

def find_noisy_pixels(img_fft):
    height, width = img_fft.shape
    center_x, center_y = np.array(img_fft.shape) // 2
    radius = 7

    selected_indices = []

    for y in range(height):
        for x in range(width):
            if round(img_fft[x,y],2) >= 14.41:
                if not (center_x - radius <= x <= center_x + radius and center_y - radius <= y <= center_y + radius):
                    selected_indices.append((x, y))
    return selected_indices
    

def get_hardcoded_noisy_pixels():
    return [(231, 251), (246, 261), (266, 251), (281, 261)]

def save_image(img, img_path):
    cv2.imwrite(img_path, img)
 
if __name__ == "__main__":
    input_image = read_image("./MaterijalLV1/slika_2.png")
    plot_image(input_image, color_map="grey")
 
    fft_img = fft(input_image)
    plot_image(get_magnitude_spectrum(fft_img))
 
    #noisy_indices = get_hardcoded_noisy_pixels()
    noisy_indices = find_noisy_pixels(get_magnitude_spectrum(fft_img))
    fft_img = remove_noisy_pixels(fft_img, noisy_indices)
    plot_image(get_magnitude_spectrum(fft_img))
 
    filtered_img = inverse_fft(fft_img)
    plot_image(filtered_img, color_map="grey")

    save_image(filtered_img, "./MaterijalLV1/slika_2_filtered.png")