import cv2
import numpy as np
import matplotlib.pyplot as plt


# Učitavanje slika
img1 = cv2.imread("C:\\Users\minap\\ELFAK\\vid\\slikeLV3\\1.JPG")
img2 = cv2.imread("C:\\Users\\minap\\ELFAK\\vid\\slikeLV3\\2.JPG")
img3 = cv2.imread("C:\\Users\\minap\\ELFAK\\vid\slikeLV3\\3.JPG")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)


# Kreiranje SIFT detektora
sift = cv2.SIFT_create()

# Pronalaženje ključnih tačaka i deskriptora za svaku sliku
# keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
# keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
# keypoints3, descriptors3 = sift.detectAndCompute(img3, None)

keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
keypoints3, descriptors3 = sift.detectAndCompute(gray3, None)


# Pronalaženje uparenih tačaka između slika 1 i 2, te slika 2 i 3
bf = cv2.BFMatcher()
matches12 = bf.knnMatch(descriptors1, descriptors2, k=2)
matches23 = bf.knnMatch(descriptors2, descriptors3, k=2)

# Primena Lowe-ovog testa za odabir dobrih upara
good_matches12 = []
good_matches23 = []

for m, n in matches12:
    if m.distance < 0.75 * n.distance:
        good_matches12.append(m)

for m, n in matches23:
    if m.distance < 0.75 * n.distance:
        good_matches23.append(m)

# Izdvajanje koordinata ključnih tačaka za dobre uparene tačke
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches12]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches12]).reshape(-1, 1, 2)

# Primena RANSAC algoritma za pronalaženje homografske matrice
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

# Perspektivna transformacija slika na osnovu homografske matrice
result12 = cv2.warpPerspective(img1, H, (img2.shape[1] + img1.shape[1], img2.shape[0]))

# Ponovno pronalaženje koordinata ključnih tačaka za nove slike
src_pts = np.float32([keypoints2[m.queryIdx].pt for m in good_matches23]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints3[m.trainIdx].pt for m in good_matches23]).reshape(-1, 1, 2)

# Primena RANSAC algoritma za pronalaženje nove homografske matrice
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

# Perspektivna transformacija slika na osnovu nove homografske matrice
# result23 = cv2.warpPerspective(result12, H, (result12.shape[1] + img3.shape[1], result12.shape[0]))
result23 = cv2.warpPerspective(result12, H, (result12.shape[1] + img3.shape[1], img3.shape[0]))

# # Spajanje rezultirajuće slike
# result = np.maximum(result23, img3)
# Učitavanje rezultirajuće slike
# result23 = cv2.warpPerspective(result12, H, (result12.shape[1] + img3.shape[1], result12.shape[0]))

# Prilagođavanje dimenzija img3 da odgovaraju dimenzijama result23
img3_resized = cv2.resize(img3, (result12.shape[1], result12.shape[0]))

# Spajanje slika
result = np.maximum(result12, img3_resized)

# Prikaz rezultata
cv2.imshow('Panoramska slika', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(result)
# plt.title('Panoramska slika')
# plt.show()