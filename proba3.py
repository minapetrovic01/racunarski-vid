import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread("C:\\Users\\minap\\ELFAK\\vid\\slikeLV3\\1.JPG")
img2 = cv2.imread("C:\\Users\\minap\\ELFAK\\vid\\slikeLV3\\2.JPG")
img3 = cv2.imread("C:\\Users\\minap\\ELFAK\\vid\\slikeLV3\\3.JPG")

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Create SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
keypoints3, descriptors3 = sift.detectAndCompute(gray3, None)

# Match keypoints between images
bf = cv2.BFMatcher()
matches12 = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe's ratio test
good_matches12 = []

for m, n in matches12:
    if m.distance < 0.75 * n.distance:
        good_matches12.append(m)

# Extract corresponding points for good matches
src_pts12 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches12]).reshape(-1, 1, 2)
dst_pts12 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches12]).reshape(-1, 1, 2)

# Find homography matrix
H12, _ = cv2.findHomography(src_pts12, dst_pts12, cv2.RANSAC, 5.0)

# Warp images
result12 = cv2.warpPerspective(img1, H12, (img2.shape[1] + img1.shape[1], img2.shape[0]))

# Concatenate the first two images
concatenated_result = np.concatenate((result12, img2), axis=1)

# Now, match the concatenated result with the third image
gray_concatenated_result = cv2.cvtColor(concatenated_result, cv2.COLOR_BGR2GRAY)
keypoints_concatenated_result, descriptors_concatenated_result = sift.detectAndCompute(gray_concatenated_result, None)
matches23 = bf.knnMatch(descriptors_concatenated_result, descriptors3, k=2)

# Apply Lowe's ratio test
good_matches23 = []

for m, n in matches23:
    if m.distance < 0.75 * n.distance:
        good_matches23.append(m)

# Extract corresponding points for good matches
src_pts23 = np.float32([keypoints_concatenated_result[m.queryIdx].pt for m in good_matches23]).reshape(-1, 1, 2)
dst_pts23 = np.float32([keypoints3[m.trainIdx].pt for m in good_matches23]).reshape(-1, 1, 2)

# Find homography matrix for the third image
H23, _ = cv2.findHomography(src_pts23, dst_pts23, cv2.RANSAC, 5.0)

# Warp the third image
result23 = cv2.warpPerspective(img3, H23, (concatenated_result.shape[1] + img3.shape[1], concatenated_result.shape[0]))

# Combine images horizontally in the order 1-2-3
result = np.concatenate((concatenated_result, result23), axis=1)

# Display result
cv2.imshow('Panoramic Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(result)
plt.title('Panoramic Image')
plt.show()
