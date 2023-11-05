import cv2
import numpy as np

# Chargez l'image four.png (assurez-vous qu'elle est dans le même répertoire)
image = cv2.imread("Images/four.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# 1. Filtrage Gaussien (optionnel)
image = cv2.GaussianBlur(image, (5, 5), 1.5)
cv2.imshow("Gaussian Blur", image)
cv2.waitKey(0)

# 2. Filtrage de Sobel, calcul de la magnitude du gradient dans chaque pixel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
cv2.imshow("Gradient Magnitude", magnitude)
cv2.waitKey(0)

# 3. Seuillage pour détecter les contours
threshold = 0.27 * magnitude.max()  # Vous pouvez ajuster cette valeur
contour_pixels = (magnitude >= threshold).astype(np.uint8)
cv2.imshow("Edge Image", contour_pixels * 255)
cv2.waitKey(0)

# 4. Initialisation de l'accumulateur
row, col = image.shape
rmin, rmax, delta_r = 1, row, 2  # row min et max
cmin, cmax, delta_c = 1, col, 2  # column min et max
radmin, radmax, delta_rad = 2, 27, 2  # rayon min et max

acc_shape = (
    (rmax - rmin) // delta_r + 1,
    (cmax - cmin) // delta_c + 1,
    (radmax - radmin) // delta_rad + 1,
)
accumulator = np.zeros(acc_shape, dtype=int)

# 5. Remplissage de l'accumulateur
contour_indices = np.argwhere(contour_pixels)
for y, x in contour_indices:
    for r_idx, r in enumerate(range(rmin, rmax + 1, delta_r)):
        for c_idx, c in enumerate(range(cmin, cmax + 1, delta_c)):
            rad = int(np.sqrt((x - c) ** 2 + (y - r) ** 2))
            if radmin <= rad <= radmax:
                rad_idx = (rad - radmin) // delta_rad
                accumulator[r_idx, c_idx, rad_idx] += magnitude[y, x]


# 6. Identification des maximas locaux dans l'accumulateur
def is_local_maximum(i, j, k):
    for m in range(i - 1, i + 2):
        for n in range(j - 1, j + 2):
            for o in range(k - 1, k + 2):
                if (m != i or n != j or o != k) and accumulator[m, n, o] >= accumulator[
                    i, j, k
                ]:
                    return False
    return True


local_maxima = np.zeros(accumulator.shape, dtype=bool)
for i in range(1, accumulator.shape[0] - 1):
    for j in range(1, accumulator.shape[1] - 1):
        for k in range(1, accumulator.shape[2] - 1):
            if accumulator[i, j, k] > 0 and is_local_maximum(i, j, k):
                local_maxima[i, j, k] = True

# 7. Sélection des N valeurs les plus grandes
N = 3  # Chosir le nombre de cercle à afficher
maxima_indices = np.argwhere(local_maxima)
maxima_values = accumulator[local_maxima]
sorted_maxima_indices = np.argsort(maxima_values)[::-1][:N]
print(maxima_indices[sorted_maxima_indices])


# Affichage des cercles détectés
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for i in sorted_maxima_indices:
    r_idx, c_idx, rad_idx = maxima_indices[i]
    r = rmin + r_idx * delta_r
    c = cmin + c_idx * delta_c
    rad = radmin + rad_idx * delta_rad
    center = (c, r)
    cv2.circle(output_image, center, rad, (0, 0, 255), 2)
    cv2.circle(output_image, center, 2, (0, 255, 0), 2)

cv2.imshow("Circles Detected", output_image)
cv2.waitKey(0)
