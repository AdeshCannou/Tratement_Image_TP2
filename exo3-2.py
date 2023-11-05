import cv2
import numpy as np


N = 4 # Nombre de cercles à afficher
# Chargez l'image
image = cv2.imread("Images/four.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# Filtrage Gaussien (optionnel)
image = cv2.GaussianBlur(image, (5, 5), 1.5)
cv2.imshow("Gaussian Blur", image)
cv2.waitKey(0)

# Filtrage de Sobel, calcul de la magnitude et de la direction du gradient
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
direction = np.arctan2(sobel_y, sobel_x)
cv2.imshow("Gradient Magnitude", magnitude)
cv2.waitKey(0)

# Seuillage pour détecter les contours
threshold = 0.27 * magnitude.max()
contour_pixels = (magnitude >= threshold).astype(np.uint8)
cv2.imshow("Edge Image", contour_pixels * 255)
cv2.waitKey(0)

# Initialisation de l'accumulateur
row, col = image.shape
rmin, rmax, delta_r = 1, row, 2
cmin, cmax, delta_c = 1, col, 2
radmin, radmax, delta_rad = 2, 27, 2

accumulator = np.zeros(
    (
        (rmax - rmin) // delta_r + 1,
        (cmax - cmin) // delta_c + 1,
        (radmax - radmin) // delta_rad + 1,
    ),
    dtype=np.uint64,
)

# Angle d'incertitude
beta = np.radians(10)

# Remplissage de l'accumulateur avec prise en compte de la direction du gradient
for y, x in np.argwhere(contour_pixels):
    grad_dir = direction[y, x]

    for rad in range(radmin, radmax + 1, delta_rad):
        rad_idx = (rad - radmin) // delta_rad

        for t in np.arange(grad_dir - beta, grad_dir + beta, np.radians(1)):
            c = int(x - rad * np.cos(t))
            r = int(y - rad * np.sin(t))

            if rmin <= r < rmax and cmin <= c < cmax:
                r_idx = (r - rmin) // delta_r
                c_idx = (c - cmin) // delta_c
                accumulator[r_idx, c_idx, rad_idx] += 1


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

# 7. detection des cercles
def detect_circle(local_rad_min, local_rad_max, level):
    indexes = []
    if level == 0:
        for i in range(1, accumulator.shape[0] - 1):
            for j in range(1, accumulator.shape[1] - 1):
                for k in range(int(local_rad_min), int(local_rad_min)):
                    if accumulator[i, j, k] > 0 and is_local_maximum(i, j, k):
                        indexes.append([i, j, k])
        return sorted(
            indexes,
            key=lambda x: accumulator[x[0], x[1], x[2] - local_rad_min],
            reverse=True,
        )[:N]
    else:
        for i in range(1, accumulator.shape[0] - 1):
            for j in range(1, accumulator.shape[1] - 1):
                for k in range(
                    int((local_rad_max - local_rad_min) / 2), int(local_rad_max)
                ):
                    if accumulator[i, j, k] > 0 and is_local_maximum(i, j, k):
                        indexes.append([i, j, k])
        child = detect_circle(
            local_rad_min, (local_rad_max - local_rad_min) / 2, level - 1
        )
        return sorted(
            child + indexes,
            key=lambda x: accumulator[x[0], x[1], x[2] - local_rad_min],
            reverse=True,
        )[:N]


maxima_indices = detect_circle(1, accumulator.shape[2] - 1, 4)

# Affichage des cercles détectés
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for i in maxima_indices:
    r_idx, c_idx, rad_idx = i
    r = rmin + r_idx * delta_r
    c = cmin + c_idx * delta_c
    rad = radmin + rad_idx * delta_rad
    center = (c, r)
    cv2.circle(output_image, center, rad, (0, 0, 255), 2)
    cv2.circle(output_image, center, 1, (0, 255, 0), 2)

cv2.imshow("Circles Detected", output_image)
cv2.waitKey(0)
