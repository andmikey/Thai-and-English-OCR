import cv2 as cv
import numpy as np
from PIL import Image

img = Image.open(
    "/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TestSet/Journal/Image/200dpi_BW/ce_001sb.bmp"
)

# TODO something funky with CV conversion - saving as all black

cropped = img.crop((748, 1882, 1427, 2059))
arr = np.array(cropped)
arr2 = cv.cvtColor(arr, cv.COLOR_GRAY2RGB)

# Would need some thresholding here if grayscale
contours, hierarchy = cv.findContours(arr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(arr2, contours, -1, (0, 255, 0), 3)

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(arr2, (x, y), (x + w, y + h), (255, 0, 0), 3)
    # Use PIL here to retrieve the rectangle of each letter
    letter_crop = img.crop((x, y, x + w, y + h))

cv.imwrite("out.png", arr)
