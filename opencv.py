import cv2
import numpy as np

# Function to draw labels on the image
def draw_labels(image):
    cv2.rectangle(image, (10, 10), (40, 40), (255, 0, 255), -1)
    cv2.putText(image, 'White Lane', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  
    cv2.putText(image, 'White Lane', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    cv2.rectangle(image, (10, 50), (40, 80), (0, 255, 0), -1)
    cv2.putText(image, 'Yellow Lane', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(image, 'Yellow Lane', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

# Function to select region of interest
def region_selection(image):
	mask = np.zeros_like(image) 
	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
	rows, cols = image.shape[:2]
	bottom_left = [cols * 0, rows]
	top_left	 = [cols * 0.4, rows * 0.6]
	bottom_right = [cols, rows]
	top_right = [cols * 0.6, rows * 0.6]
	vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image


   
# Read the image
img = cv2.imread("roadd.png")
im2 = img.copy()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Mask for white lanes
white_lower = np.array([0, 0, 180])
white_upper = np.array([180, 25, 255])
white_mask = cv2.inRange(hsv, white_lower, white_upper)

# Mask for yellow lanes
yellow_lower = np.array([15, 110, 80])
yellow_upper = np.array([35, 255, 255])
yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)


# Combine masks
#rblur  = cv2.bilateralFilter(img, 9, 75, 75)
blurrr = cv2.GaussianBlur(img, (5, 5), 10)
cann = cv2.Canny(blurrr, 50, 150)
selection = region_selection(cann)


rblur1  = cv2.bilateralFilter(white_mask, 9, 75, 75)
blurrr1 = cv2.GaussianBlur(rblur1, (5, 5), 10)
cann1 = cv2.Canny(blurrr1, 50, 150)
selection1 = region_selection(cann1)
white_layer = cv2.bitwise_and(selection1, selection)

rblur2 = cv2.bilateralFilter(yellow_mask, 9, 75, 75)
blurrr2 = cv2.GaussianBlur(rblur2, (5, 5), 10)
cann2 = cv2.Canny(blurrr2, 100, 250)
selection2 = region_selection(cann2)
yellow_layer = cv2.bitwise_and(selection2, selection)

# Draw the lanes
col1 = np.zeros((selection1.shape[0], selection1.shape[1], 3), dtype=np.uint8)
col1[selection1 == 255] = (255, 0, 255)

col2 = np.zeros((selection2.shape[0], selection2.shape[1], 3), dtype=np.uint8)
col2[selection2 == 255] = (0, 255, 0)

# Combine the lanes with the original image
final = cv2.addWeighted(img, 0.8, col1, 1, 0)
final_again = cv2.addWeighted(final, 1, col2, 1, 0)

# Display the image
cv2.imshow('prefinal', draw_labels(final_again))
cv2.waitKey(0)
cv2.destroyAllWindows()