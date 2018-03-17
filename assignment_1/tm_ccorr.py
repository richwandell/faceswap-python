import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv

full_image_path = 'Picture1.png'
template_path = 'Picture2.png'


full_image = cv2.imread(full_image_path)
full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
image_template = cv2.imread(template_path)
image_template = cv2.cvtColor(image_template, cv2.COLOR_BGR2RGB)

full_image = np.zeros((9, 9, 3)).astype('uint8')
full_image[:, :, :] = 255
full_image[3:, 2:-1, :] = 0

image_template = np.zeros((3, 3, 3)).astype('uint8')
image_template[:, :, :] = 255
image_template[1:, :2, :] = 0

template_mean = image_template.mean()
image_mean = full_image.mean()

res = cv2.matchTemplate(full_image, image_template, cv2.TM_CCORR)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
h, w = image_template.shape[0:2]

bottom_right = (max_loc[0] + w, max_loc[1] + h)
matched_im = cv2.rectangle(full_image.copy(), max_loc, bottom_right, 255, 2)


plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(full_image, plt.cm.gray)
plt.subplot(2, 2, 2)
plt.imshow(image_template, plt.cm.gray)
plt.subplot(2, 2, 3)
plt.imshow(res, plt.cm.gray)
plt.subplot(2, 2, 4)
plt.imshow(matched_im, plt.cm.gray)
plt.show()


fheight, fwidth = full_image.shape[0:2]
theight, twidth = image_template.shape[0:2]

bestX = 0
bestY = 0
bestC = 0
result = []
for y in range(fheight):
    for x in range(fwidth):
        if x + twidth < fwidth + 1 and y + theight < fheight + 1:
            t = full_image[y:y + theight, x:x + twidth, :]
            m = image_template.astype(float) * t
            correlation = m.sum()
            if len(result) - 1 < y:
                result.append([])
            result[y].append(correlation)
result = np.array(result)

min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result)
h2, w2 = image_template.shape[0:2]

bottom_right2 = (max_loc2[0] + w2, max_loc2[1] + h2)
matched_im2 = cv2.rectangle(full_image.copy(), max_loc2, bottom_right2, 255, 2)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(full_image, plt.cm.gray)
plt.subplot(2, 2, 2)
plt.imshow(image_template, plt.cm.gray)
plt.subplot(2, 2, 3)
plt.imshow(result, plt.cm.gray)
plt.subplot(2, 2, 4)
plt.imshow(matched_im2, plt.cm.gray)
plt.show()



