import cv2
import numpy as np


def getImages(which):
    images = {
        'm': ['samples/m_1_1.jpg', 'samples/m_1_2.jpg', 'samples/m_1_3.jpg'],
        'mountain': ['samples/mountain_1.jpg', 'samples/mountain_2.jpg'],
        'rm': ['samples/rm1.jpg', 'samples/rm2.jpg', 'samples/rm3.jpg'],
        'bryce': ['samples/bryce_left_01.png', 'samples/bryce_right_01.png']
    }
    for im in images[which]:
        image = cv2.imread(im)
        yield image


sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

last_image = None
for i, current_image in enumerate(getImages('m')):
    if i > 0:
        kp1, des1 = sift.detectAndCompute(current_image, None)
        kp2, des2 = sift.detectAndCompute(last_image, None)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                good.append(m)

        pts_src = np.array(list(kp1[m.queryIdx].pt for m in good))
        pts_dst = np.array(list(kp2[m.trainIdx].pt for m in good))

        M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
        x = int(M[0][2]) if int(M[0][2]) > 0 else 0
        y = int(M[1][2]) if int(M[1][2]) > 0 else 0

        result = cv2.warpPerspective(
            current_image,
            M,
            (
                last_image.shape[1] + x,
                last_image.shape[0] + y
            )
        )

        result[0:last_image.shape[0], 0:last_image.shape[1]] = last_image


        cv2.imshow("name", result)
        cv2.waitKey(0)
        last_image = result


    else:
        last_image = current_image




# print(im1)