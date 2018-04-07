import cv2
import numpy as np


def getImages(which):
    images = {
        'm': ['samples/m_1_1.jpg', 'samples/m_1_2.jpg', 'samples/m_1_3.jpg'],
        'mountain': ['samples/mountain_1.jpg', 'samples/mountain_2.jpg'],
        'rm': ['samples/rm1.jpg', 'samples/rm2.jpg', 'samples/rm3.jpg']
    }
    for im in images[which]:
        yield cv2.imread(im)


sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

last_image = None
for i, image in enumerate(getImages('rm')):
    if i > 0:
        kp1, des1 = sift.detectAndCompute(last_image, None)
        kp2, des2 = sift.detectAndCompute(image, None)
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                good.append(m)

        pts_src = np.array(list(kp1[m.queryIdx].pt for m in good))
        pts_dst = np.array(list(kp2[m.trainIdx].pt for m in good))

        M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

        w, h = image.shape[:2]
        # image = cv2.warpPerspective(image, M, (w, h))

        result = cv2.warpPerspective(last_image, M, (last_image.shape[1] + image.shape[1], last_image.shape[0]))
        result[0:image.shape[0], 0:image.shape[1]] = image


        # im3 = last_image.

        # im3 = image.copy()
        # tmp = cv2.drawMatchesKnn(last_image, kp1, image, kp2, good, im3, flags=2)

        cv2.imshow("name", result)
        cv2.waitKey(0)



    else:
        last_image = image




# print(im1)