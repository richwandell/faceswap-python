import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

last_image = None
while True:
    ret, current_image = video_capture.read()
    current_image = cv2.resize(current_image, (int(current_image.shape[1] * .2), int(current_image.shape[0] * .2)))
    if last_image is not None:
        try:
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

            result = cv2.warpPerspective(
                current_image,
                M,
                (
                    last_image.shape[1] + current_image.shape[1],
                    last_image.shape[0] + current_image.shape[0]
                )
            )
            result[0:last_image.shape[0], 0:last_image.shape[1]] = last_image
            edges = cv2.Canny(result, 100, 200)
            indexes = np.where(edges==255)

            result = result[0:indexes[0].max(), 0:indexes[1].max()]

            cv2.imshow("name", result)
            last_image = result
        except:
            last_image = None
            pass


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    else:
        last_image = current_image


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()