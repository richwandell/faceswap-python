import cv2, dlib
import numpy as np

# poisson blending

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 0.5
FEATHER_AMOUNT = 11
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))
COLOUR_CORRECT_BLUR_FRAC = 0.6
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def correct_colours(im1, im2, landmarks1):
    lep = landmarks1[LEFT_EYE_POINTS]
    rep = landmarks1[RIGHT_EYE_POINTS]
    left_eye_mean = np.mean(lep, axis=0)
    right_eye_mean = np.mean(rep, axis=0)
    mean = left_eye_mean - right_eye_mean
    norm = np.linalg.norm(mean)

    blur_amount = COLOUR_CORRECT_BLUR_FRAC * norm

    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    s = im.shape[:2]
    im = np.zeros(s, dtype=np.float64)

    for group in OVERLAY_POINTS:
        to_draw = landmarks[group]

        draw_convex_hull(im,
                         to_draw,
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    a = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0

    im = (a) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    t = points1.T
    mul = t * points2
    U, S, Vt = np.linalg.svd(mul)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T



    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])


def read_landmarks(im):
    im = cv2.resize(im, (int(im.shape[1] * SCALE_FACTOR),
                         int(im.shape[0] * SCALE_FACTOR)))

    rects = detector(im, 1)

    if len(rects) > 1:
            raise Exception
    if len(rects) == 0:
            raise Exception

    ps = predictor(im, rects[0])
    s = np.matrix([[p.x, p.y] for p in ps.parts()])

    return im, s


if __name__ == "__main__":
    brad_face_image = cv2.imread('brad-face.jpg')
    my_face_image = cv2.imread('my-face.jpg')
    im2, landmarks2 = read_landmarks(brad_face_image)

    video_capture = cv2.VideoCapture(0)



    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        try:
            im1, landmarks1 = read_landmarks(frame)
            M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                           landmarks2[ALIGN_POINTS])

            mask1 = get_face_mask(im1, landmarks1)
            mask2 = get_face_mask(im2, landmarks2)

            warped_mask2 = warp_im(mask2, M, im1.shape)

            combined_mask = np.max([mask1, warped_mask2], axis=0)
            # cv2.imwrite('outfile1.jpg', combined_mask)
            warped_im2 = warp_im(im2, M, im1.shape)
            warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

            output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

            cv2.imshow('Video', output_im.astype("uint8"))


        except Exception as e:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()