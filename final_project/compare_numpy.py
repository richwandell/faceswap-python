import numpy as np
import cv2, dlib

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
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

points1 = np.matrix([
    [555., 550.],
    [571., 542.],
    [589., 540.],
    [606., 547.],
    [614., 561.],
    [472., 582.],
    [483., 569.],
    [498., 568.],
    [511., 580.],
    [498., 586.],
    [482., 588.],
    [564., 579.],
    [576., 567.],
    [590., 567.],
    [599., 576.],
    [591., 584.],
    [577., 584.],
    [451., 570.],
    [464., 555.],
    [483., 546.],
    [504., 545.],
    [525., 550.],
    [543., 572.],
    [546., 590.],
    [549., 607.],
    [552., 625.],
    [529., 647.],
    [539., 648.],
    [549., 649.],
    [557., 648.],
    [519., 698.],
    [530., 686.],
    [542., 677.],
    [548., 679.],
    [555., 675.],
    [565., 682.],
    [571., 693.],
    [564., 691.],
    [556., 692.],
    [548., 693.],
    [541., 693.],
    [530., 694.],
    [525., 696.]
    ])

points2 = np.matrix([
    [144., 159.],
    [147., 180.],
    [149., 201.],
    [153., 222.],
    [161., 241.],
    [176., 255.],
    [194., 268.],
    [214., 278.],
    [235., 279.],
    [254., 274.],
    [269., 261.],
    [282., 248.],
    [292., 232.],
    [296., 213.],
    [296., 192.],
    [296., 172.],
    [297., 152.],
    [162., 147.],
    [170., 135.],
    [184., 127.],
    [201., 125.],
    [215., 130.],
    [239., 129.],
    [252., 123.],
    [268., 124.],
    [281., 131.],
    [288., 143.],
    [229., 144.],
    [229., 156.],
    [230., 168.],
    [231., 181.],
    [217., 194.],
    [224., 195.],
    [232., 197.],
    [239., 194.],
    [245., 192.],
    [183., 151.],
    [190., 147.],
    [199., 146.],
    [207., 150.],
    [199., 152.],
    [191., 153.],
    [246., 148.],
    [254., 143.],
    [263., 143.],
    [270., 147.],
    [263., 149.],
    [255., 149.],
    [205., 222.],
    [216., 219.],
    [225., 215.],
    [233., 216.],
    [241., 214.],
    [250., 215.],
    [260., 216.],
    [252., 223.],
    [244., 227.],
    [236., 228.],
    [228., 229.],
    [217., 227.],
    [210., 222.],
    [226., 221.],
    [234., 220.],
    [241., 219.],
    [256., 217.],
    [242., 219.],
    [235., 221.],
    [227., 221.]
    ])


def read_landmarks(im):
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))

    rects = detector(im, 1)

    if len(rects) > 1:
            raise Exception
    if len(rects) == 0:
            raise Exception

    ps = predictor(im, rects[0])
    s = np.matrix([[p.x, p.y] for p in ps.parts()])

    return im, s


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

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

brad_face_image = cv2.imread('brad-face.jpg')
im2, landmarks2 = read_landmarks(brad_face_image)


mask = get_face_mask(im2, landmarks2)
cv2.imshow("show", mask)
cv2.waitKey(0)




