import cv2 as cv

# キーポイントなどを見やすくするためにグレースケールで画像読み込み
img = cv.imread('../atami_frame/result_Frame_img_3836_127.99.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# キーポイントの検出と特徴の記述
akaze = cv.AKAZE_create()
kp, descriptor = akaze.detectAndCompute(img, None)

keypoints_img = cv.drawKeypoints(img, kp, img)
cv.imwrite('img/result/result_keypoints_01.jpg', keypoints_img)