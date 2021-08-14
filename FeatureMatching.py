import cv2 as cv

# 画像読み込み
img01 = cv.imread('../atami_frame/result_frame_img_3840.jpg')
img02 = cv.imread('../atami_frame/result_frame_img_3870.jpg')

# キーポイントの検出と特徴の記述
akaze = cv.AKAZE_create()
img01_kp, img01_des = akaze.detectAndCompute(img01, None)
img02_kp, img02_des = akaze.detectAndCompute(img02, None)

# 特徴のマッチング
bf = cv.BFMatcher()
matches = bf.knnMatch(img01_des, img01_des, k=2)

# 正しいマッチングのみ保持
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

matches_img = cv.drawMatchesKnn(
    img01,
    img01_kp,
    img02,
    img02_kp,
    good_matches,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('img/result/result_matches_01.jpg', matches_img)