import numpy as np
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

# 適切なキーポイントを選択
img02_matched_kpts = np.float32(
    [img01_kp[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
sensed_matched_kpts = np.float32(
    [img02_kp[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# ホモグラフィを計算
H, status = cv.findHomography(
    img02_matched_kpts, sensed_matched_kpts, cv.RANSAC, 5.0)

# 画像を変換
warped_image = cv.warpPerspective(
    img01, H, (img01.shape[1], img01.shape[0]))

cv.imwrite('img/result/result_warped_01.jpg', warped_image)