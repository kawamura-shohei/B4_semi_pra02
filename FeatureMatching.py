import cv2

# 画像読み込み
img01 = cv2.imread('../atami_registration/result_registration_7980.jpg')
img01 = cv2.cvtColor(img01, cv2.COLOR_BGR2RGB)
img02 = cv2.imread('img/atami_haikei/20210706_saigaigo_joku.jpg')
img02 = cv2.cvtColor(img02, cv2.COLOR_BGR2RGB)

# キーポイントの検出と特徴の記述
sift = cv2.SIFT_create()
img01_kp, img01_des = sift.detectAndCompute(img01, None)
img02_kp, img02_des = sift.detectAndCompute(img02, None)

# 特徴のマッチング
bf = cv2.BFMatcher()
matches = bf.knnMatch(img01_des, img02_des, k=2)

# 正しいマッチングのみ保持
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

matches_img = cv2.drawMatchesKnn(
    img01,
    img01_kp,
    img02,
    img02_kp,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
matches_img = cv2.cvtColor(matches_img, cv2.COLOR_RGB2BGR)

cv2.imwrite('img/result/result_matches_01.jpg', matches_img)