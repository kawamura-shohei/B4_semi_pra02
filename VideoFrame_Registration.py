import numpy as np
import cv2

def main():
    cv2.imwrite('img/result/result_warped.jpg', 
        FeatureMatching_And_Warped('../atami_frame/result_frame_img_', 3840, 4740, 900))


# 再帰的にSIFTによる特徴点記述とマッチングを行う関数
def FeatureMatching_And_Warped(img_path, FirstFrame_num, ProcessingFrame_num, Frame_step):
    # 画像読み込み
    if ProcessingFrame_num - Frame_step == FirstFrame_num:
        img01 = cv2.imread(img_path + str(FirstFrame_num) + '.jpg')
        img01 = cv2.cvtColor(img01, cv2.COLOR_BGR2RGB)
        img02 = cv2.imread(img_path + str(ProcessingFrame_num) + '.jpg')
        img02 = cv2.cvtColor(img02, cv2.COLOR_BGR2RGB)
    else:
        img01 = FeatureMatching_And_Warped(img_path, FirstFrame_num, ProcessingFrame_num - Frame_step, Frame_step)
        img01 = cv2.cvtColor(img01, cv2.COLOR_BGR2RGB)
        img02 = cv2.imread(img_path + str(ProcessingFrame_num) + '.jpg')
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

    # 適切なキーポイントを選択
    img02_matched_kpts = np.float32(
        [img01_kp[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sensed_matched_kpts = np.float32(
        [img02_kp[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # ホモグラフィを計算
    H, status = cv2.findHomography(
        img02_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)

    # 画像の変換と色をBGRに戻す
    warped_image = cv2.warpPerspective(
        img01, H, (img01.shape[1], img01.shape[0]))
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR)

    return warped_image


if __name__ == '__main__':
        main()
