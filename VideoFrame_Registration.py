import cv2
import numpy as np
from PIL import Image

def main():
    # 順に  初期フレーム, 使用するフレームの間隔, 最後のフレーム(の1つ前), 使用したフレーム数の合計
    firstframe_num = 3840
    stepframe_num = 180
    endframe_num = 8101
    usedframe_num = ((endframe_num-1) - firstframe_num) // stepframe_num

    # 連続で貼り合わせる
    for i in range(firstframe_num + stepframe_num, endframe_num, stepframe_num):
        gousei(firstframe_num, stepframe_num ,i)

    # 対応点を与えて射影変換する
    homography('../atami_registration/result_registration_' + str(firstframe_num + stepframe_num * usedframe_num) + '.jpg')


# フレーム間でSIFTを用い対応点通りに射影変換するプログラム
## 引数は順に   フレーム画像のパス, 貼り合わせ画像のパス, 処理中のフレームの番号
def warped(img_path_01, registrationImg_path_02, processingframe_num):
    # 画像読み込み
    img01 = cv2.imread(img_path_01)
    img01 = cv2.cvtColor(img01, cv2.COLOR_BGR2RGB)
    img02 = cv2.imread(registrationImg_path_02)
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

    # 画像の形と色の変換
    warped_image = cv2.warpPerspective(
        img01, H, (img01.shape[1], img01.shape[0]))
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR)

    # 画像の保存
    cv2.imwrite('../atami_homography/result_warped_' + str(processingframe_num) + '.jpg', warped_image)
    
    return warped_image


# マスク画像作成プログラム
## 引数は順に   フレーム画像のパス, 貼り合わせ画像のパス, 処理中のフレームの番号
def mask(maskImg_path_01, maskImg_path_02, maskProcessingframe_num):
    # グレースケール化
    img_gray = cv2.cvtColor(warped(maskImg_path_01, maskImg_path_02, maskProcessingframe_num), cv2.COLOR_BGR2GRAY) #GRAYを指定
    
    ret2, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU) #大津の二値化
    print("ret: {}".format(ret2)) #閾値表示

    #膨張20回、収縮20回で穴を消す
    kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    ], dtype=np.uint8)
    img_delation = cv2.dilate(img_binary, kernel, iterations=15)
    img_erosion = cv2.erode(img_delation, kernel, iterations=16)

    # BGRを指定
    img_erosion = cv2.cvtColor(img_erosion, cv2.COLOR_GRAY2BGR)

    return img_erosion


# OpenCV型からPIL型へ変えるプログラム
def cv2pil(img_CV):
    img_PIL = img_CV.copy()
    if img_PIL.ndim == 2:  # モノクロ
        pass
    elif img_PIL.shape[2] == 3:  # カラー
        img_PIL = cv2.cvtColor(img_PIL, cv2.COLOR_BGR2RGB)
    elif img_PIL.shape[2] == 4:  # 透過
        img_PIL = cv2.cvtColor(img_PIL, cv2.COLOR_BGRA2RGBA)
    img_PIL = Image.fromarray(img_PIL)
    return img_PIL


# マスク画像を基に画像を貼り合わせるプログラム
## 引数は順に   フレーム画像のパス, 初期フレーム番号, 処理フレーム間隔 , 処理中のフレームの番号
def gousei(gouseiFirstframe_num, gouseiStepframe_num , gouseiProcessingframe_num):
    # 画像の読み込み
    if gouseiFirstframe_num + gouseiStepframe_num == gouseiProcessingframe_num: #初期フレームだった場合
        img_mask = cv2pil(mask('../atami_frame/result_frame_img_' + str(gouseiProcessingframe_num) + '.jpg', '../atami_frame/result_frame_img_' + str(gouseiFirstframe_num) + '.jpg', gouseiProcessingframe_num)).convert("L")
        img_background = Image.open('../atami_frame/result_frame_img_' + str(gouseiFirstframe_num) + '.jpg')
    else: # それ以外の場合
        img_mask = cv2pil(mask('../atami_frame/result_frame_img_' + str(gouseiProcessingframe_num) + '.jpg', '../atami_registration/result_registration_' + str(gouseiProcessingframe_num-gouseiStepframe_num) + '.jpg' , gouseiProcessingframe_num)).convert("L")
        img_background = Image.open('../atami_registration/result_registration_' + str(gouseiProcessingframe_num-gouseiStepframe_num) + '.jpg')
    img_registration = Image.open('../atami_homography/result_warped_' + str(gouseiProcessingframe_num) + '.jpg')
    img_copy = img_background.copy()

    # マスク画像を基に貼り合わせ
    img_copy.paste(img_registration, (0, 0), img_mask)
    img_copy.save('../atami_registration/result_registration_' + str(gouseiProcessingframe_num) + '.jpg')


# 手動で対応点を与えて射影変換するプログラム
def homography(img_in_name):
    img_in = cv2.imread(img_in_name) #鳥瞰視点画像

    # 変換前後の対応点を設定
    p_original = np.float32([[741, 566], [360, 282], [777, 126], [1096, 193]]) #鳥瞰視点画像の4点
    p_trans = np.float32([[905, 363], [749, 538], [461, 317], [675, 162]]) #直下視画像の4点

    # 変換マトリクスと射影変換
    M, mask = cv2.findHomography(p_original, p_trans, cv2.RANSAC) #0:すべてのポイントを使用する通常の方法、RANSAC：RANSACベースの堅牢な方法、LMEDS：最小中央値のロバストな方法
    img_trans = cv2.warpPerspective(img_in, M, (1280, 720)) #サイズ指定

    cv2.imwrite("img/result/result_homography.jpg", img_trans)
    return img_trans


if __name__ == '__main__':
        main()
