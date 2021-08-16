import cv2
import numpy as np
from PIL import Image

def main():
    mask()
    gousei()


# マスク画像作成
def mask():
    # グレースケール化
    img_gray = cv2.imread('img/result/result_warped_01.jpg')
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY) #BGR2を指定
    cv2.imwrite("img/result/result_gray.jpg", img_gray)

    ret2, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU) #大津の二値化
    print("ret: {}".format(ret2)) #閾値表示
    cv2.imwrite("img/result/result_binary.jpg", img_binary)

    #膨張20回、収縮20回で穴を消す
    kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    ], dtype=np.uint8)
    img_delation = cv2.dilate(img_binary, kernel, iterations=15)
    img_erosion = cv2.erode(img_delation, kernel, iterations=15)
    cv2.imwrite("img/result/result_mask.jpg", img_erosion)


def gousei():
    img_background = Image.open('../atami_frame/result_frame_img_3840.jpg')
    img_registration = Image.open('img/result/result_warped_01.jpg')
    img_copy = img_background.copy()

    # マスク画像読み込み
    img_mask = Image.open('img/result/result_mask.jpg').convert("L")

    # マスク画像を基に貼り合わせ
    img_copy.paste(img_registration, (0, 0), img_mask)
    img_copy.save("img/result/result_registration.jpg")


if __name__ == '__main__':
        main()
