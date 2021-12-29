from matplotlib import pyplot as plt
import cv2
import numpy as np
print('opencv版本: ', cv2.__version__)


def bgr_rgb(img):
    (r, g, b) = cv2.split(img)
    return cv2.merge([b, g, r])


def preprocess(img11, img21, filter=None, meanfilter_time1=0, meanfilter_time2=0):# 图像预处理
    img1 = img11.copy()
    img2 = img21.copy()

    if filter:#历史照片的种类
        if filter == "gray":
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if filter == "oldyellow":
            img2 = retro_style(img2)

    # 指定均值滤波次数
    if meanfilter_time1 > 0:#是否对输入的历史图片进行3x3的均值滤波处理，且连续滤波几次
        for i in range(meanfilter_time1):
            img1 = cv2.blur(img1, (3, 3))
    if meanfilter_time2 > 0:#是否对输入的现实图片进行3x3的均值滤波处理，且连续滤波几次
        for i in range(meanfilter_time2):
            img2 = cv2.blur(img2, (3, 3))

    plt.figure(1, dpi=200)
    plt.axis("off")
    plt.imshow(img1)
    plt.figure(2, dpi=200)
    plt.axis("off")
    plt.imshow(img2)

    return img1, img2


def detect(img1, img2, sf="sift", ratiolimit=0.6):

    if sf == "sift":
        sift = cv2.xfeatures2d.SIFT_create()
    elif sf == "surf":
        sift = cv2.xfeatures2d.SURF_create()
    elif sf == "orb":
        return orb_detect(img1, img2)
    else:
        print("无效算法")
        return None, None

    # 使用指定算法查找关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher 使用默认参数
    bf = cv2.BFMatcher()
    rawmatches = bf.knnMatch(des1, des2, k=2)

    kps1 = np.float32([kp.pt for kp in kp1])
    kps2 = np.float32([kp.pt for kp in kp2])
    matches = []
    good = []
    # 遍历初始匹配点
    for m in rawmatches:
        # 应用ratio测试，选出符合条件的匹配点(Lowe's ratio test)
        # 取图像1中的某个关键点，并找出其与图像2中距离最近的前两个关键点，在这两个关键点中，若最近的距离除以次近的距离小于某个阈值，则接受这一对匹配点。
        # 实验结果表明ratio取值（limit）在为最佳
        if len(m) == 2 and m[0].distance < m[1].distance * ratiolimit:
            matches.append((m[0].trainIdx, m[0].queryIdx))
            good.append([m[0]])

    # 使用cv2.drawMatchesKnn将匹配点画出
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    # 计算出一个单应性变换至少需要4对匹配点
    if len(matches) > 4:
        # 构造这两组点坐标为对应形式
        ptsA = np.float32([kps1[i] for (_, i) in matches])
        ptsB = np.float32([kps2[i] for (i, _) in matches])

        # 计算两组点之间的单应性变换矩阵以及每个匹配点的状态
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4)

    else:
        print("没有足够的匹配点")
        return matches, None, None, bgr_rgb(img3)
    return matches, H, status, bgr_rgb(img3)


def orb_detect(image_a, image_b):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image_a, None)
    kp2, des2 = orb.detectAndCompute(image_b, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    rawmatches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(image_a, kp1, image_b, kp2,
                           rawmatches[:10], None, flags=2)
    goodPoints = matches[:20] if len(matches) > 20 else matches[:]
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4)
    return matches, M, mask, bgr_rgb(img3)




def retro_style(img):# 泛黄老照片效果
    img2 = img.copy()
    height, width, n = img.shape
    for i in range(height):
        for j in range(width):
            b = img[i, j][0]
            g = img[i, j][1]
            r = img[i, j][2]
            # 计算新的图像中的RGB值
            B = int(0.273 * r + 0.535 * g + 0.131 * b)
            G = int(0.347 * r + 0.683 * g + 0.167 * b)
            R = int(0.395 * r + 0.763 * g + 0.188 * b)
            # 约束图像像素值，防止溢出
            img2[i, j][0] = max(0, min(B, 255))
            img2[i, j][1] = max(0, min(G, 255))
            img2[i, j][2] = max(0, min(R, 255))
    return img2

def projection(image_a, image_b, H):
    # 使用单应性变换矩阵进行原图的透视变换，将图1投影到图2大小的初始图上
    result = cv2.warpPerspective(
            image_a, H, (image_b.shape[1], image_b.shape[0]))
    result = bgr_rgb(result)
    img = bgr_rgb(image_b)

        # 将图2上对应位置替换成投影后的图1像素
    for i in range(image_b.shape[0]):
        for j in range(image_b.shape[1]):
            if sum(result[i, j]) > 0:
                img[i, j] = result[i, j]
    return img

def opt(image_a, image_b, sf="sift", filter=None, meanfilter_time1=1, meanfilter_time2=1, ratiolimit=0.6):
    #调用函数perprocess进行图像预处理
    img1, img2 = preprocess(image_a, image_b, filter,
                            meanfilter_time1, meanfilter_time2)
    #调用detect函数进行特征匹配和计算单应性矩阵
    matches, H, status, matchesimg = detect(img1, img2, sf, ratiolimit)

    if H.any():
        # 调用projection函数使用单应性变换矩阵进行原图的透视变换，将图1投影到图2大小的初始图上
        img = projection(image_a, image_b, H)
        return matchesimg, img
    else:
        print("失败")
        return matchesimg, None

if __name__ == '__main__': 

    #使用opt函数处理其他图片
    image_a = cv2.imread('xx/1.png')
    image_b = cv2.imread('xx/2.png')
    matchesimg, img = opt(image_a, image_b, "sift", "gray", 1,1,0.8)
    plt.figure(3, dpi=200)
    plt.axis("off")
    plt.imshow(matchesimg)
    plt.savefig("matches.png")
    if img.any():
        plt.figure(4, dpi=200)
        plt.imshow(img)
        plt.axis("off")
        plt.savefig("result.png")
    plt.show()
    print("done")