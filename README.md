# A LOOK INTO PAST
## 成员及分工

* 戴瑞  PB18061305

    * 设计，coding

* 朱舒涵  PB18000107

    * 调研，debug

## 问题描述
* ### 动机
    现实生活中有很多地方都留下了许多珍贵且富有意义的照片，现在的人们也很喜欢对比同一处的景色在不同时间的差别，尤其有一种对比方式尤为有趣——拿着过去的照片与现在的实景进行拼接，带来一种历史和现实交错的美感。然而现实中完成这样的创意并不简单，通常需要作者亲自前往此处进行麻烦的摆拍，效果也还不一定太好。

* ### 创意描述
    本次项目中，我们试图完成一个程序，能自动将历史照片和现实照片适当的进行拼接，以达到上述的效果。

* ### 具体实现
    该效果实现与全景拼接技术有些类似，使用算法识别历史图片和现实图片相近的特征点，根据这些特征点的对应位置获得图片位置的对应关系和投影方法，再将历史图片投影到现实图片上

## 原理分析
* ### 总体步骤
    1.利用sift（或其他算法）算法找出两种图片的相似特征点（特征匹配），计算对应的单应性矩阵。

    2.使用单应性矩阵变换一张图片到另一种图片上合适的位置。


### 1，单应性矩阵 
    
    描述物体在世界坐标系和像素坐标系之间的位置映射关系。对应的变换矩阵称为单应性矩阵。它在图像校正、图像拼接、相机位姿估计、视觉SLAM等领域有非常重要的作用。
### 2，SIFT算法和特征匹配
    
    SIFT即尺度不变特征变换，是用于图像处理领域的一种描述。这种描述具有尺度不变性，可在图像中检测出关键点，是一种局部特征描述子。

一、SIFT算法特点：

    1、具有较好的稳定性和不变性，能够适应旋转、尺度缩放、亮度的变化，能在一定程度上不受视角变化、仿射变换、噪声的干扰。

    2、区分性好，能够在海量特征数据库中进行快速准确的区分信息进行匹配

    3、多量性，就算只有单个物体，也能产生大量特征向量

    4、高速性，能够快速的进行特征向量匹配

    5、可扩展性，能够与其它形式的特征向量进行联合

二、SIFT算法实质

在不同的尺度空间上查找关键点，并计算出关键点的方向。

![p1](/data/text.png)

三、SIFT算法实现特征匹配主要有以下三个流程：

    1、提取关键点：关键点是一些十分突出的不会因光照、尺度、旋转等因素而消失的点，比如角点、边缘点、暗区域的亮点以及亮区域的暗点。此步骤是搜索所有尺度空间上的图像位置。通过高斯微分函数来识别潜在的具有尺度和旋转不变的兴趣点。

    2、定位关键点并确定特征方向：在每个候选的位置上，通过一个拟合精细的模型来确定位置和尺度。关键点的选择依据于它们的稳定程度。然后基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向。所有后面的对图像数据的操作都相对于关键点的方向、尺度和位置进行变换，从而提供对于这些变换的不变性。

    3、 通过各关键点的特征向量，进行两两比较找出相互匹配的若干对特征点，建立景物间的对应关系。


更多有关sift算法具体细节请看
https://blog.csdn.net/qq_37374643/article/details/88606351

### 3，利用RANSAC算法求解单应性矩阵
  
    虽然SIFT是具有很强稳健性的描述子，当这方法仍远非完美，还会存在一些错误的匹配。而单应性矩阵需要选取4对特征点计算，万一选中了不正确的匹配点，那么计算的单应性矩阵肯定是不正确的。因此，为了提高计算结果的鲁棒性，我们下一步就是要把这些不正确的匹配点给剔除掉，获得正确的单应性矩阵。  
    RANSAC(Random Sample Consensus)即随机采样一致性，该方法是用来找到正确模型来拟合带有噪声数据的迭代方法。给定一个模型，例如点集之间的单应性矩阵，RANSAC的基本思想在于，找到正确数据点的同时摒弃噪声点。
### 4，Lowe's ratio test
    为了进一步筛选匹配点，来获取优秀的匹配点，这就是所谓的“去粗取精”。一般会采用Lowe’s ratio test算法来进一步获取优秀匹配点。
    为了排除因为图像遮挡和背景混乱而产生的无匹配关系的关键点，SIFT的作者Lowe提出了比较最近邻距离与次近邻距离的SIFT匹配方式：取一幅图像中的一个SIFT关键点，并找出其与另一幅图像中欧式距离最近的前两个关键点，在这两个关键点中，如果最近的距离除以次近的距离得到的比率ratio少于某个阈值T，则接受这一对匹配点。因为对于错误匹配，由于特征空间的高维性，相似的距离可能有大量其他的错误匹配，从而它的ratio值比较高。显然降低这个比例阈值ratiolimit，SIFT匹配点数目会减少，但更加稳定，反之亦然。


## 代码实现
### 1 加载图片和主函数调用
```python
    #加载图像
    image_a = cv2.imread('picture{}/1.png'.format(i+1))
    image_b = cv2.imread('picture{}/2.png'.format(i+1))
    #加载预置参数
    sf,filter,meanfiler_time1,meanfiler_time2,ratiolimit = np.load('picture{}/parm.npy'.format(i+1),allow_pickle=True)
    #调用主函数opt
    matchesimg, img = opt(image_a, image_b, sf, filter, int(meanfiler_time1),int(meanfiler_time2),float(ratiolimit))
```

参数说明
```python
sf = ["sift","surf","orb"] #使用的特征识别算法
```
```python
filter = ["gray","oldyellow",None] #历史照片的种类
```
说明：使用时，历史照片可能有很多种类，如泛白的黑白照片或泛黄的老照片，选择对应种类后会对现实照片进行相应预处理使之更容易和历史照片进行特征匹配，增加鲁棒性。
```python
meanfiler_time1 = 0 or 1 or 2 or 3......#是否对输入的历史图片进行3x3的均值滤波处理，且连续滤波几次
meanfiler_time2 = 0 or 1 or 2 or 3......#是否对输入的现实图片进行3x3的均值滤波处理，且连续滤波几次
```
说明：实际使用中，输入的历史照片和现实照片的清晰度和细节丰富度根本不可相比，虽然sift算法拥有尺度不变特性，但为了更好的特征匹配和鲁棒性，经常会在检测前先对输入的现实图像进行均值滤波处理，这两个参数可以控制对输入的两张图片是否滤波和滤波的次数。也可大大提高运算速度（特征点变少）。
```python
ratiolimit = 0.5~0.98 #比率限制，见实验原理Lowe's ratio test
```

### 2 主函数 opt
```python
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

```
### 3 图像预处理函数 perprocess
```python
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
```
### 4 特征匹配和计算单应性矩阵函数 detect
```python
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

```
### 5 使用单应性变换矩阵进行原图的透视变换函数 projection
```python
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
```
### 6 其他函数
```python
def bgr_rgb(img):
    (r, g, b) = cv2.split(img)
    return cv2.merge([b, g, r])
```
```python
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
```
```python
def orb_detect(image_a, image_b):#orb算法detect，效果较差
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

```
## 效果展示

### Picture 1 真实历史照片拼接示例
使用的参数
```python
sf = "sift" 
filter = "gray" 
meanfiler_time1 = 1 
meanfiler_time2 = 1 
ratiolimit = 0.85
```
<font color='red' size=4> 
真实历史图片的拼接需要仔细调参，每一个参数对结果都有巨大影响，上述是最佳参数 
</font>

![1](/data/picture1/原图.png)
![1](/data/picture1/2.png)
![1](/data/picture1/matches.png)
![1](/data/picture1/result.png)

### Picture 2 真实同一地点不同时间拼接示例
![2](/data/picture2/1.png)
![2](/data/picture2/2.png)
![2](/data/picture2/matches.png)
![2](/data/picture2/result.png)

### Picture 3 原图部分人工后期处理再次拼接示例
![3](/data/picture3/原图.png)
![3](/data/picture3/2.png)
![3](/data/picture3/matches.png)
![3](/data/picture3/result.png)

## 工程结构
    ├── cv
    │ ├── data
    │ │ ├── picture1
    │ │ │ ├── 1.png
    │ │ │ ├── 2.png
    │ │ │ ├── 原图.png
    │ │ │ ├── matches.png    
    │ │ │ ├── parm.npy
    │ │ │ └── result.png
    │ │ ├── picture2
    │ │ │ ├── 1.png
    │ │ │ ├── 2.png
    │ │ │ ├── matches.png    
    │ │ │ ├── parm.npy
    │ │ │ └── result.png
    │ │ ├── picture3
    │ │ │ ├── 1.png
    │ │ │ ├── 2.png
    │ │ │ ├── 原图.png
    │ │ │ ├── matches.png    
    │ │ │ ├── parm.npy
    │ │ │ └── result.png
    │ │ ├── text.png
    │ │ └── tiaochan.ipynb
    │ ├── cv.py
    │ ├── cv2.py
    │ └── README.md


## 运行说明

### opencv版本 3.4.2
### Python 3.7.6 (default, Jan  8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
```python
python cv.py #会自动处理data预览中的3对图片
```

或修改cv2.py中的文件路径和参数，对其他图片进行处理
```python
python cv2.py #或修改cv2.py中的文件路径和参数，对其他图片进行处理
```