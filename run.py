import cv2
import numpy as np
from network.single_predict import detect_model
from PIL import Image
from Recognition import vsm_model


def __point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


# import easyocr
# reader = easyocr.Reader(['ch_sim', 'en'])  # 只需要运行一次就可以将模型加载到内存中
resnet18_model = detect_model()
vsm_model = vsm_model()

MIN_AREA = 250
min_area_ratio = 0.5  # 轮廓与外接矩形的最小比值
# folder_path = "./result"  # 结果文件夹路径

img = cv2.imread("pics/5.jpg", 1)
pic_hight, pic_width = img.shape[:2]

# 调用OpenCV库函数中的高斯滤波函数
result = cv2.GaussianBlur(img, (5, 5), 1, 1)  # 传入读取的图像和核尺寸 (5,5)必须是奇数
# 中值滤波
result = cv2.medianBlur(result, 3)

img_RGB = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
height, width, channels = img.shape
img_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
# cv2.imshow('res1', img)
# cv2.imshow('res2', result)

lower = np.array([60, 30, 30])
upper = np.array([80, 255, 255])  # 设定绿色的hsv阈值
mask2 = cv2.inRange(img_hsv, lower, upper)  # 设置掩模 只保留绿色部分
res = cv2.bitwise_and(img, img, mask=mask2)  # 利用掩模与原图像做“与”操作 过滤出绿色
# mask = np.stack([mask2] * 3, axis=2)  # mask矩阵拓展
# cv2.imshow('res', res)  # 显示最终视频

# 闭运算
kernel = np.ones((5, 5), dtype=np.uint8)
dilate = cv2.dilate(res, kernel, 5)  # 更改迭代次数为2
# cv2.imshow('res1', dilate)

kernel = np.ones((3, 3), dtype=np.uint8)
erosion = cv2.erode(dilate, kernel, iterations=2)
# cv2.imshow('res2', erosion)

img1 = cv2.cvtColor(erosion, cv2.COLOR_RGB2GRAY)
# cv2.imshow('img1',img1)
contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

oldimg = img
# # 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
card_imgs = []
car_contours = []
num = 0
img_show = img
if not contours:
    print('未检测到车牌\n')
else:
    for cnt in contours:  # TODO：此处需要一个异常处理（因为有可能/0）
        # 框选 生成最小外接矩形 返回值（中心(x,y), (宽,高), 旋转角度）rect[0]：矩形中心点坐标；rect[1]：矩形的高和宽；rect[2]：矩形的旋转角度
        rect = cv2.minAreaRect(np.float32(cnt))
        # print('宽高:',rect[1])
        area_width, area_height = rect[1]
        # 计算最小矩形的面积，初步筛选
        area = rect[1][0] * rect[1][1]  # 最小矩形的面积
        # TODO:矩形面积为什么会等于0
        if not area:
            continue
        # 计算轮廓面积
        retval = cv2.contourArea(cnt)
        # 计算轮廓面积和最小矩形面积的比值进行筛选
        area_ratio = retval / area
        if area_ratio < min_area_ratio:
            continue
        if area > MIN_AREA:
            # 选择宽大于高的区域
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            wh_ratio = area_width / area_height
            # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
            if 2.7 < wh_ratio < 5.5:
                car_contours.append(rect)  # rect是minAreaRect的返回值，根据minAreaRect的返回值计算矩形的四个点
                box = cv2.boxPoints(rect)  # box里面放的是最小矩形的四个顶点坐标
                # 倾斜校正
                if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
                    angle = 1
                else:
                    angle = rect[2]
                rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除
                box = cv2.boxPoints(rect)
                heigth_point = right_point = [0, 0]
                left_point = low_point = [pic_width, pic_hight]
                for point in box:
                    if left_point[0] > point[0]:
                        left_point = point
                    if low_point[1] > point[1]:
                        low_point = point
                    if heigth_point[1] < point[1]:
                        heigth_point = point
                    if right_point[0] < point[0]:
                        right_point = point

                if left_point[1] <= right_point[1]:  # 正角度
                    new_right_point = [right_point[0], heigth_point[1]]
                    pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
                    pts1 = np.float32([left_point, heigth_point, right_point])
                    M = cv2.getAffineTransform(pts1, pts2)
                    dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                    __point_limit(new_right_point)
                    __point_limit(heigth_point)
                    __point_limit(left_point)
                    card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
                    # card_imgs.append(card_img)

                elif left_point[1] > right_point[1]:  # 负角度

                    new_left_point = [left_point[0], heigth_point[1]]
                    pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
                    pts1 = np.float32([left_point, heigth_point, right_point])
                    M = cv2.getAffineTransform(pts1, pts2)
                    dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                    __point_limit(right_point)
                    __point_limit(heigth_point)
                    __point_limit(new_left_point)
                    card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
                    # card_imgs.append(card_img)
                if card_img.shape[0] > card_img.shape[1]:
                    continue
            else:
                continue

            cv2.imshow('card_img', card_img)

            # -------对图像进行字符分割-------
            # 读入图像
            myImage = data = cv2.resize(card_img, dsize=(480, 140), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            # 灰度化
            grayImage = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
            # 二值化
            ret, thresh = cv2.threshold(grayImage, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            cv2.imshow('bw' + str(num), thresh)

            # 形态学操作
            kernel = np.ones((3, 3), dtype=np.uint8)
            erosion = cv2.erode(thresh, kernel, iterations=2)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilation = cv2.dilate(erosion, horizontal_kernel, iterations=5)
            cv2.imshow('bw', dilation)
            cv2.imshow('bw2' + str(num), erosion)

            # 查找轮廓
            horizontal_contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # print(len(horizontal_contours))
            if len(horizontal_contours) < 3:
                continue
            num += 1

            chars = []
            # 在原始图像上绘制红色矩形和进行字符分割
            for cont in horizontal_contours:
                area = cv2.contourArea(cont)
                if area < 40:
                    continue
                x, y, w, h = cv2.boundingRect(cont)
                # 通过宽度和高度比值筛选掉一些矩形
                if w / h > 0.85 or w * h / (card_img.shape[0] * card_img.shape[1]) < 0.01:
                    continue
                else:
                    chars.append((x, y, w, h))

            chars.sort()
            print("char:", chars)

            count = 0
            for x, y, w, h in chars:
                # 绘制红色矩形
                rect = cv2.rectangle(myImage, (x, y), (x + w, y + h), (255, 255, 255), 1)

                # 在水平轮廓区域内进行字符分割
                roi = thresh[y:y + h, x:x + w]
                # cv2.imwrite("./pic/pic"+str(count)+".jpg", roi)
                size = roi.shape
                roi = cv2.copyMakeBorder(roi, 3, 3, int((size[0] - size[1]) / 4), int((size[0] - size[1]) / 4),
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
                roi1 = cv2.resize(roi, dsize=(20, 20), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
                roi3 = cv2.cvtColor(roi1, cv2.COLOR_GRAY2BGR)
                # cv2.imshow("1" + str(count), roi1)

                roi2 = Image.fromarray(cv2.cvtColor(roi3, cv2.COLOR_BGR2RGB))
                img_np = np.array(roi2)

                # 使用ocr识别字符
                # result = reader.readtext(roi1, detail=0)
                # print(result)
                # 使用模型识别
                # resnet18_model.detect(roi2)
                if count == 0:
                    vsm_model.detect_chinese(img_np)
                else:
                    vsm_model.detect(img_np)
                # print("shape", roi.shape)
                # cv2.imwrite("./pic/pic"+str(count)+".jpg", roi1)
                cv2.imshow("pic" + str(count), roi1)

                count = count + 1

                # # 进行字符分割的额外步骤，例如形态学操作、查找字符轮廓等
                # char_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                # char_dilation = cv2.dilate(roi, char_kernel, iterations=1)
                # char_contours, _ = cv2.findContours(char_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #
                # # 在原始图像上绘制字符的边界
                # for char_cnt in char_contours:
                #     char_x, char_y, char_w, char_h = cv2.boundingRect(char_cnt)
                #     char_rect = cv2.rectangle(myImage, (x + char_x, y + char_y),
                #                               (x + char_x + char_w, y + char_y + char_h),
                #                               (0, 255, 255), 1)

                # -------字符识别-------

            # 显示图像
            cv2.imshow('Image with Boundaries' + str(num), myImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
