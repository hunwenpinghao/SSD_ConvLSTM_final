import cv2
from data import RDVOC_ROOT_test, RDVOCAnnotationTransform, RDVOCDetection, BaseTransform

dataset_mean = (104, 117, 123)

def plot(img, rect1, rect2, label1, label2, score, savepath):
    (xmin_1, ymin_1, xmax_1, ymax_1) = (int(rect1[0]), int(rect1[1]), int(rect1[2]), int(rect1[3]))
    (xmin_2, ymin_2, xmax_2, ymax_2) = (int(rect2[0]), int(rect2[1]), int(rect2[2]), int(rect2[3]))
    pt1_1 = (xmin_1, ymin_1)
    pt2_1 = (xmax_1, ymax_1)
    pt1_2 = (xmin_2, ymin_2)
    pt2_2 = (xmax_2, ymax_2)
    cv2.rectangle(img, pt1_1, pt2_1, (0, 0, 255), 1)
    cv2.rectangle(img, pt1_2, pt2_2, (255, 0, 0), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, label1+':'+str(score), (xmin_1 - 3, ymin_1 - 3), font, 0.6, (0, 0, 255), 1)  # 图像 文字内容 坐标 字体 大小 颜色 字体厚度
    cv2.putText(img, label2, (xmin_2 - 3, ymax_2 + 20), font, 0.6, (255, 0, 0), 1)  # 图像 文字内容 坐标 字体 大小 颜色 字体厚度
    # cv2.putText(img, str(score), (xmin_1 + 30, ymin_1 - 3), font, 0.6, (255, 0, 0), 1)
    # cv2.imshow('IMG', img)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()
    cv2.imwrite(savepath, img, [int(cv2.IMWRITE_JPEG_QUALITY),95])
    return img

# def plot(img, rect1, label1, score1):
#     (xmin_1, ymin_1, xmax_1, ymax_1) = (int(rect1[0]), int(rect1[1]), int(rect1[2]), int(rect1[3]))
#     pt1_1 = (xmin_1, ymin_1)
#     pt2_1 = (xmax_1, ymax_1)
#     cv2.rectangle(img, pt1_1, pt2_1, (0, 0, 255), 1)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(img, label1, (xmin_1+5, ymin_1+5), font, 0.8, (0, 0, 255), 1)  # 图像 文字内容 坐标 字体 大小 颜色 字体厚度
#     cv2.putText(img, score1, (xmin_1+5, ymin_1+5), font, 0.8, (0, 0, 255), 1)
#     # cv2.imshow('IMG', img)
#     # cv2.waitKey(50)
#     # cv2.destroyAllWindows()
#     return img

if __name__ == '__main__':
    classes = ('target1', 'target2', 'target3')
    dataset = RDVOCDetection(RDVOC_ROOT_test, [('RDVOC_new')],
                               BaseTransform(300, dataset_mean),
                               RDVOCAnnotationTransform())
    img =  dataset.pull_image(1)
    anno = dataset.pull_anno(1) # (imgid, [xmin, ymin, xmax, ymax, label_ind])
    label_ind = anno[1][0][4]
    cls = classes[label_ind]
    rect = anno[1][0][:4]
    plot(img, rect, cls)
