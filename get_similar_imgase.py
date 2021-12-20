from PIL import Image
import os
import numpy as np
import time


from  PIL import Image

#得到切割的图像，方便更准去的统计直方图相似度
def get_split_img(img, split_num = 10):
    w, h = img.size
    pw, ph = int(w/split_num), int(h/split_num)
    assert w % pw == h % ph == 0
    return [img.crop((i, j, i+pw, j+ph)).copy() for i in range(0, w, pw) \
            for j in range(0, h, ph)]

#图片归一化大小
def noraml_resize(img, size=(500, 500)):
    return img.resize(size).convert('RGB')

def hist_similar_with_cv(lh, rh):
    assert len(lh) == len(rh)
    return cv2.compareHist(np.float32(lh), np.float32(rh), cv2.HISTCMP_CORREL)
#     return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) for l, r in zip(lh, rh))/len(lh)


#直接调用opencv的相关性方法
def hist_similar_cv(lh, rh):
    assert len(lh) == len(rh)
    return cv2.compareHist(np.float32(lh), np.float32(rh), cv2.HISTCMP_CORREL)

#直接调用方法实现
def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    split_image_similar_sum = 0
    for l, r in zip(lh, rh):
        if l == r:
            split_image_similar_sum +=0
        else:
            split_image_similar_sum += float(abs(l - r))/max(l, r)
        
    return 1 - split_image_similar_sum/len(lh)

def calc_similar(img0, img1, use_cv = False):
    split_num = 1
    img0_split = get_split_img(img0, split_num=split_num)
    img1_split = get_split_img(img1, split_num=split_num)
    hist_total = 0
    #通过opencv的方法实现
    if use_cv:
        for img0_, img1_ in zip(img0_split, img1_split):
            hist_total += hist_similar_cv(img0_.histogram(), img1_.histogram())
    else:#通过公式实现
        for img0_, img1_ in zip(img0_split, img1_split):
            hist_total += hist_similar(img0_.histogram(), img1_.histogram())        
    return hist_total/(split_num*split_num)

def calc_similar_by_path(img0_path, img1_path, use_cv = False):
    img0, img1 = noraml_resize(Image.open(img0_path)), noraml_resize(Image.open(img1_path))
    return calc_similar(img0, img1, use_cv = use_cv)

#对from_img目录下的图片进行便利，按照相似度进行聚类,相似图片会copy在to_img目录
from pathlib import Path
import shutil
if __name__ == "__main__":
    min_similar = 0.6
    from_img = './JPEGImages'
    to_img = 'diff_img'
    if not os.path.exists(to_img):
        os.makedirs(to_img)
    all_imgs = os.listdir(from_img)
    img_count = len(all_imgs)
    has_compare = []
    
    for i in range(img_count):
        count = 0
        img_index = os.path.join(from_img, all_imgs[i])
        print('-----------------', img_index, img_index not in has_compare)
        similar_imgs = []
        if img_index not in has_compare and os.path.exists(img_index):
            has_compare.append(img_index)
            similar_imgs.append(img_index)
            for j in range(i+1, img_count):
                count +=1
                compare_img_path = os.path.join(from_img, all_imgs[j])
                if os.path.exists(compare_img_path):
                    similar = calc_similar_by_path(img_index, compare_img_path)
                    print(similar, count)
                    if similar>min_similar:
                        similar_imgs.append(compare_img_path)
                        has_compare.append(compare_img_path)
            if len(similar_imgs) > 1:
                base_name = Path(img_index).name.split('.')[0]
                dir_name = os.path.join(to_img, base_name)
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                for img in similar_imgs:
                    img_name = Path(img).name
                    shutil.copy(img, os.path.join(dir_name, img_name))



