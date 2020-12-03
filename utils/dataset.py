# import some packages you need here
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class voc_seg(Dataset):

    def __init__(self, label_path, image_path, cut_out=False):
        # class에 들어갈 변수 및 기능 선언
        # 입력값으로 mask image가 들어있는 폴더, origin image가 있는 폴더, cut_out 여부를 받음.
        # __init__ 부분에서는 클래스 내부에서 동작하는 변수들을 선언
        # self.~ 변수는 voc_seg 안에서만 사용되는 변수

        # 마스크 이미지 경로와 원본 이미지 경로 선언
        self.label_path = label_path
        self.image_path = image_path

        # 이미지가 어떤 라벨을 가지는지 선언. 본 실험에서는 단일 클래스로 세그멘테이션 진행
        self.classes = ["teeth"]

        # 마스크 이미지 경로 파일 읽어오도록 변수 선언
        self.data_list = os.listdir(label_path)

        # cut out augmentation 진행할 cut_out 변수 선언
        self.cut_out = cut_out

        # 이미지 전처리를 위한 작업. 텐서화 / 리사이즈 (256,256) / RGB값 정규화 진행
        self.transform_1 = transforms.ToTensor()
        self.resize = transforms.Resize((256, 256))
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # class에 들어오는 변수들의 연산을 선언

        # data_list에 있는 파일 이름을 읽어옴
        base = self.data_list[idx]

        # 마스크 이미지와 원본 이미지의 확장자를 jpg로 통일
        label = os.path.join(self.label_path, base)
        image = os.path.join(self.image_path, base.replace(".bmp", ".jpg"))
        file_name = os.path.join(self.image_path, base.replace(".bmp", ".jpg"))
        label_name = os.path.join(self.label_path, base.replace(".png", ".jpg"))

        # 원본 이미지 전처리 과정
        # 기존의 RGB scale을 가지는 이미지를 gray scale로 변환
        image = Image.open(image)
        image = self.resize(image).convert("L")

        if self.cut_out:
            cut = cutout(mask_size=32, p=0.5, cutout_inside=True)
            image = cut(image)

        # 단일 이미지를 3차원으로 재변환
        # __init__에서 선언해준 전처리 진행 변수를 통해 전처리 시행(transform_1, normalize)
        image = self.transform_1(image)
        image = image.repeat(3, 1, 1)
        image = self.normalize(image)
        label = Image.open(label)
        label = self.resize(label).convert("L")
        label = self.transform_1(label)
        label = (label != 0) * 1.0
        label[label == 255] = 0
        non_smooth = label

        return image, label, file_name, non_smooth, label_name

    def get_classes(self):
        return self.classes


# cut_out augmentation 정의
# 이미지에 다수의 구멍을 내서 탐지하려는 객체의 일부분만 보고도 학습이 가능하게끔 적용

def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    # 본 코드에서는 구멍을 mask라고 정의
    # mask를 계산하기 위해 척도 변환
    # mask size를 반으로 나누고, 이를 기점으로 중심점과 x,y좌표값의 최소 최대값을 산출
    # cutout함수가 재귀로 짜여져 있기 때문에 _cutout에 들어가는 input image에 대한 설명이 별도로 존재하지 않음.
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image
        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        # 랜덤한 포인트에 center x, center y점을 잡고, 그 점을 기준으로 해서 mask 생성
        # mask 위치에 (0,0,0)인 검은색 픽셀 적용
        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout


if __name__ == '__main__':
    label_path = "Data/VOCdevkit/VOC2010/SegmentationClass/"
    image_path = "Data/VOCdevkit/VOC2010/JPEGImages"

    trainset = voc_seg(label_path, image_path, cut_out=False)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=36,
                                              shuffle=False,
                                              num_workers=0)