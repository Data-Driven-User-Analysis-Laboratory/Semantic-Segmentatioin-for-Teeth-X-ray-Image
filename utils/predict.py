import os
import time
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 모델 학습 결과를 시각화 하기위한 모듈
# 원본 이미지, 마스크 이미지, 모델 예측값을 받아서 이미지에 도식하는 모듈

def save_fig(result_dir, image, predic, target, iou, dice):
    # 원본 이미지, 마스크 이미지, 모델 예측값 이미지 shape 통일
    image = image.reshape(256, 256)
    predic = predic.reshape(256, 256)
    # 기존의 1차원 이미지를 (3,x,y)로 reshape
    image = np.dstack([image, image, image])
    target = np.dstack([target, target, target])

    # 원본 이미지와 마스크 이미지할 영역을 입력으로 받으며, 마스크 영역을 지정된 색깔에 따라 칠해주는 모듈
    def _overlay_mask(img, mask, color='red'):
        # convert gray to color
        color_img = img
        mask_idx = np.where(mask == 1.0) # 마스크 영역에 라벨링
        if color == 'red':
            print('img shape : ', image.shape, 'pred shape : ', predic.shape, 'target shape', target.shape)
            color_img[mask_idx[0], mask_idx[1], :] = np.array([255, 0, 0])
        elif color == 'blue':
            print('img shape : ', image.shape, 'pred shape : ', predic.shape, 'target shape', target.shape)
            color_img[mask_idx[0], mask_idx[1], :] = np.array([0, 0, 255])
        return color_img

    fig = plt.figure(figsize=(15, 5))

    # ax = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)]
    ax = []

    # 원본 이미지 시각화. 원본 이미지는 받아온 이미지 그대로 시각화 진행
    ax.append(fig.add_subplot(1, 3, 1))
    plt.imshow(image, 'gray')

    # 마스크 이미지 시각화 원본 이미지에 라벨 영역을 빨간색으로 도식
    ax.append(fig.add_subplot(1, 3, 2))
    plt.imshow(_overlay_mask(image, target, color='red'))
    ax[1].set_title("Segmentation Label")

    # 예측영역 이미지 시각화 원본 이미지에 예측된 영역 파란색으로 도식
    ax.append(fig.add_subplot(1, 3, 3))
    plt.imshow(_overlay_mask(image, predic, color='blue'))
    ax[-1].set_title('IoU = {0:.4f} \n DICE={1:.4f} \n Prediction Map'.format(iou, dice))

    # 이미지마다 Y축 제거
    for i in ax:
        i.axes.get_yaxis().set_visible(False)
    name = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    if iou == -1:
        res_img_path = os.path.join(result_dir, 'FILE{slice_id:0>4}_{iou}.png'.format(slice_id=name, iou='NA'))
    else:
        res_img_path = os.path.join(result_dir, 'FILE{slice_id:0>4}_{iou:.4f}.png'.format(slice_id=name, iou=iou))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 최종 이미지 저장
    plt.savefig(res_img_path, bbox_inches='tight')
    plt.close()
