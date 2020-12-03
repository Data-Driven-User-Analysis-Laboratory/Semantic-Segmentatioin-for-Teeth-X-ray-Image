# coding=utf-8
import argparse
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import average_precision_score, jaccard_score
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler

from utils import dataset
from models.fcn import FCNs, VGGNet
from utils.predict import save_fig
from utils.losses import DiceLoss
from utils.optimizers import RAdam

current_time = str(datetime.datetime.now().timestamp())

# 파서 설정
# 각 파서들은 코드내부에서 args.파서이름 이런식으로 활용됨
# ex) --mode의 default값은 segmentation. args.mode는 segmentation으로 설정되어 있음.
parser = argparse.ArgumentParser()

# 세그멘테이션 / 분류 모델 결정 파서 => main함수에서 사용
parser.add_argument("--mode", default="segmentation", type=str,
                    help="Task Type, For example segmentation or classification")

# 옵티마이저 결정 (sgd, adam, radam) => main함수에서 사용
parser.add_argument("--optim", default="radam", type=str, help="Optimizers")

# loss functioin 결정 [cross_entropy, bce(binary cross entropy), dice] => main함수에서 사용
parser.add_argument("--loss-function", default="cross_entropy", type=str)

# 시행횟수 결정 => main함수에서 사용
parser.add_argument("--epochs", default=50, type=int)

# agumentation 시행여부 결정 [cutout,smooth] => main함수에서 사용
parser.add_argument("--tricks", default="None", type=str)

# train과 validation 학습 진행시 이미지 몇개를 가져다가 학습할건지 결정 => main함수에서 사용
parser.add_argument("--batch-train", default=8, type=int)
parser.add_argument("--batch-val", default=8, type=int)

args = parser.parse_args()

# tensorboard에 시각화하기 위해 SummaryWriter 선언 (최초에 선언해줘야 텐서보드 활성화 가능)
# 로그를 기록하기 위한 용도로 사용.
writer_train = SummaryWriter(log_dir='logs/tensorboard/fcn_train/' + current_time, comment="SK")
writer_valid = SummaryWriter(log_dir='logs/tensorboard/fcn_valid/' + current_time, comment="SK")


# train 함수 정의
def train(model, trn_loader, criterion, optimizer, epoch, mode="classification"):
    # 시작시간 체크
    start_time = time.time()
    trn_loss = 0
    # total_prob와  total_target 파라미터를 담기 위해 빈 np.array 생성
    total_prob = np.zeros((0, 20))
    total_target = np.zeros((0, 20))

    result_dir = 'logs/fcn'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # trn_loader 리턴값에 따라서 인자를 다섯개로 받게 설정. dataset.py 참조
    for i, (image, target, file_name, non_smooth_target, label_name) in enumerate(trn_loader):
        # 후에 save_fig를 통해 학습 결과를 시각화하기 위해 origin_image와 origin_label을 직접 받아옴
        # dataloader가 가져오는 이미지와 라벨값이 배치가 잡혀있기 때문
        # origin_image.shape -> (3292,1536) + gray scale adapt <- [.convert("L")]
        origin_image = np.array(Image.open(file_name[0]).resize((256, 256)).convert("L"))
        # origin_image.shape -> (3292,1536) + gray scale adapt <- [.convert("L")]
        origin_label = np.array(Image.open(label_name[0]).resize((256, 256)).convert("L"))
        origin_label = (origin_label != 0) * 1.0  # int 자료형을 float자료형으로 변환
        model.train()  # .train()을 통해 학습준비를 위한 세팅
        x = image.cuda()  # image와 target을 gpu로 할당
        y = target.cuda()  # image와 target을 gpu로 할당
        y_pred = model(x)  # 모델에 데이터 투입
        ious = np.zeros((2,))  # iou를 담기위한 빈 np.array 생성

        if mode == "segmentation":
            # loss_function에 라벨이미지와 모델출력값 입력. criterion은 main함수에서 정의된 loss_function의
            # 통상적으로 pytorch 사용자들이 loss_function 이름을 criterion으로 사용함
            loss = criterion(y_pred, y)
            pred = F.softmax(y_pred, dim=1)  # 예측값에 softmax 적용
            pos_probs = torch.sigmoid(y_pred)  # 예측값에 sigmoid 적용
            pos_preds = (pos_probs > 0.5).float().cpu().numpy()  # save_fig의 인풋이 cpu임으로 cpu자료형으로 변환
            _, max_index = torch.max(pred, 1)
            iou, dice = performance(y_pred, y)  # dice loss와 iou 계산 / main 함수 이전 performance 참조
            # savie_fig 함수를 통해 실제이미지, 라벨링이미지, 예측한 이미지 출력. + 파일 출저 설명
            save_fig(result_dir=result_dir, image=origin_image, predic=pos_preds[0],
                     target=origin_label, iou=iou, dice=dice)
            ious += iou

        elif mode == "classification":
            loss = criterion(y_pred, y)
            pos_probs = torch.sigmoid(y_pred)
            total_prob = np.concatenate((total_prob, pos_probs.detach().cpu().numpy()))

        optimizer.zero_grad()  # 역전파 실행전에 0값으로 초기화 / main 참조
        loss.backward()  # 역전파 연산 실행 / main 참조
        optimizer.step()  # 매개변수 갱신
        trn_loss += loss
        end_time = time.time()  # 종료시간 체크
        # 시행횟수 출력
        print(" [Training] [{0}] [{1}/{2}]".format(epoch, i, len(trn_loader)))

    trn_loss = trn_loss / len(trn_loader)

    if mode == "classification":
        total_measure = average_precision_score(total_target, total_prob)

    if mode == "segmentation":
        total_measure = ious

    folder_name = 'teeth ' + time.strftime('%Y-%m-%d', time.localtime(time.time()))
    if not os.path.exists(folder_name + '/{}-{}-{}/'.format(args.mode, args.method, args.tricks)):
        os.makedirs(folder_name + '/{}-{}-{}/'.format(args.mode, args.method, args.tricks))

    # 모델 정보와 파라미터를 가진 pickle파일 저장
    if epoch == 15 or epoch == 24 or epoch == 49 or epoch == 74 or epoch == 99 or epoch == 124 or epoch == 149 or \
            epoch == 174 or epoch == 199:
        if not os.path.exists(folder_name + '/workspace/pjh/deep_lr_prj/{}-{}-{}/'.format(args.mode, args.method,
                                                                                          args.tricks)):
            os.makedirs(folder_name + '/workspace/pjh/deep_lr_prj/{}-{}-{}/'.format(args.mode, args.method,
                                                                                    args.tricks))
            torch.save(model.state_dict(), folder_name + '/workspace/pjh/deep_lr_prj/{}-{}-{}/{}-model.pth'
                       .format(args.mode, args.method, args.tricks, epoch))

    # train 상태 출력
    print(" [Training] [{0}] [{1}/{2}] ", '\n',
          "Train Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}] Measure = [{5:.3f}]".format(epoch, i + 1, len(trn_loader),
                                                                                          trn_loss,
                                                                                          end_time - start_time,
                                                                                          total_measure[0]))
    # 텐서보드에 이미지 입력
    image_y = pos_probs.reshape(3, 256, 256)  # criterion 함수 통과서 (3,1,256,256)으로 나옴. 이를 reshpae해서 tensorboard로 넣음
    writer_train.add_image("prediction_image", image_y, epoch)
    writer_train.close()

    return trn_loss, total_measure


def validate(model, val_loader, criterion, epoch, mode="segmentation"):
    start_time = time.time()
    val_loss = 0
    model.eval()  # validation 준비
    # 값들을 넣기위한 빈 np.array 생성
    ious = np.zeros((2,))
    total_prob = np.zeros((0, 20))
    total_target = np.zeros((0, 20))
    # detach()의 경우 변수마다 거는 방식. 집단으로 연산추적을 방지하려면 with torch.no_grad() 함수 이용
    with torch.no_grad():
        for i, (data, target, file_name, non_smooth_target, label_name) in enumerate(val_loader):
            x = data.cuda()  # 원본 이미지, 라벨 이미지 gpu로 할당
            y = target.cuda()
            y_pred = model(x)  # 모델에 데이터 입력

            if mode == "segmentation":
                loss = criterion(y_pred, y)  # loss_function에 예측값 입력
                pred = F.softmax(y_pred, dim=1)  # 예측값에 softmax 연산 시행

                # 자카드 점수 계산하기 위한 자료형 변환
                _, max_index = torch.max(pred, 1)
                index_flatten = max_index.view(-1).cpu()
                target_flatten = non_smooth_target.view(-1).cpu()
                li = [0, 1]
                iou = jaccard_score(index_flatten, target_flatten.round(), labels=li, average=None)
                ious += iou

            val_loss += (loss)
            end_time = time.time()
            print(" [Validation] [{0}] [{1}/{2}]".format(epoch, i + 1, len(val_loader)))

    val_loss /= len(val_loader)
    if mode == "segmentation":
        mean_total = ious
    else:
        mean_total = average_precision_score(total_target, total_prob)

    print(" [Validation] [{0}] [{1}/{2}]", '\n',
          "Validation Losses = [{3:.4f}] Time(Seconds) = [{4:.2f}] Measure [{5:.3f}]".format(epoch, i + 1,
                                                                                             len(val_loader),
                                                                                             val_loss,
                                                                                             end_time - start_time,
                                                                                             mean_total[0] / len(
                                                                                                 val_loader)))
    return val_loss, mean_total / len(val_loader)


# dice loss 와 iou 값 계산
def performance(output, target):
    pos_probs = torch.sigmoid(output)
    pos_preds = (pos_probs > 0.5).float()
    pos_preds = pos_preds.cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()
    if target.sum() == 0:  # background patch
        return 0, 0
    # IoU
    union = ((pos_preds + target) != 0).sum()
    intersection = (pos_preds * target).sum()
    iou = intersection / union
    # dice
    dice = (2 * intersection) / (pos_preds.sum() + target.sum())
    return iou, dice


def main():
    # 각 설정에 따라 실험에 필요한 라벨 경로, 이미지 경로, trainset 환경 구축 및 데이터 분배
    if args.mode == "segmentation":
        label_path = "Data/VOCdevkit/VOC2010/SegmentationClass/"
        image_path = "Data/VOCdevkit/VOC2010/JPEGImages"

        # augmentation 값에 따른 dataset 호출.
        # cutout은 이미지 일부를 삭제함으로서 특징 일부만 가지고 학습이 가능하게함
        # 본 실험에서는 augmentation을 시행하지 않고도 좋은 성능을 측정함
        if args.tricks == "cut-out":
            trainset = dataset.voc_seg(label_path, image_path, cut_out=True)
            valset = dataset.voc_seg(label_path, image_path, cut_out=False)
        else:
            trainset = dataset.voc_seg(label_path, image_path, cut_out=False)
            valset = dataset.voc_seg(label_path, image_path, cut_out=False)

        total_idx = list(range(len(trainset)))
        split_idx = int(len(trainset) * 0.7)  # data split
        trn_idx = total_idx[:split_idx]
        val_idx = total_idx[split_idx:]

    if args.tricks == "cut-out":
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_train, shuffle=False,
                                                  sampler=SubsetRandomSampler(trn_idx))
        testloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_val, shuffle=False,
                                                 sampler=SubsetRandomSampler(trn_idx))
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_train, shuffle=False,
                                                  sampler=SubsetRandomSampler(trn_idx))
        testloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_val, shuffle=False,
                                                 sampler=SubsetRandomSampler(trn_idx))
    # backbone network 설정
    if args.mode == "segmentation":
        vgg_model = VGGNet(requires_grad=True)
        net = FCNs(pretrained_net=vgg_model, n_class=1)

    elif args.mode == "classification":
        net = torchvision.models.resnet50(pretrained=False, num_classes=1)

    else:
        raise NotImplementedError

    # optimizer 설정
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    elif args.optim == 'radam':
        optimizer = RAdam(net.parameters(), lr=0.0001)
    else:
        raise NotImplementedError

    # multi-gpu 설정
    net = nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    # loss function 설정
    if args.loss_function == "bce":
        criterion = nn.BCEWithLogitsLoss().cuda()

    elif args.loss_function == "dice":
        criterion = DiceLoss().cuda()

    elif args.loss_function == "cross_entropy":
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        raise NotImplementedError

    # training & validation
    losses = []
    tr_acc = []
    val_losses = []
    val_acc = []
    for epoch in range(args.epochs):
        tr, tac = train(net, trainloader, criterion, optimizer, epoch, mode=args.mode)
        va, vac = validate(net, testloader, criterion, epoch, mode=args.mode)
        losses.append(tr)
        tr_acc.append(tac)
        val_losses.append(va)
        val_acc.append(vac)

    return losses, tr_acc, val_losses, val_acc


if __name__ == '__main__':
    import os
    import time
    import matplotlib.pyplot as plt

    tr_loss, tr_acc, val_loss, val_acc = main()

    folder_name = 'teeth ' + time.strftime('%Y-%m-%d', time.localtime(time.time()))

    if not os.path.exists(folder_name + '/{}-{}-{}/'.format(args.mode, args.method, args.tricks)):
        os.makedirs(folder_name + '/{}-{}-{}/'.format(args.mode, args.method, args.tricks))

    x_lab = []
    for i in range(args.epochs):
        x_lab.append(i)

    # plot for loss
    plt.plot(x_lab, tr_loss, label='train_loss')
    plt.plot(x_lab, val_loss, label='val_loss')
    plt.title("FCN loss")
    plt.legend()
    plt.savefig(folder_name + '/fcn_measure' + current_time + '.png', dpi=300)

    try:
        torch.save(tr_loss, folder_name + '/deep_lr_prj/{}-{}-{}/{}-{}-{}-Trainloss.pkl'.format(args.mode, args.method,
                                                                                                args.tricks, args.mode,
                                                                                                args.method,
                                                                                                str(args.epochs)))
        torch.save(tr_acc, folder_name + '/deep_lr_prj/{}-{}-{}/{}-{}-{}-Trainacc.pkl'.format(args.mode, args.method,
                                                                                              args.tricks, args.mode,
                                                                                              args.method,
                                                                                              str(args.epochs)))
        torch.save(val_loss, folder_name + '/deep_lr_prj/{}-{}-{}/{}-{}-{}-Validloss.pkl'.format(args.mode, args.method,
                                                                                                 args.tricks, args.mode,
                                                                                                 args.method,
                                                                                                 str(args.epochs)))
        torch.save(val_acc, folder_name + '/deep_lr_prj/{}-{}-{}/{}-{}-{}-Validacc.pkl'.format(args.mode, args.method,
                                                                                               args.tricks, args.mode,
                                                                                               args.method,
                                                                                               str(args.epochs)))
    except:
        pass
