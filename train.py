import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import glob
import hdf5storage
from random import shuffle
import time
import os
from model.Deasfn import Deasfn
from config.args import TrainArgs
import sys, getopt

os.environ["CUDA_VISIBLE_DEVICES"] = TrainArgs.CudaDevice


def getSequenceMinibatch(FileNames):
    sequence_num = len(FileNames)
    CsiDatas = torch.zeros(sequence_num, TrainArgs.NumFrames, 30, 25, 3, 3)
    heatmaps = torch.zeros(sequence_num, TrainArgs.NumFrames, 57, 46, 62)

    for i in range(sequence_num):
        for j in range(TrainArgs.NumFrames):
            data = hdf5storage.loadmat(FileNames[i][j], variable_names={'csi_serial', 'heatmaps'})
            CsiDatas[i, j, :, :, :, :] = torch.from_numpy(data['csi_serial']).type(torch.FloatTensor).permute(1, 0, 2, 3)
            heatmaps[i, j, :, :, :] = torch.from_numpy(data['heatmaps']).type(torch.FloatTensor)

    return CsiDatas, heatmaps


def TakeIndex(elem):
    return int(elem.split("/")[-1].split(".")[0])


def LoadDataset(DatasetName):

    # load the training data
    if DatasetName == 'SPE':
        data = []
        for action in TrainArgs.actions:
            for subject in TrainArgs.subjects:
                data.append(glob.glob(TrainArgs.DataPath + 'SPE/' + action + '/' + subject +  '/train/*.mat'))
    elif DatasetName == 'GPE':
        data = []
        for path in TrainArgs.GpePaths:
            data.append(glob.glob(TrainArgs.DataPath + 'GPE/' + path + '/train/*.mat'))
    else:
        print("Dataset name should be SPE or GPE")
        sys.exit()

    # sort by index
    for i in range(len(data)):
        data[i].sort(key = TakeIndex)

    # pack the data
    TrainData = []
    TrainRatio = TrainArgs.SpeTrainRatio if DatasetName == 'SPE' else TrainArgs.GpeTrainRatio
    for i in range(len(data)) :
        for j in range(0, int(len(data[i])*TrainRatio), TrainArgs.NumFrames):
            per_sequence = []
            for k in range(TrainArgs.NumFrames):
                per_sequence.append(data[i][j+k*TrainArgs.DilatedRate])
            TrainData.append(per_sequence)

    return TrainData


def train(TrainData):

    NumTrainData = len(TrainData)
    NumBatch = int(np.floor(NumTrainData/TrainArgs.BatchSize))
    print('NumFrames =', TrainArgs.NumFrames)
    print('NumTrainData =', NumTrainData)

    model = Deasfn()
    model.weights_init()
    optimizer = torch.optim.Adam(model.cuda().parameters(), lr=TrainArgs.LearningRate)

    # load the model
    if TrainArgs.LoadModel:
        if os.path.isfile(TrainArgs.CheckPoint):
            CheckPoint = torch.load(TrainArgs.CheckPoint)
            StartEpoch = CheckPoint['epoch']
            model.load_state_dict(CheckPoint['state_dict'])
            optimizer.load_state_dict(CheckPoint['optimizer'])
            print("=> loaded CheckPoint '{}' (epoch {})".format(TrainArgs.CheckPoint, CheckPoint['epoch']))
        else:
            StartEpoch = 0
            print("=> no CheckPoint found at '{}'".format(TrainArgs.CheckPoint))
    else:
        StartEpoch = 0
        print("=> do not load CheckPoint")


    model = model.cuda()
    criterion_L2 = nn.MSELoss(reduction='none').cuda()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=TrainArgs.milestone, gamma=0.5)

    model.train()
    for epoch in range(StartEpoch, TrainArgs.MaxEpoch):

        print('=> epoch:', epoch)
        start = time.time()
        shuffle(TrainData)

        # in each minibatch
        for batch_index in range(NumBatch):
            if batch_index < NumBatch:
                FileNames = TrainData[batch_index*TrainArgs.BatchSize:(batch_index+1)*TrainArgs.BatchSize]
            else:
                FileNames = TrainData[NumBatch*TrainArgs.BatchSize:]

            CsiDatas, heatmaps = getSequenceMinibatch(FileNames)
            CsiDatas = Variable(CsiDatas.cuda())
            heatmaps = Variable(heatmaps.cuda())
            heatmaps = heatmaps.permute(1, 0, 2, 3, 4).reshape(-1, 57, 46, 62)

            mask = torch.ones(TrainArgs.BatchSize*TrainArgs.NumFrames, 57, 46, 62).cuda()
            mask = TrainArgs.k * torch.abs(heatmaps) + mask

            PredictJH, PredictPAF = model(CsiDatas)
            prediction = torch.cat((PredictJH, PredictPAF), axis=1)

            loss = torch.sum(torch.mul(mask, criterion_L2(heatmaps, prediction)))
            if (batch_index+1) % TrainArgs.PrintFreq == 0 or (batch_index+1) == NumBatch:
                print("Batch {}/{}\t Loss {:.3f}".format(batch_index+1, NumBatch, loss.item()/TrainArgs.NumFrames/TrainArgs.BatchSize))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        endl = time.time()
        print('Costing time:', (endl-start)/60)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print('Current time:', current_time)

        if not os.path.exists('./checkpoint'):
            os.makedirs('./checkpoint')

        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, TrainArgs.CheckPoint)


def main():

    DatasetName = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:' , ["dataset="])
        for opt, arg in opts:
            if opt in ('-d', '--dataset'):
                DatasetName = arg
            else:
                sys.exit()

    except getopt.GetoptError:
        print ("getopt error!")
        sys.exit()

    assert DatasetName != None, 'python3 TrainDEASFN.py --dataset=SPE/GPE'

    TrainData = LoadDataset(DatasetName)

    train(TrainData)

if __name__ == '__main__':
    main()
