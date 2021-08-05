import numpy as np
import torch
from torch.autograd import Variable
import hdf5storage
import os
from model.Deasfn import Deasfn
from config.args import TestArgs
from tqdm import tqdm
import shutil
import sys , getopt

os.environ["CUDA_VISIBLE_DEVICES"] = TestArgs.CudaDevice


def test(DatasetName):

    model = Deasfn()
    checkpoint = torch.load(TestArgs.CheckPoint)

    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda().eval()

    if os.path.exists('predict/'):
        shutil.rmtree('predict/')

    if DatasetName == 'SPE':
        for action, TestRange in TestArgs.SpeInfo:
            if not os.path.exists('./predict/' + action):
                os.makedirs('./predict/' + action)
            for subject in tqdm(TestArgs.subjects, desc=action):
                for r in TestRange:
                    for index in r:
                        csi_data = torch.zeros(1, TestArgs.NumFrames, 30, 25, 3, 3)
                        for j in range(TestArgs.NumFrames):
                            data = hdf5storage.loadmat(TestArgs.DataPath +'SPE/'+action+'/'+subject+'/train/' + str(index - (TestArgs.NumFrames+1+j) * TestArgs.DilatedRate) + '.mat', variable_names={'csi_serial'})
                            csi_data[0, j, :, :, :, :] = torch.from_numpy(data['csi_serial']).type(torch.FloatTensor).permute(1, 0, 2, 3)

                        csi_data = Variable(csi_data.cuda())
                        PredictJH, PredictPAF = model(csi_data)
                        PredictJH = PredictJH.cpu().detach().numpy()
                        PredictPAF = PredictPAF.cpu().detach().numpy()
                        Prediction = np.concatenate((PredictJH[-1], PredictPAF[-1]))
                        np.save('predict/'+action+'/pred_'+subject+'_'+action+'_' + str(index) , Prediction)

    elif DatasetName == 'GPE':
        for path in TestArgs.GpeInfo:
            if not os.path.exists('./predict/' + path[0]):
                os.makedirs('./predict/' + path[0])
            for index in tqdm(path[1], desc=path[0]):
                csi_data = torch.zeros(1, TestArgs.NumFrames, 30, 25, 3, 3)
                for j in range(TestArgs.NumFrames):
                    data = hdf5storage.loadmat(TestArgs.DataPath + 'GPE/' + path[0] + '/train/' + str(index) + '.mat', variable_names={'csi_serial'})
                    csi_data[0, j, :, :, :, :] = torch.from_numpy(data['csi_serial']).type(torch.FloatTensor).permute(1, 0, 2, 3)

                csi_data = Variable(csi_data.cuda())
                pred_JH_tensor, pred_PAF_tensor = model(csi_data)
                pred_JH = pred_JH_tensor.cpu().detach().numpy()
                pred_PAF = pred_PAF_tensor.cpu().detach().numpy()
                pred_heatmaps = np.concatenate((pred_JH[-1], pred_PAF[-1]))
                np.save('predict/' + path[0] + '/pred_' + str(index) , pred_heatmaps)

    else:
        print("Dataset name should be SPE or GPE")
        sys.exit()


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

    assert DatasetName != None, 'python3 TestDEASFN.py --dataset=SPE/GPE'

    test(DatasetName)


if __name__ == '__main__':
    main()
