import numpy as np
from Evaluation import evaluation, net_evaluation
from Global_Vars import Global_Vars
from Model_SCBAMA import Model_SCBAMA
from Model_UNetplusplus import Model_UNetplusplus


def objfun_1(Soln):
    Unet_Path = './UNET/'
    Image_Path = 'Images'
    Mask_Path = 'Mask'
    Predict_Path = 'Predict - Images'
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            Images, results = Model_UNetplusplus(Unet_Path, Image_Path, Mask_Path, Predict_Path, sol)
            Eval = net_evaluation(Images, results)
            Fitn[i] = 1 / Eval[5]
        return Fitn
    else:
        sol = Soln
        Images, results = Model_UNetplusplus(Unet_Path, Image_Path, Mask_Path, Predict_Path, sol)
        Eval = net_evaluation(Images, results)
        Fitn = 1 / Eval[5]
        return Fitn


def objfun_2(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, pred = Model_SCBAMA(data, Tar, sol)
            Eval = evaluation(Tar, pred)
            Fitn[i] = (1 / Eval[7]) + Eval[14]
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, pred = Model_SCBAMA(data, Tar, sol)
        Eval = evaluation(Tar, pred)
        Fitn = (1 / Eval[7]) + Eval[14]
        return Fitn
