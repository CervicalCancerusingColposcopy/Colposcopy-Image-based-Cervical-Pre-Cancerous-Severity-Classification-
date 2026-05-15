import os
import pandas as pd
from numpy import matlib
from ECO import ECO
from FOA import FOA
from Global_Vars import Global_Vars
from Image_Results import *
from Model_CNN import Model_CNN
from Model_GAN import Model_GAN
from Model_ResNet50 import Model_ResNet50
from Model_ResUnet import Model_ResUnet
from Model_SCBAMA import Model_SCBAMA
from Model_UNetplusplus import Model_UNetplusplus
from Model_Unet3plus import Model_Unet3plus
from Objfun import objfun_1, objfun_2
from Plot_Results import *
from Proposed import Proposed
from SAA import SAA
from SCO import SCO
from UNET_Model import Model_Unet
import warnings

warnings.filterwarnings('ignore')

# Read Dataset
an = 0
if an == 1:
    Image = []
    Path = './Dataset/original'
    out_dir = os.listdir(Path)
    for i in range(len(out_dir)):
        Folder = Path + '/' + out_dir[i]
        in_dir = os.listdir(Folder)
        for j in range(len(in_dir)):
            FileName = Folder + '/' + in_dir[j]
            Img = cv.imread(FileName)
            Resize_Img = cv.resize(Img, (512, 512))
            Image.append(Resize_Img)

    index = np.arange(len(Image))
    np.random.shuffle(index)
    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    np.save('Index.npy', index)
    np.save('Image.npy', Shuffled_Datas)

# Read GroundTruth
an = 0
if an == 1:
    index = np.load('Index.npy', allow_pickle=True)
    Image = []
    Path = './Dataset/cervix'
    out_dir = os.listdir(Path)
    for i in range(len(out_dir)):
        Folder = Path + '/' + out_dir[i]
        in_dir = os.listdir(Folder)
        for j in range(len(in_dir)):
            FileName = Folder + '/' + in_dir[j]
            Img = cv.imread(FileName)
            Resize_Img = cv.resize(Img, (512, 512))
            Image.append(Resize_Img)

    Org_Img = np.asarray(Image)
    Shuffled_Datas = Org_Img[index]
    np.save('GroundTruth.npy', Shuffled_Datas)

# Generate Target
an = 0
if an == 1:
    Tar = []
    Ground_Truth = np.load('GroundTruth.npy', allow_pickle=True)
    for i in range(len(Ground_Truth)):
        image = Ground_Truth[i]
        if np.count_nonzero(image == 255) == 0:
            Tar.append(0)  # Normal
        elif (np.count_nonzero(image == 255) > 100) & (np.count_nonzero(image == 255) <= 2000):
            Tar.append(1)  # low
        else:
            Tar.append(2)  # High

    Target = np.asarray(Tar)
    # unique code
    df = pd.DataFrame(Target)
    uniq = df[0].unique()
    Tar = np.asarray(df[0])
    target = np.zeros((Tar.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Tar == uniq[uni])
        target[index[0], uni] = 1

    np.save('Target.npy', target)

# Image Preprocessing
an = 0
if an == 1:
    Image = np.load('Image.npy', allow_pickle=True)
    Pre_Image = []
    for i in range(len(Image)):
        Img = Image[i]
        gray = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv.threshold(blurred, 220, 255, cv.THRESH_BINARY)
        result_image = cv.inpaint(Img, thresh, 3, cv.INPAINT_TELEA)
        clahe = cv.createCLAHE(clipLimit=5)
        gray_img = cv.cvtColor(result_image, cv.COLOR_BGR2GRAY)
        clahe_img = np.clip(clahe.apply(gray_img) + 30, 0, 255).astype(np.uint8)
        normalized_image = cv.normalize(
            clahe_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        Pre_Img = cv.resize(normalized_image, (512, 512))
        Pre_Image.append(Pre_Img)
    np.save('Pre_Image.npy', np.asarray(Pre_Image))

# Optimization for Segmentation
an = 0
if an == 1:
    Data = np.load('Pre_Image.npy', allow_pickle=True)
    GT = np.load('GroundTruth.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.GT = GT
    Npop = 10
    Chlen = 3  # Hidden Neuron, Learning Rate, Step per Epochs
    xmin = matlib.repmat([5, 0.01, 100], Npop, 1)
    xmax = matlib.repmat([255, 0.99, 500], Npop, 1)
    initsol = np.zeros(xmax.shape)
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = objfun_1
    Max_iter = 50

    print("SCO...")
    [bestfit1, fitness1, bestsol1, time1] = SCO(initsol, fname, xmin, xmax, Max_iter)

    print("SAA...")
    [bestfit4, fitness4, bestsol4, time4] = SAA(initsol, fname, xmin, xmax, Max_iter)

    print("ECO...")
    [bestfit2, fitness2, bestsol2, time2] = ECO(initsol, fname, xmin, xmax, Max_iter)

    print("FOA...")
    [bestfit3, fitness3, bestsol3, time3] = FOA(initsol, fname, xmin, xmax, Max_iter)

    print("Proposed")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('BestSol.npy', BestSol)

# UNET++ Segmentation
an = 0
if an == 1:
    Unet_Path = './UNET/'
    Image_Path = 'Images'
    Mask_Path = 'Mask'
    Predict_Path = 'Predict - Images'
    BestSol = np.load('BestSol.npy', allow_pickle=True)
    Eval = np.zeros((10, 16))
    for j in range(BestSol.shape[0]):
        sol = np.round(BestSol[j, :]).astype(np.int16)
        Eval[j, :], Images1 = Model_UNetplusplus(Unet_Path, Image_Path, Mask_Path, Predict_Path, sol)
    Eval[5, :], Images2 = Model_Unet(Unet_Path, Image_Path, Mask_Path, Predict_Path)
    Eval[6, :], Images3 = Model_Unet3plus(Unet_Path, Image_Path, Mask_Path, Predict_Path)
    Eval[7, :], Images4 = Model_ResUnet(Unet_Path, Image_Path, Mask_Path, Predict_Path)
    Eval[8, :], Images = Model_UNetplusplus(Unet_Path, Image_Path, Mask_Path, Predict_Path)
    Eval[9, :] = Eval[4, :]
    np.save('Evaluate_Seg_all.npy', Eval)
    np.save('Seg_Proposed.npy', Images)

# Optimization for Classification
an = 0
if an == 1:
    Data = np.load('Seg_Proposed.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden neuron count, Learning Rate, Step per Epochs
    xmin = matlib.repmat([5, 0.01, 100], Npop, 1)
    xmax = matlib.repmat([255, 0.99, 500], Npop, 1)
    fname = objfun_2
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("SCO...")
    [bestfit1, fitness1, bestsol1, time1] = SCO(initsol, fname, xmin, xmax, Max_iter)

    print("SAA...")
    [bestfit4, fitness4, bestsol4, time4] = SAA(initsol, fname, xmin, xmax, Max_iter)

    print("ECO...")
    [bestfit2, fitness2, bestsol2, time2] = ECO(initsol, fname, xmin, xmax, Max_iter)

    print("FOA...")
    [bestfit3, fitness3, bestsol3, time3] = FOA(initsol, fname, xmin, xmax, Max_iter)

    print("Proposed")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('Sol.npy', BestSol)

# Classification
an = 0
if an == 1:
    Feat = np.load('Seg_Proposed.npy', allow_pickle=True)  # loading step
    Target = np.load('Target.npy', allow_pickle=True)  # loading step
    BestSol = np.load('Sol.npy', allow_pickle=True)  # loading step
    EVAL = []
    Epochs = [20, 40, 60, 80]
    for act in range(len(Epochs)):
        learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 25))
        for j in range(BestSol.shape[0]):
            print(act, j)
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], pred = Model_SCBAMA(Feat, Target, Epochs[act], sol)  # With optimization
        Eval[5, :], pred1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target,
                                      Epochs[act])  # Model CNN
        Eval[6, :], pred2 = Model_ResNet50(Train_Data, Train_Target, Test_Data,
                                           Test_Target, Epochs[act])  # Model ResNet
        Eval[7, :], pred3 = Model_GAN(Train_Data, Train_Target, Test_Data, Test_Target,
                                      Epochs[act])  # Model GAN
        Eval[8, :], pred4 = Model_SCBAMA(Feat, Target, Epochs[act])  # Without optimization
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Evaluate_all.npy', EVAL)  # Save Eval all

plotConvResults()
Plot_ROC_Curve()
Plots_Results()
Table()
plot_seg_results()
Image_Results()
Sample_Images()
