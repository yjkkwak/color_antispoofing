import glob
import numpy as np
import torch
import time
import os
from scipy import interpolate
from utils import readscore
import matplotlib
import matplotlib.pyplot as plt

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

def drawplotwonlytpr(TPR1, THR1, TPR2, THR2, TPR3, THR3, strtestdb):
  fig = plt.figure()
  plt.subplot(1, 1, 1)# rows, cols, index
  plt.plot(THR1, TPR1, 'r', label='res-s')
  plt.plot(THR2, TPR2, 'r--', label='res-18')
  if THR3 is not None:
    plt.plot(THR3, TPR3, 'r:', label='res-sx2')
  plt.xlabel('THR')
  plt.ylabel('TPR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.tight_layout()
  fig.suptitle(strtestdb)
  fig.subplots_adjust(top=0.88)

  plt.savefig("./{}.png".format(strtestdb))


def drawplotXY(Xs, Ys, Ys2, Xlb, Ylb, Ylb2, clist, lnstyles=['r', 'r--'], lnstyles2=['r', 'r--'], lnlabels=['old','new']):

  # fig = plt.figure(figsize=(12, 7))
  ax = plt.subplot(1, 1, 1)# rows, cols, index

  for idx, _ in enumerate(Xs):
    ax.plot(Xs[idx], Ys[idx], color=clist[idx], linestyle=lnstyles[idx], label="FAR_"+lnlabels[idx])
  ax.set_xlabel(Xlb)
  ax.set_ylabel(Ylb)
  ax2=ax.twinx()

  for idx, _ in enumerate(Xs):
    ax2.plot(Xs[idx], Ys2[idx], color=clist[idx], linestyle=lnstyles2[idx], label="FRR_"+lnlabels[idx])
  ax2.set_ylabel(Ylb2)


  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 0.2])
  ax2.set_ylim([0.0, 0.2])
  ax.legend(loc='upper left', fontsize=10)
  ax2.legend(loc='upper right', fontsize=10)
  # plt.tight_layout()
  # #fig.suptitle(strtestdb)
  #fig.subplots_adjust(top=0.99)
  plt.savefig("./pdle.pdf")
  # plt.show()



def drawplotwvalue(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, strtestdb):
  fig = plt.figure()
  plt.subplot(1, 3, 1)# rows, cols, index
  plt.plot(THR1, FAR1, 'r', label='res-s')
  plt.plot(THR2, FAR2, 'r--', label='res-18')
  plt.xlabel('THR')
  plt.ylabel('FAR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.subplot(1, 3, 2)  # rows, cols, index
  plt.plot(THR1, FRR1, 'g', label='res-s')
  plt.plot(THR2, FRR2, 'g--', label='res-18')
  plt.xlabel('THR')
  plt.ylabel('FRR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.subplot(1, 3, 3)  # rows, cols, index
  plt.plot(THR1, EER1, 'b', label='res-s')
  plt.plot(THR2, EER2, 'b--', label='res-18')
  plt.xlabel('THR')
  plt.ylabel('EER')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.tight_layout()
  fig.suptitle(strtestdb)
  fig.subplots_adjust(top=0.88)

  plt.savefig("./{}.png".format(strtestdb))


def drawplotwvalue3(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, strtestdb):
  fig = plt.figure()
  plt.subplot(1, 3, 1)# rows, cols, index
  plt.plot(THR1, FAR1, 'r', label='res-s')
  plt.plot(THR2, FAR2, 'r--', label='res-18')
  plt.plot(THR3, FAR3, 'r:', label='res-sx2')
  plt.xlabel('THR')
  plt.ylabel('FAR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.subplot(1, 3, 2)  # rows, cols, index
  plt.plot(THR1, FRR1, 'g', label='res-s')
  plt.plot(THR2, FRR2, 'g--', label='res-18')
  plt.plot(THR3, FRR3, 'g:', label='res-sx2')
  plt.xlabel('THR')
  plt.ylabel('FRR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.subplot(1, 3, 3)  # rows, cols, index
  plt.plot(THR1, EER1, 'b', label='res-s')
  plt.plot(THR2, EER2, 'b--', label='res-18')
  plt.plot(THR3, EER3, 'b:', label='res-sx2')
  plt.xlabel('THR')
  plt.ylabel('EER')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.tight_layout()
  fig.suptitle(strtestdb)
  fig.subplots_adjust(top=0.88)

  plt.savefig("./{}.png".format(strtestdb))


def drawplotwvalue4(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, FAR4, FRR4, TPR4, EER4, THR4, strtestdb):
  fig = plt.figure(figsize=(10, 8))

  plt.subplot(1, 3, 1)# rows, cols, index
  plt.plot(THR1, FAR1, 'r', label='res-s w/o oulu')
  plt.plot(THR2, FAR2, 'b', label='res-18 w/o oulu')
  plt.plot(THR3, FAR3, 'r--', label='res-s w/ oulu')
  plt.plot(THR4, FAR4, 'b--', label='res-18 w/ oulu')
  plt.xlabel('THR')
  plt.ylabel('FAR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.subplot(1, 3, 2)  # rows, cols, index
  plt.plot(THR1, FRR1, 'r', label='res-s w/o oulu')
  plt.plot(THR2, FRR2, 'b', label='res-18 w/o oulu')
  plt.plot(THR3, FRR3, 'r--', label='res-s w/ oulu')
  plt.plot(THR4, FRR4, 'b--', label='res-18 w/ oulu')
  plt.xlabel('THR')
  plt.ylabel('FRR')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.subplot(1, 3, 3)  # rows, cols, index
  plt.plot(THR1, EER1, 'r', label='res-s w/o oulu')
  plt.plot(THR2, EER2, 'b', label='res-18 w/o oulu')
  plt.plot(THR3, EER3, 'r--', label='res-s w/ oulu')
  plt.plot(THR4, EER4, 'b--', label='res-18 w/ oulu')
  plt.xlabel('THR')
  plt.ylabel('EER')
  plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  plt.legend(loc='upper right', fontsize=8)

  plt.tight_layout()
  fig.suptitle(strtestdb)
  fig.subplots_adjust(top=0.88)

  plt.savefig("./{}.png".format(strtestdb))

def drawplot4(strmodelpathforamt1, strmodelpathforamt2, strmodelpathforamt3, strmodelpathforamt4, strtestordev, strtestdb):
  FAR1, FRR1, TPR1, EER1, THR1 = getfarfrreer(
    strmodelpathforamt1.format(
      strtestordev, strtestdb))
  FAR2, FRR2, TPR2, EER2, THR2 = getfarfrreer(
    strmodelpathforamt2.format(
      strtestordev, strtestdb))

  FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer(
    strmodelpathforamt3.format(
      strtestordev, strtestdb))

  FAR4, FRR4, TPR4, EER4, THR4 = getfarfrreer(
    strmodelpathforamt4.format(
      strtestordev, strtestdb))

  drawplotwvalue4(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, FAR4, FRR4, TPR4, EER4, THR4, strtestdb)


def drawplot(strmodelpathforamt1, strmodelpathforamt2, strtestordev, strtestdb):
  FAR1, FRR1, TPR1, EER1, THR1 = getfarfrreer(
    strmodelpathforamt1.format(
      strtestordev, strtestdb))
  FAR2, FRR2, TPR2, EER2, THR2 = getfarfrreer(
    strmodelpathforamt2.format(
      strtestordev, strtestdb))

  #ensemble score
  # if "OULU" in strtestdb:
  #   FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer("./../ensemble/Dev_v220419_01_OULUNPU_1by1_260x260Dev_v220419_01_OULUNPU_4by3_244x324.txt")
  #   drawplotwvalue3(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, strtestdb)
  # elif "CelebA" in strtestdb:
  #   FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer("./../ensemble/Test_v220419_01_CelebA_1by1_260x260Test_v220419_01_CelebA_4by3_244x324.txt")
  #   drawplotwvalue3(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, strtestdb)
  # elif "LDRGB" in strtestdb:
  #   FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer("./../ensemble/Test_v220419_01_LDRGB_1by1_260x260Test_v220419_01_LDRGB_4by3_244x324.txt")
  #   drawplotwvalue3(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, strtestdb)
  # elif "LD3007" in strtestdb:
  #   FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer("./../ensemble/Test_v220419_01_LD3007_1by1_260x260Test_v220419_01_LD3007_4by3_244x324.txt")
  #   drawplotwvalue3(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, strtestdb)
  # elif "SiW" in strtestdb:
  #   FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer("./../ensemble/Test_v220419_01_SiW_1by1_260x260Test_v220419_01_SiW_4by3_244x324.txt")
  #   drawplotwvalue3(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, FAR3, FRR3, TPR3, EER3, THR3, strtestdb)
  # else:
  #   drawplotwvalue(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, strtestdb)


  drawplotwvalue(FAR1, FRR1, TPR1, EER1, THR1, FAR2, FRR2, TPR2, EER2, THR2, strtestdb)

def getfarfrreer(scorefile):
  npscore = readscore(scorefile)
  lb = npscore[:, 2]
  livelb = np.where(npscore[:, 2] == 1.0)[0]
  fakelb = np.where(npscore[:, 2] == 0.0)[0]

  """"""
  TPR = []  # TPR = 1 - FRR
  FRR = []  # FRR = 1 - TPR
  FAR = []
  EER = []  # (FAR+FRR) / 2
  THR = []

  thre = np.arange(0.1, 1.0, 0.01)  # Generate an arithmetic list of model thresholds

  class_in = npscore[livelb]
  class_out = npscore[fakelb]

  tmpin = np.where(class_in[:, 1] > 0.5)[0]
  tmpout = np.where(class_out[:, 1] < 0.5)[0]
  acc = (len(tmpin) + len(tmpout)) / len(lb)

  # print (thre)
  for i in range(len(thre)):
    frr = np.sum(class_in[:, 1] < thre[i]) / len(livelb)
    far = np.sum(class_out[:, 1] > thre[i]) / len(class_out)
    tpr = 1.0 - frr
    eer = (frr + far) / 2.0
    FRR.append(frr)
    TPR.append(tpr)
    FAR.append(far)
    EER.append(eer)
    THR.append(thre[i])
    print ("ACC {:0.5} TPR {:0.5} / FAR {:0.5} / EER {:0.5} th {:0.5}".format(acc, tpr, far, eer, thre[i]))
  print ("")
  inter_tprwfar = interpolate.interp1d(FAR, TPR, fill_value='extrapolate')
  print ("TPR {} at FAR 0.1, TPR {} at FAR 0.05, TPR {} at FAR 0.02".format(inter_tprwfar(0.1)*100, inter_tprwfar(0.05)*100, inter_tprwfar(0.02)*100))
  return FAR, FRR, TPR, EER, THR
  # interpolation.. soon

  # inter_tprwthr = interpolate.interp1d(THR, TPR, fill_value='extrapolate')
  # inter_eerwthr = interpolate.interp1d(THR, EER, fill_value='extrapolate')


def gettpronly(scorefile):
  npscore = readscore(scorefile)
  lb = npscore[:, 2]
  livelb = np.where(npscore[:, 2] == 1.0)[0]
  fakelb = np.where(npscore[:, 2] == 0.0)[0]

  """"""
  TPR = []  # TPR = 1 - FRR
  THR = []

  thre = np.arange(0.1, 1.0, 0.01)  # Generate an arithmetic list of model thresholds

  class_in = npscore[livelb]

  tmpin = np.where(class_in[:, 1] > 0.5)[0]

  # print (thre)
  for i in range(len(thre)):
    frr = np.sum(class_in[:, 1] < thre[i]) / len(livelb)
    tpr = 1.0 - frr
    TPR.append(tpr)
    THR.append(thre[i])

  return TPR, THR

def rundrawplot_1014():
  # old
  strmodelpathforamt1_old = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_1by1_260x260_220510_XWtdsCV5xfQ28a8PLyYYke_lr0.005_gamma_0.92_epochs_80_meta_163264/Test_4C0_RECOD_1by1_260x260.db/72.score"
  # strmodelpathforamt1_old = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_4by3_244x324_220504_UNnaHEdifqijaML6w6uS3W_lr0.005_gamma_0.92_epochs_80_meta_163264/Test_4C0_RECOD_4by3_244x324.db/65.score"
  #
  #
  strmodelpathforamt1_newbase = "/home/user/model_2022/v220922/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db_221010_ifUcCDxJv533hBgeYAvV3C_bsize256_optadam_lr0.0001_gamma_0.99_epochs_80_meta_woCW_resnet18_adam_binary_lamda_1.0/Test_4C0_RECOD_1by1_260x260.db/18.score"
  # strmodelpathforamt1_newbase = "/home/user/model_2022/v220922/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_4by3_244x324.db_221010_LoDzPYZubka3wyJej2JUBA_bsize256_optadam_lr0.0001_gamma_0.99_epochs_80_meta_woCW_resnet18_adam_binary_lamda_1.0/Test_4C0_RECOD_4by3_244x324.db/21.score"

  # strmodelpathforamt1_pdle = "/home/user/model_2022/v220922_pdle/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db_221011_YDWLULy5cKpPRHbzGSdFNk_bsize128_optadam_lr1e-05_gamma_0.99_epochs_100_meta_woCW_resnet18_adam_pdle_lamda_1.0/Test_4C0_RECOD_1by1_260x260.db/51.score"
  strmodelpathforamt1_pdle = "/home/user/model_2022/v220922_pdle/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db_221014_7Ka6RguFhyzCYsamxtZZw7_bsize128_optadam_lr1e-05_gamma_0.99_epochs_100_meta_woCW_resnet18_adam_pdle_lamda_1.0/Test_4C0_RECOD_1by1_260x260.db/45.score"
  #
  # strmodelpathforamt1_pdle = "/home/user/model_2022/v220922_pdle/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_4by3_244x324.db_221011_3692NUKAZCwmQbeLCotFE5_bsize128_optadam_lr1e-05_gamma_0.99_epochs_100_meta_woCW_resnet18_adam_pdle_lamda_1.0/Test_4C0_RECOD_4by3_244x324.db/78.score"

  FAR1, FRR1, TPR1, EER1, THR1 = getfarfrreer(strmodelpathforamt1_old)
  FAR2, FRR2, TPR2, EER2, THR2 = getfarfrreer(strmodelpathforamt1_newbase)
  FAR3, FRR3, TPR3, EER3, THR3 = getfarfrreer(strmodelpathforamt1_pdle)

  # CB91_Blue = '#2CBDFE'
  # CB91_Green = '#47DBCD'
  # CB91_Pink = '#F3A0F2'
  # CB91_Purple = '#9D2EC5'
  # CB91_Violet = '#661D98'
  # CB91_Amber = '#F5B14C'

  drawplotXY([THR1, THR2, THR3], [FAR1, FAR2, FAR3], [FRR1, FRR2, FRR3], 'THR', 'FAR', 'FRR', [CB91_Blue,CB91_Pink, CB91_Amber], lnstyles=['-', '-', '-'], lnstyles2=['--', '--', '--'], lnlabels=['release_06','baseline_10', 'pdle'])


def rundrawplot2():
  strmodelpathforamt1 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_3uKCX7S9pwbeSTzoTydcgV_lr0.005_gamma_0.92_epochs_80_meta_163264/{}_v220419_01_{}/78.score"
  strmodelpathforamt2 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_Pn2ww7BGgZGmhJD5oeG2L6_lr0.005_gamma_0.92_epochs_80_meta_baselineres18/{}_v220419_01_{}/78.score"
  strmodelpathforamt3 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_1by1_260x260_220510_XWtdsCV5xfQ28a8PLyYYke_lr0.005_gamma_0.92_epochs_80_meta_163264/{}_v220419_01_{}/72.score"
  strmodelpathforamt4 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_1by1_260x260_220510_KHi4YQxF4Qx9S6XayeRBkx_lr0.005_gamma_0.92_epochs_80_meta_baselineres18/{}_v220419_01_{}/73.score"
  drawplot4(strmodelpathforamt1, strmodelpathforamt2, strmodelpathforamt3, strmodelpathforamt4, "Dev", "OULUNPU_1by1_260x260")

def rundrawplot():
  # strmodelpathforamt1 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_3uKCX7S9pwbeSTzoTydcgV_lr0.005_gamma_0.92_epochs_80_meta_163264/{}_v220419_01_{}/78.score"
  # strmodelpathforamt2 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_Pn2ww7BGgZGmhJD5oeG2L6_lr0.005_gamma_0.92_epochs_80_meta_baselineres18/{}_v220419_01_{}/78.score"

  strmodelpathforamt1 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_1by1_260x260_220510_XWtdsCV5xfQ28a8PLyYYke_lr0.005_gamma_0.92_epochs_80_meta_163264/{}_v220419_01_{}/72.score"
  strmodelpathforamt2 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_1by1_260x260_220510_KHi4YQxF4Qx9S6XayeRBkx_lr0.005_gamma_0.92_epochs_80_meta_baselineres18/{}_v220419_01_{}/73.score"

  TPR1, THR1 = gettpronly(
    strmodelpathforamt1.format(
      "Test", "Emotion_1by1_260x260"))

  TPR2, THR2 = gettpronly(
    strmodelpathforamt2.format(
      "Test", "Emotion_1by1_260x260"))

  # TPR3, THR3 = gettpronly("./../ensemble/Test_v220419_01_Emotion_1by1_260x260Test_v220419_01_Emotion_4by3_244x324.txt")
  # drawplotwonlytpr(TPR1, THR1, TPR2, THR2, TPR3, THR3, "Emotion_1by1_260x260")

  drawplotwonlytpr(TPR1, THR1, TPR2, THR2, None, None, "Emotion_1by1_260x260")

  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "SiW_1by1_260x260")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "LDRGB_1by1_260x260")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "LD3007_1by1_260x260")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "CelebA_1by1_260x260")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Dev", "OULUNPU_1by1_260x260")
  
  return

  strmodelpathforamt1 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_4by3_244x324_220504_eNeMv72oynyYhUikgY4mbv_lr0.001_gamma_0.92_epochs_80_meta_163264/{}_v220419_01_{}/69.score"
  strmodelpathforamt2 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_4by3_244x324_220505_edzapQSW8VwscSyfxJjcZr_lr0.005_gamma_0.92_epochs_80_meta_baselineres18/{}_v220419_01_{}/61.score"

  TPR1, THR1 = gettpronly(
    strmodelpathforamt1.format(
      "Test", "Emotion_4by3_244x324"))

  TPR2, THR2 = gettpronly(
    strmodelpathforamt2.format(
      "Test", "Emotion_4by3_244x324"))

  TPR3, THR3 = gettpronly("./../ensemble/Test_v220419_01_Emotion_1by1_260x260Test_v220419_01_Emotion_4by3_244x324.txt")
  drawplotwonlytpr(TPR1, THR1, TPR2, THR2, TPR3, THR3, "Emotion_4by3_244x324")


  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "SiW_4by3_244x324")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "LDRGB_4by3_244x324")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "LD3007_4by3_244x324")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Test", "CelebA_4by3_244x324")
  drawplot(strmodelpathforamt1, strmodelpathforamt2, "Dev", "OULUNPU_4by3_244x324")

if __name__ == '__main__':
  #rundrawplot()
  # rundrawplot2()
  rundrawplot_1014()