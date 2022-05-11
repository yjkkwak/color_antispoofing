import os



def sendtestbaseline(dbtype):
  strpython = "python -u /home/user/work_2022/AntiSpoofing/train_baseline_resnet18pretrain.py"
  strlogoption = "testlog_{}_gamma092epoch80metabaselineresent18".format(dbtype)
  ###

  # screenoption = "screen -L -Logfile {}{}{}gpu0.txt -d -m ".format(strlogoption, "CelebA_SiW_LDRGB_LD3007", "test0")
  # lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_{}.db".format(dbtype)
  # strcmd = "{} {} --ckptpath /home/user/model_2022/test --lmdbpath {} --gamma 0.92 --epochs 80 --GPU 0 --batch_size 256 --meta test0baselineres18".format(screenoption, strpython, lmdbpath)
  # os.system(strcmd)

  screenoption = "screen -L -Logfile {}{}{}gpu1.txt -d -m ".format(strlogoption, "CelebA_SiW_LDRGB_LD3007", "test1")
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_{}.db".format(dbtype)
  strcmd = "{} {} --ckptpath /home/user/model_2022/test --lmdbpath {} --gamma 0.92 --epochs 80 --GPU 1 --batch_size 256 --meta test1baselineres18".format(screenoption, strpython, lmdbpath)
  os.system(strcmd)

  screenoption = "screen -L -Logfile {}{}{}gpu2.txt -d -m ".format(strlogoption, "CelebA_SiW_LDRGB_LD3007", "test2")
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_{}.db".format(dbtype)
  strcmd = "{} {} --ckptpath /home/user/model_2022/test --lmdbpath {} --gamma 0.92 --epochs 80 --GPU 2 --batch_size 256 --meta test1baselineres18".format(screenoption, strpython, lmdbpath)
  os.system(strcmd)

  screenoption = "screen -L -Logfile {}{}{}gpu3.txt -d -m ".format(strlogoption, "CelebA_SiW_LDRGB_LD3007", "test3")
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_{}.db".format(dbtype)
  strcmd = "{} {} --ckptpath /home/user/model_2022/test --lmdbpath {} --gamma 0.92 --epochs 80 --GPU 3 --batch_size 256 --meta test3baselineres18".format(screenoption,
                                                                                                   strpython, lmdbpath)
  os.system(strcmd)


def sendbaseline(dbtype):
  strpython = "python -u /home/user/work_2022/AntiSpoofing/train_baseline_resnet18pretrain.py"
  strlogoption = "log_{}_gamma092epoch80metabaselineresent18".format(dbtype)
  ###
  screenoption = "screen -L -Logfile {}{}{}gpu0.txt -d -m ".format(strlogoption, "CelebA_SiW_LDRGB_LD3007_OULUNPU", "_lr0.005")
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --lmdbpath {} --gamma 0.92 --epochs 80 --GPU 0 --meta baselineres18 --lr 0.005 --resume /home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_1by1_260x260_220510_KHi4YQxF4Qx9S6XayeRBkx_lr0.005_gamma_0.92_epochs_80_meta_baselineres18/epoch_60.ckpt".format(screenoption, strpython, lmdbpath)
  os.system(strcmd)

def sendjobs(dbtype):
  strpython = "python -u /home/user/work_2022/AntiSpoofing/train.py"
  strlogoption = "log_{}_gamma092epoch80meta163264".format(dbtype)
  ###
  screenoption = "screen -L -Logfile {}gpu0.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_LDRGB_LD3007_{}.db".format(dbtype)
  strcmd = "{} {} --lmdbpath {} --gamma 0.92 --epochs 80 --meta 163264 --GPU 0".format(screenoption, strpython, lmdbpath)
  # os.system(strcmd)
  ###

  ###
  screenoption = "screen -L -Logfile {}gpu1.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LD3007_{}.db".format(dbtype)
  strcmd = "{} {} --lmdbpath {} --gamma 0.92 --epochs 80 --meta 163264 --GPU 1".format(screenoption, strpython, lmdbpath)
  # os.system(strcmd)
  #

  ###
  screenoption = "screen -L -Logfile {}gpu2.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_{}.db".format(dbtype)
  strcmd = "{} {} --lmdbpath {} --gamma 0.92 --epochs 80 --meta 163264 --GPU 2".format(screenoption, strpython, lmdbpath)
  # os.system(strcmd)
  #

  ###
  screenoption = "screen -L -Logfile {}gpu3.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_SiW_LDRGB_LD3007_{}.db".format(dbtype)
  strcmd = "{} {} --lmdbpath {} --gamma 0.92 --epochs 80 --meta 163264 --GPU 3".format(screenoption, strpython, lmdbpath)
  # os.system(strcmd)
  #
#

  ######
  screenoption = "screen -L -Logfile {}{}{}gpu1.txt -d -m ".format(strlogoption, "CelebA_SiW_LDRGB_LD3007_OULUNPU", "_lr0.001")
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --lmdbpath {} --gamma 0.92 --epochs 80 --meta 163264 --GPU 1 --lr 0.001".format(screenoption, strpython, lmdbpath)
  os.system(strcmd)

  screenoption = "screen -L -Logfile {}{}{}gpu3.txt -d -m ".format(strlogoption, "CelebA_SiW_LDRGB_LD3007_OULUNPU", "_lr0.005")
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --lmdbpath {} --gamma 0.92 --epochs 80 --meta 163264 --GPU 3 --lr 0.005".format(screenoption,
                                                                                                 strpython, lmdbpath)
  os.system(strcmd)

  #
if __name__ == '__main__':
  #1by1_260x260
  # sendjobs("4by3_244x324")
  # sendjobs("1by1_260x260")

  sendbaseline("1by1_260x260")
  # sendbaseline("4by3_244x324")
  # sendtestbaseline("1by1_260x260")
