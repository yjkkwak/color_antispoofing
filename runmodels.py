import os

def sendbaseline(dbtype):
  strpython = "python -u /home/user/work_2022/AntiSpoofing/train_baseline_resnet18pretrain.py"
  strlogoption = "log_{}_gamma092epoch80metabaselineresent18".format(dbtype)
  ###
  screenoption = "screen -L -Logfile {}{}{}gpu3.txt -d -m ".format(strlogoption, "CelebA_SiW_LDRGB_LD3007", "_lr0.005")
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_{}.db".format(dbtype)
  strcmd = "{} {} --lmdbpath {} --gamma 0.92 --epochs 80 --GPU 3 --meta baselineres18 --lr 0.005".format(screenoption, strpython, lmdbpath)
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

#
  screenoption = "screen -L -Logfile {}{}{}gpu1.txt -d -m ".format(strlogoption, "CelebA_SiW_LDRGB_LD3007", "_lr0.005")
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_{}.db".format(dbtype)
  strcmd = "{} {} --lmdbpath {} --gamma 0.92 --epochs 80 --meta 163264 --GPU 1 --lr 0.005".format(screenoption, strpython, lmdbpath)
  os.system(strcmd)
  #
if __name__ == '__main__':
  #1by1_260x260
  # sendjobs("4by3_244x324")
  # sendjobs("1by1_260x260")

  sendbaseline("1by1_260x260")
