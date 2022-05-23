import os


def sendjobs(dbtype):
  strbaseckpt = "/home/user/model_2022/v220513_01/"
  strpython = "python -u /home/user/work_2022/AntiSpoofing/train.py"
  strlogoption = "log_{}_baselinelossmeta163264".format(dbtype)

  nepoch=16
  screenoption = "screen -L -Logfile {}{}{}{}againgpu0.txt -d -m ".format(strlogoption, "baseline_OULUNPU", "_lr0.005again", "_e{}_bsize{}".format(nepoch, 256))
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --batch_size 256 --ckptpath {} --lmdbpath {} --gamma 0.88 --epochs {} --meta 163264again --GPU 0 --lr 0.005".format(screenoption,
                                                                                                 strpython, strbaseckpt, lmdbpath, nepoch)
  os.system(strcmd)



def sendarclossjobs(dbtype):
  strbaseckpt = "/home/user/model_2022/v220513_01/"
  strpython = "python -u /home/user/work_2022/AntiSpoofing/train_arcloss.py"
  strlogoption = "log_{}_arclossmeta163264".format(dbtype)

  nepoch=31
  screenoption = "screen -L -Logfile {}{}{}{}gpu0_w0.0.txt -d -m ".format(strlogoption, "Arcloss_OULUNPU", "_lr0.005", "_e{}_bsize{}".format(nepoch, 512))
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --batch_size 512 --ckptpath {} --lmdbpath {} --gamma 0.90 --epochs {} --meta arcloss163264_w0.0 --GPU 0 --lr 0.005 --w1 0.0".format(screenoption,
                                                                                                 strpython, strbaseckpt, lmdbpath, nepoch)
  os.system(strcmd)

  screenoption = "screen -L -Logfile {}{}{}{}gpu1_w0.1.txt -d -m ".format(strlogoption, "Arcloss_OULUNPU", "_lr0.005", "_e{}_bsize{}".format(nepoch, 512))
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --batch_size 512 --ckptpath {} --lmdbpath {} --gamma 0.90 --epochs {} --meta arcloss163264_w0.1 --GPU 1 --lr 0.005 --w1 0.2".format(screenoption,
                                                                                                 strpython, strbaseckpt, lmdbpath, nepoch)
  os.system(strcmd)

  screenoption = "screen -L -Logfile {}{}{}{}gpu2_w0.5.txt -d -m ".format(strlogoption, "Arcloss_OULUNPU", "_lr0.005", "_e{}_bsize{}".format(nepoch, 512))
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --batch_size 512 --ckptpath {} --lmdbpath {} --gamma 0.90 --epochs {} --meta arcloss163264_w0.5 --GPU 2 --lr 0.005 --w1 0.5".format(
    screenoption,
    strpython, strbaseckpt, lmdbpath, nepoch)
  os.system(strcmd)

  screenoption = "screen -L -Logfile {}{}{}{}gpu3_w1.0.txt -d -m ".format(strlogoption, "Arcloss_OULUNPU", "_lr0.005", "_e{}_bsize{}".format(nepoch, 512))
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --batch_size 512 --ckptpath {} --lmdbpath {} --gamma 0.90 --epochs {} --meta arcloss163264_w1.0 --GPU 3 --lr 0.005 --w1 1.0".format(
    screenoption,
    strpython, strbaseckpt, lmdbpath, nepoch)
  os.system(strcmd)

if __name__ == '__main__':
  sendjobs("1by1_260x260")
  # sendarclossjobs("1by1_260x260")
