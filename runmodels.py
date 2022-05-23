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
  strbaseckpt = "/home/user/model_2022/v220419_02/"
  strpython = "python -u /home/user/work_2022/AntiSpoofing/train_arcloss.py"

  strgamma = 0.90
  nepoch=81
  strbsize = 512
  strgpu = 2
  strw1 = 0.0
  stropti = "Adam"
  strlogoption = "log_{}_{}_{}_{}_{}_{}_{}_{}".format(dbtype,
                                                stropti,
                                                "Arcloss",
                                                "lr0.005",
                                                "gamma{}".format(strgamma),
                                                "e{}".format(nepoch),
                                                "bsize{}".format(strbsize),
                                                "gpu{}".format(strgpu),
                                                "w{}".format(strw1))
  strmeta = "arcloss163264_w1_{}_{}".format(strw1, stropti)
  screenoption = "screen -L -Logfile {}.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --lr 0.005  --gamma {} --epochs {} --batch_size {} --GPU {} --w1 {} --meta {} ".format(
    screenoption, strpython, strbaseckpt, lmdbpath, strgamma, nepoch, strbsize, strgpu, strw1, strmeta)
  os.system(strcmd)


  strgpu = 3
  strw1 = 1.0
  stropti = "Adam"
  strlogoption = "log_{}_{}_{}_{}_{}_{}_{}_{}".format(dbtype,
                                                stropti,
                                                "Arcloss",
                                                "lr0.005",
                                                "gamma{}".format(strgamma),
                                                "e{}".format(nepoch),
                                                "bsize{}".format(strbsize),
                                                "gpu{}".format(strgpu),
                                                "w{}".format(strw1))
  strmeta = "arcloss163264_w1_{}_{}".format(strw1, stropti)
  screenoption = "screen -L -Logfile {}.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --lr 0.005  --gamma {} --epochs {} --batch_size {} --GPU {} --w1 {} --meta {} ".format(
    screenoption, strpython, strbaseckpt, lmdbpath, strgamma, nepoch, strbsize, strgpu, strw1, strmeta)
  os.system(strcmd)


if __name__ == '__main__':
  # sendjobs("1by1_260x260")
  sendarclossjobs("1by1_260x260")
