import os


def sendjobs(dbtype):
  strbaseckpt = "/home/user/model_2022/v220513_02"
  strpython = "python -u /home/user/work_2022/AntiSpoofing/train.py"

  strgamma = 0.92
  nepoch = 81
  strbsize = 512
  strgpu = 2
  stropti = "ADAM"
  strlogoption = "log_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(dbtype,
                                                          stropti,
                                                          "clsloss",
                                                          "opt{}".format(stropti),
                                                          "lr0.005",
                                                          "gamma{}".format(strgamma),
                                                          "e{}".format(nepoch),
                                                          "bsize{}".format(strbsize),
                                                          "gpu{}".format(strgpu))
  strmeta = "clsloss163264_{}".format(stropti)
  screenoption = "screen -L -Logfile {}.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --opt {} --lr 0.005  --gamma {} --epochs {} --batch_size {} --GPU {} --meta {} ".format(
    screenoption, strpython, strbaseckpt, lmdbpath, stropti, strgamma, nepoch, strbsize, strgpu, strmeta)
  os.system(strcmd)



def sendarclossjobs(dbtype):
  strbaseckpt = "/home/user/model_2022/v220513_02"
  strpython = "python -u /home/user/work_2022/AntiSpoofing/train_arcloss.py"

  strgamma = 0.92
  nepoch = 81
  strbsize = 512
  strgpu = 0
  strw1 = 0.0
  stropti = "ADAM"
  strlogoption = "log_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(dbtype,
                                                stropti,
                                                "Arcloss",
                                                "opt{}".format(stropti),
                                                "lr0.005",
                                                "gamma{}".format(strgamma),
                                                "e{}".format(nepoch),
                                                "bsize{}".format(strbsize),
                                                "gpu{}".format(strgpu),
                                                "w{}".format(strw1))
  strmeta = "arcloss163264_w1_{}_{}".format(strw1, stropti)
  screenoption = "screen -L -Logfile {}.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --opt {} --lr 0.005  --gamma {} --epochs {} --batch_size {} --GPU {} --w1 {} --meta {} ".format(
    screenoption, strpython, strbaseckpt, lmdbpath, stropti, strgamma, nepoch, strbsize, strgpu, strw1, strmeta)
  # os.system(strcmd)


  strgpu = 1
  strw1 = 0.1
  stropti = "ADAM"
  strlogoption = "log_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(dbtype,
                                                         stropti,
                                                         "Arcloss",
                                                         "opt{}".format(stropti),
                                                         "lr0.005",
                                                         "gamma{}".format(strgamma),
                                                         "e{}".format(nepoch),
                                                         "bsize{}".format(strbsize),
                                                         "gpu{}".format(strgpu),
                                                         "w{}".format(strw1))
  strmeta = "arcloss163264_w1_{}_{}".format(strw1, stropti)
  screenoption = "screen -L -Logfile {}.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_OULUNPU_{}.db".format(dbtype)
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --opt {} --lr 0.005  --gamma {} --epochs {} --batch_size {} --GPU {} --w1 {} --meta {} ".format(
    screenoption, strpython, strbaseckpt, lmdbpath, stropti, strgamma, nepoch, strbsize, strgpu, strw1, strmeta)
  os.system(strcmd)


if __name__ == '__main__':
  # sendjobs("1by1_260x260")
  sendarclossjobs("1by1_260x260")
