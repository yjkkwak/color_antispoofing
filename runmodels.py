import os

def runjobs():
  strbaseckpt = "/home/user/model_2022/v220922/"
  strpython = "python -u /home/user/work_2022/AntiSpoofing/train.py"

  strseed = 20220406
  strlr = 0.0001
  strgamma = 0.99
  nepoch = 100
  strbsize = 256
  stropti = "adam"

  strgpu = 0
  strDB = "Train_4C3_SiW_RECOD_AIHUBx2_CASIA_MSU_OULU_1by1_260x260.db.sort"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  strgpu = 1
  strDB = "Train_4C3_SiW_RECOD_AIHUBx2_CASIA_MSU_REPLAY_1by1_260x260.db.sort"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  strgpu = 2
  strDB = "Train_4C3_SiW_RECOD_AIHUBx2_CASIA_OULU_REPLAY_1by1_260x260.db.sort"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

  strgpu = 3
  strDB = "Train_4C3_SiW_RECOD_AIHUBx2_MSU_OULU_REPLAY_1by1_260x260.db.sort"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed)

def send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed):
  strmeta = "resnet18_{}_binary".format(stropti)
  strlogoption = "log_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(strDB,
                                                      stropti,
                                                      "bcls",
                                                      "lr{}".format(strlr),
                                                      "gamma{}".format(strgamma),
                                                      "e{}".format(nepoch),
                                                      "bsize{}".format(strbsize),
                                                      "gpu{}".format(strgpu),
                                                      "meta{}".format(strmeta),
                                                      "seed{}".format(strseed))
  screenoption = "screen -L -Logfile {}.txt -d -m ".format(strlogoption)
  lmdbpath = "/home/user/work_db/v220922/{}".format(strDB)
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --lr {}  --gamma {} --opt {} --epochs {} --batch_size {} --GPU {} --meta {} --random_seed {}".format(
    screenoption, strpython, strbaseckpt, lmdbpath, strlr, strgamma, stropti, nepoch, strbsize, strgpu, strmeta, strseed)
  os.system(strcmd)

if __name__ == '__main__':
  runjobs()
