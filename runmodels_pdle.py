import os

def runjobs():
  strbaseckpt = "/home/user/model_2022/v220922_pdle/"
  strpython = "python -u /home/user/work_2022/AntiSpoofing/trainwpdle.py"

  # strseed = 20220406

  strgamma = 0.99
  nepoch = 50
  strbsize = 128//2
  stropti = "adam"
  strresume = ""

  strseed = 20220407
  strgpu = 0
  strlr = 0.00001
  strDB = "Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db.sort"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strresume)

  strseed = 20220408
  strgpu = 1
  strlr = 0.00001
  strDB = "Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db.sort"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strresume)
  return
  strseed = 20220409
  strgpu = 2
  strlr = 0.00001
  strDB = "Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db.sort"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strresume)

  strgpu = 3
  strlr = 0.000011
  strDB = "Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db.sort"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strresume)



def send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strresume):
  strmeta = "woCW_resnet18_{}_pdle".format(stropti)
  strlogoption = "log_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(strDB,
                                                      stropti,
                                                      "pdle",
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
    screenoption, strpython, strbaseckpt, lmdbpath, strlr, strgamma, stropti, nepoch, strbsize, strgpu, strmeta, strseed, strresume)
  os.system(strcmd)

if __name__ == '__main__':
  runjobs()
