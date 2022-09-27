import os

def runjobs():
  strbaseckpt = "/home/user/model_2022/v220922/"
  strpython = "python -u /home/user/work_2022/AntiSpoofing/trainwpdle.py"

  strseed = 20220406
  strlr = 0.0001
  strgamma = 0.99
  nepoch = 100
  strbsize = 256
  stropti = "adam"
  strresume = ""

  strgpu = 0
  strDB = "Train_4C3_SiW_RECOD_AIHUBx2_CASIA_MSU_OULU_1by1_260x260.db.sort"
  #strresume = "/home/user/model_2022/v220922/Train_4C3_SiW_RECOD_AIHUBx2_CASIA_MSU_OULU_1by1_260x260.db_220926_Xj64ZccCfPgWULrQBbpisa_bsize256_optadam_lr0.0001_gamma_0.99_epochs_100_meta_resnet18_adam_binary_lamda_1.0/epoch_21.ckpt"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strresume)

  strgpu = 1
  strDB = "Train_4C3_SiW_RECOD_AIHUBx2_CASIA_MSU_REPLAY_1by1_260x260.db.sort"
  #strresume = "/home/user/model_2022/v220922/Train_4C3_SiW_RECOD_AIHUBx2_CASIA_MSU_REPLAY_1by1_260x260.db_220926_b8eGgziEr8Gc5de3J4wXGP_bsize256_optadam_lr0.0001_gamma_0.99_epochs_100_meta_resnet18_adam_binary_lamda_1.0/epoch_21.ckpt"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strresume)

  strgpu = 2
  strDB = "Train_4C3_SiW_RECOD_AIHUBx2_CASIA_OULU_REPLAY_1by1_260x260.db.sort"
  #strresume = "/home/user/model_2022/v220922/Train_4C3_SiW_RECOD_AIHUBx2_CASIA_OULU_REPLAY_1by1_260x260.db_220926_eHsnNDjdsz9oPqqXpMkLTL_bsize256_optadam_lr0.0001_gamma_0.99_epochs_100_meta_resnet18_adam_binary_lamda_1.0/epoch_21.ckpt"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strresume)

  strgpu = 3
  strDB = "Train_4C3_SiW_RECOD_AIHUBx2_MSU_OULU_REPLAY_1by1_260x260.db.sort"
  #strresume = "/home/user/model_2022/v220922/Train_4C3_SiW_RECOD_AIHUBx2_MSU_OULU_REPLAY_1by1_260x260.db_220926_e6RAeWF9LV7xnFoGe2PXxX_bsize256_optadam_lr0.0001_gamma_0.99_epochs_100_meta_resnet18_adam_binary_lamda_1.0/epoch_21.ckpt"
  send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strresume)

def send4C4jobs(strpython, strbaseckpt, strDB, stropti, strlr, strgamma, nepoch, strbsize, strgpu, strseed, strresume):
  strmeta = "resnet18_{}_pdle".format(stropti)
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
  strcmd = "{} {} --ckptpath {} --lmdbpath {} --lr {}  --gamma {} --opt {} --epochs {} --batch_size {} --GPU {} --meta {} --random_seed {} --resume {}".format(
    screenoption, strpython, strbaseckpt, lmdbpath, strlr, strgamma, stropti, nepoch, strbsize, strgpu, strmeta, strseed, strresume)
  os.system(strcmd)

if __name__ == '__main__':
  runjobs()
