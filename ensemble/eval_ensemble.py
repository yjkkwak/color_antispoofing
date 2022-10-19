from utils import genfarfrreerwth, genfarfrreerwthlist, genfarfrreer
from eval.performance import ssan_performances_val

def main():
  print ("abc")
  getbasemodels()

def getscorewsortbypath(strsocrepath):
  the_file = open(strsocrepath, "r")
  strlines = the_file.readlines()
  scorelist = []
  for strline in strlines:
    strtokens = strline.split()
    scorelist.append([strtokens[0], float(strtokens[1]), float(strtokens[2]), strtokens[3]])
  the_file.close()
  scorelist = sorted(scorelist, key = lambda x:x[3])
  return scorelist

def ensemble_scores(spath1,spath2, spath3, oname="tmp_ensmeble"):
  # spath1 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_1by1_260x260_220502_3uKCX7S9pwbeSTzoTydcgV_lr0.005_gamma_0.92_epochs_80_meta_163264/{}/78.score".format(db1)
  # spath2 = "/home/user/model_2022/v220419_01/Train_v220419_01_CelebA_SiW_LDRGB_LD3007_4by3_244x324_220504_eNeMv72oynyYhUikgY4mbv_lr0.001_gamma_0.92_epochs_80_meta_163264/{}/69.score".format(db2)

  scorelist1 = getscorewsortbypath(spath1)
  scorelist2 = getscorewsortbypath(spath2)
  scorelist3 = getscorewsortbypath(spath3)

  wlist = [0.2, 0.2, 0.6]
  the_file = open("./{}.txt".format(oname), "w")
  for item in zip(scorelist1, scorelist2, scorelist3):
    fakescore = wlist[0]*float(item[0][1]) + wlist[1]*float(item[1][1]) + wlist[2]*float(item[2][1])
    livescore = wlist[0] * float(item[0][2]) + wlist[1] * float(item[1][2]) + wlist[2] * float(item[2][2])
    the_file.write("{} {} {} {}\n".format(item[0][0], fakescore, livescore, item[0][2]))
  the_file.close()

  genfarfrreerwthlist(spath1)
  genfarfrreerwthlist(spath2)
  genfarfrreerwthlist(spath3)
  genfarfrreerwthlist("./{}.txt".format(oname))
  ssan_performances_val("./{}.txt".format(oname))



def getbasemodels():
  # ensemble_scores("Test_v220419_01_CelebA_1by1_260x260", "Test_v220419_01_CelebA_4by3_244x324")
  # ensemble_scores("Test_v220419_01_LD3007_1by1_260x260", "Test_v220419_01_LD3007_4by3_244x324")
  # ensemble_scores("Test_v220419_01_LDRGB_1by1_260x260", "Test_v220419_01_LDRGB_4by3_244x324")
  #ensemble_scores("Test_v220419_01_SiW_1by1_260x260", "Test_v220419_01_SiW_4by3_244x324")
  #ensemble_scores("Dev_v220419_01_OULUNPU_1by1_260x260", "Dev_v220419_01_OULUNPU_4by3_244x324")

  #ensemble_scores("Test_v220419_01_Emotion_1by1_260x260", "Test_v220419_01_Emotion_4by3_244x324")

  strmodelpathforamt1_newbase = "/home/user/model_2022/v220922/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db_221010_ifUcCDxJv533hBgeYAvV3C_bsize256_optadam_lr0.0001_gamma_0.99_epochs_80_meta_woCW_resnet18_adam_binary_lamda_1.0/Test_4C0_RECOD_1by1_260x260.db/18.score"
  strmodelpathforamt1_newbase2 = "/home/user/model_2022/v220922/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_4by3_244x324.db_221010_LoDzPYZubka3wyJej2JUBA_bsize256_optadam_lr0.0001_gamma_0.99_epochs_80_meta_woCW_resnet18_adam_binary_lamda_1.0/Test_4C0_RECOD_4by3_244x324.db/21.score"

  # strmodelpathforamt1_pdle = "/home/user/model_2022/v220922_pdle/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db_221011_YDWLULy5cKpPRHbzGSdFNk_bsize128_optadam_lr1e-05_gamma_0.99_epochs_100_meta_woCW_resnet18_adam_pdle_lamda_1.0/Test_4C0_RECOD_1by1_260x260.db/51.score"
  strmodelpathforamt1_pdle = "/home/user/model_2022/v220922_pdle/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db_221014_7Ka6RguFhyzCYsamxtZZw7_bsize128_optadam_lr1e-05_gamma_0.99_epochs_100_meta_woCW_resnet18_adam_pdle_lamda_1.0/Test_4C0_RECOD_1by1_260x260.db/45.score"


  ensemble_scores(strmodelpathforamt1_newbase,strmodelpathforamt1_newbase2,
                  strmodelpathforamt1_pdle)


if __name__ == '__main__':
  main()
