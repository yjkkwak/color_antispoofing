from utils import genfarfrreerwth, genfarfrreerwthlist

def main():
  print ("abc")
  getbasemodels()

def getscorewsortbypath(strsocrepath):
  the_file = open(strsocrepath, "r")
  strlines = the_file.readlines()
  scorelist = []
  for strline in strlines:
    strtokens = strline.split()
    scorelist.append([float(strtokens[0]), float(strtokens[1]), strtokens[2]])
  the_file.close()
  scorelist = sorted(scorelist, key = lambda x:x[2])
  return scorelist

def ensemble_scores():

  #spath1 = "/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_1by1_260x260_220414_HDZCuMsB2eriabbcwYkRC5_lr0.01_gamma_0.92_epochs_80_meta_163264/Test_v220401_01_Emotion_1by1_260x260/68.score"
  spath1 = "/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_4by3_244x324_220415_9JK2EGmzAk4hgnseEPZ8Ck_lr0.01_gamma_0.92_epochs_80_meta_163264/Test_v220401_01_Emotion_4by3_244x324/79.score"
  spath2 = "/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_4by3_244x324_220415_9JK2EGmzAk4hgnseEPZ8Ck_lr0.01_gamma_0.92_epochs_80_meta_163264/Test_v220401_01_Emotion_4by3_244x324/72.score"
  scorelist1 = getscorewsortbypath(spath1)
  scorelist2 = getscorewsortbypath(spath2)

  wsum_score = []
  wlist = [0.3, 0.7]
  the_file = open("./tmpfile.txt", "w")
  for item in zip(scorelist1, scorelist2):
    fakescore = wlist[0]*float(item[0][0]) + wlist[1]*float(item[1][0])
    livescore = wlist[0] * float(item[0][1]) + wlist[1] * float(item[1][1])
    imgpath = item[0][2]
    the_file.write("{} {} {}\n".format(fakescore, livescore, imgpath))
  the_file.close()

  genfarfrreerwthlist(spath1)
  genfarfrreerwthlist(spath2)
  genfarfrreerwthlist("./tmpfile.txt")




def getbasemodels():
  ensemble_scores()
  return
  baesdblist = ["CelebA_LDRGB_LD3007",
            #"CelebA_SiW_LD3007",
            #"CelebA_SiW_LDRGB",
            #"SiW_LDRGB_LD3007",
                ]

  basepatchlist = ["1by1_260x260",
               "4by3_244x324"]

  basemodellist = []
  for basedb in baesdblist:
    for basepatch in basepatchlist:
      basemodellist.append("{}_{}".format(basedb, basepatch))
  print (basemodellist)


if __name__ == '__main__':
  main()
