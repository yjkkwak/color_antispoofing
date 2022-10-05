import os
import time

def getbasenamewoext(srcfile):
  pathname, extension = os.path.splitext(srcfile)
  return pathname

def main():
  basemeta = "/home/user/work_2022/AntiSpoofing/meta/v220922/"
  baselmdb = "/home/user/work_db/v220922/"
  dblist = ["Test_4C1_CW.list",
            # "Test_4C1_CASIA.list",
            # "Test_4C1_FASD.list",
            # "Test_4C1_MSU.list",
            # "Test_4C1_OULU.list",
            # "Test_4C1_RECOD.list",
            # "Test_4C1_REPLAY.list",
            # "Train_4C3_SiW_RECOD_AIHUBx2_CASIA_MSU_OULU.list",
            # "Train_4C3_SiW_RECOD_AIHUBx2_CASIA_MSU_REPLAY.list",
            # "Train_4C3_SiW_RECOD_AIHUBx2_CASIA_OULU_REPLAY.list",
            # "Train_4C3_SiW_RECOD_AIHUBx2_MSU_OULU_REPLAY.list"
            ]

  patchtypelist = ["1by1_260x260"]#, "4by3_244x324"]# 1by1_260x260 / 4by3_244x324

  for dbitem in dblist:
    for patchitem in patchtypelist:
      strpythoncmd = "python -u lmdbwriter.py "
      stroptions = " -listpath {} -dbpath {} -patchtype {}".format(
        os.path.join(basemeta, dbitem),
        baselmdb,
        patchitem)
      strcmd = "{} {}".format(strpythoncmd, stroptions)
      usedpathname = "{}_{}.db.sort.path".format(getbasenamewoext(dbitem), patchitem)
      usedimagepath = os.path.join(baselmdb, "{}".format(usedpathname))
      print (usedimagepath)
  #    time.sleep(10)
      os.system(strcmd)
      while True:
        if os.path.exists(usedimagepath):
          break
        time.sleep(100)
        print ("Wait!, generating lmdb", usedimagepath)

  print ("Done!!!")



if __name__ == '__main__':
  main()
