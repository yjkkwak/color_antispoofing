import os
import time

def getbasenamewoext(srcfile):
  pathname, extension = os.path.splitext(srcfile)
  return pathname

def main():
  basemeta = "/home/user/work_2022/AntiSpoofing/meta/v220401_01/"
  baselmdb = "/home/user/work_db/v220401_01/"
  dblist = [
            "Test_v220401_01_SiW.list",
            "Test_v220401_01_CelebA.list",
            "Test_v220401_01_LD3007.list",
            "Test_v220401_01_LDRGB.list",
            "Train_v220401_01_CelebA_LDRGB_LD3007.list",
            "Train_v220401_01_CelebA_SiW_LD3007.list",
            "Train_v220401_01_CelebA_SiW_LDRGB.list",
            "Train_v220401_01_SiW_LDRGB_LD3007.list"
            ]
  patchtypelist = ["1by1_260x260"]# 1by1_260x260 / 4by3_244x324

  for dbitem in dblist:
    for patchitem in patchtypelist:
      strpythoncmd = "python -u lmdbwriter.py "
      stroptions = " -listpath {} -dbpath {} -patchtype {}".format(
        os.path.join(basemeta, dbitem),
        baselmdb,
        patchitem)
      strcmd = "{} {}".format(strpythoncmd, stroptions)
      usedpathname = "{}_{}.db.path".format(getbasenamewoext(dbitem), patchitem)
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
