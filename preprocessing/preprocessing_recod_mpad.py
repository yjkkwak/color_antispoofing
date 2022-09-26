import glob
import os
import shutil


#
def getbasenamewoext(srcfile):
  pathname, extension = os.path.splitext(srcfile)
  return pathname


def splittraintest(jpgpath):
  print(jpgpath)
  fjpglist = glob.glob("{}/**/*.jpg".format(jpgpath), recursive=True)
  # test item user_4X
  with open("./allimgs.txt", "w") as the_file:
    for fjpgpath in fjpglist:
      the_file.write("{}\n".format(fjpgpath))
    the_file.close()




def main():
  splittraintest("/home/user/data1/DBs/antispoofing/RECOD-MPAD")

if __name__ == '__main__':
    main()
