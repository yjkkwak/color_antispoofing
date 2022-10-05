import glob
import tarfile
import zipfile
import os
from tqdm import tqdm
import multiprocessing as mp
import cv2

def mydelcmd(strcmd):
  print (strcmd)
  os.system(strcmd)

def getbasenamewoext(srcfile):
  pathname, extension = os.path.splitext(srcfile)
  return pathname

def extractpngfrommp4(fullpath):
  """worker unzips one file"""
  print ("extracting... {}".format(fullpath))
  fdirname = os.path.dirname(fullpath)
  dstpath = os.path.join(fdirname, "frames")
  # print (dstpath)

  if os.path.exists(dstpath) == False:
    os.makedirs(dstpath, exist_ok=True)

  vidcap = cv2.VideoCapture(fullpath)
  success, image = vidcap.read()
  count = 0
  while success:
    pngpath = os.path.join(dstpath, "frame_{:05d}.png".format(count))
    cv2.imwrite(pngpath, image)  # save frame as JPEG file
    # print (pngpath)
    success, image = vidcap.read()
    # print('Read a new frame: ', success, pngpath)
    # pass 0.7 sec
    # vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 700))  # added this line
    count += 1


def convertmp42png(movpath):
  print(movpath)
  fmovlist = glob.glob("{}/**/*.mp4".format(movpath), recursive=True)
  # for fmovpath in fmovlist:
  #   extractpngfrommp4(fmovpath)
  #   break

  pool = mp.Pool(min(mp.cpu_count(), len(fmovlist)))  # number of workejpgrs
  pool.map(extractpngfrommp4, fmovlist, chunksize=10)
  pool.close()

def main():
  convertmp42png("/home/user/data2/s3/CW")
  
if __name__ == '__main__':
    main()
