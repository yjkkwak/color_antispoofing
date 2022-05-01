from PIL import Image, ImageDraw, ImageFile, ImageOps
from multiprocessing import Pool

import os
import numpy as np
import argparse
import random
import mydatum_pb2
import lmdb

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument("-listpath", type=str, default="/home/user/work_2022/AntiSpoofing/meta/v220401_01/Test_v220401_01_SiW.list", help='the list file recording all images')
parser.add_argument("-dbpath", type=str, default="/home/user/work_db/v220401_01", help='version of db')
parser.add_argument("-patchtype", type=str, default="1by1_260x260", help='patch type')
# parser.add_argument("-patchtype", type=str, default="4by3_244x324", help='patch type')
args = parser.parse_args()
print (args)


tmpfacesize = {}
tmpfacesize["100"] = 1
tmpfacesize["200to300"] = 1
tmpfacesize["300to400"] = 1
tmpfacesize["400to500"] = 1
tmpfacesize["500to600"] = 1
tmpfacesize["600to"] = 1

def getbasenamewoext(srcfile):
  pathname, extension = os.path.splitext(srcfile)
  return pathname

def setfacesize(neww, imgpath):
  if neww < 100:
    tmpfacesize["100"] = tmpfacesize["100"] + 1
  if neww > 200 and neww < 300:
    tmpfacesize["200to300"] = tmpfacesize["200to300"] + 1
  if neww > 300 and neww < 400:
    tmpfacesize["300to400"] = tmpfacesize["300to400"] + 1
  if neww > 400 and neww < 500:
    # print("400to500", imgpath)
    tmpfacesize["400to500"] = tmpfacesize["400to500"] + 1
  if neww > 500 and neww < 600:
    if "GOPRO" not in imgpath:
      print ("500to600", imgpath)
      tmpfacesize["500to600"] = tmpfacesize["500to600"] + 1
  if neww > 600:
    if "GOPRO" not in imgpath:
      print("600to", imgpath)
      tmpfacesize["600to"] = tmpfacesize["600to"] + 1

def genXbyYcorrdinate(x, y, w, h, imgw, imgh, XbyY, imgpath):
  x2, y2 = (x + w), (y + h)
  cx = (x + x2) // 2
  cy = (y + y2) // 2

  if XbyY == "1by1":
    halfmaxw = max(w, h) // 2
    halfmaxh = max(w, h) // 2
  elif XbyY == "4by3":
    halfmaxw = max(w, h) // 2
    halfmaxh = max(w, h) // 1.5

  newx = cx - halfmaxw
  newy = cy - halfmaxh
  neww = int(halfmaxw * 2)
  newh = int(halfmaxh * 2)

  # debug
  # print (newx, newy, neww, newh, newx + neww, newy + newh, imgw, imgh)
  # 1. exception : out of bound
  if newx < 0 or newy < 0 or (newx + neww) > imgw or (newy + newh) > imgh:
    # print("{}.fd occur out of bound for image".format(imgpath))
    return -1, 0, 0, 0, 0
  # 2. eception : Small and Big face
  # if neww > 500:
  #   # print("{}.fd occur big face {}".format(imgpath, neww))
  #   return -2, 0, 0, 0, 0
  # if neww < 100:
  #   # print("{}.fd occur small face {}".format(imgpath, neww))
  #   return -3, 0, 0, 0, 0

  newx = cx - halfmaxw
  newy = cy - halfmaxh
  neww = int(halfmaxw * 2)
  newh = int(halfmaxh * 2)
  return 1, newx, newy, neww, newh

def genpatch(imgpath):
  fd_path = "{}.fd".format(imgpath)
  if os.path.exists(fd_path) == False:
    # print ("{}.fd not exists".format(imgpath))
    return None

  pilimg = Image.open(imgpath)
  pilimg = ImageOps.exif_transpose(pilimg)

  with open(fd_path, "r") as the_file:
    strline = the_file.readline()
    the_file.close()

  strtokens = strline.split()
  if len(strtokens) != 4:
    # print ("fd format is not collect, do fd again {}".format(imgpath))
    return None
  x, y, w, h = int(strtokens[0]), int(strtokens[1]), int(strtokens[2]), int(strtokens[3])
  # 256 x 256 from 260 x 260
  ecode1, x_1by1, y_1by1, w_1by1, h_1by1 = genXbyYcorrdinate(x, y, w, h, pilimg.width, pilimg.height, "1by1", imgpath)
  # 240 x 320 from 244 x 324
  ecode2, x_4by3, y_4by3, w_4by3, h_4by3 = genXbyYcorrdinate(x, y, w, h, pilimg.width, pilimg.height, "4by3", imgpath)

  if ecode1 + ecode2 < 2:
    # print ("gen patch error code 1by1:{} 4by3:{}".format(ecode1, ecode2))
    return None

  if False:
    draw = ImageDraw.Draw(pilimg)
    draw.rectangle([(x,y), (x+w,y+h)], outline="red")
    draw.rectangle([(x_1by1, y_1by1), (x_1by1 + w_1by1, y_1by1 + h_1by1)], outline="blue")
    draw.rectangle([(x_4by3, y_4by3), (x_4by3 + w_4by3, y_4by3 + h_4by3)], outline="green")
    pilimg.show()
    # import cv2
    # img_raw = cv2.imread(imgpath)
    # cv2.rectangle(img_raw, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # name = "./fdbox_.jpg"
    # cv2.imwrite(name, img_raw)

  if "1by1" in args.patchtype:
    pilimg_1by1 = pilimg.crop([x_1by1, y_1by1, x_1by1 + w_1by1, y_1by1 + h_1by1])
    pilimg_cropresize = pilimg_1by1.resize((260, 260))
    # pilimg_1by1.show()
  elif "4by3" in args.patchtype:
    pilimg_4by3 = pilimg.crop([x_4by3, y_4by3, x_4by3 + w_4by3, y_4by3 + h_4by3])
    pilimg_cropresize = pilimg_4by3.resize((244, 324))
    # pilimg_4by3.show()
  # pilimg_cropresize.show()
  npimg = np.array(pilimg_cropresize)
  return npimg

def showpatch(imgpath):
  fd_path = "{}.fd".format(imgpath)
  if os.path.exists(fd_path) == False:
    print ("{}.fd not exists".format(imgpath))
    return None

  pilimg = Image.open(imgpath)
  pilimg = ImageOps.exif_transpose(pilimg)

  with open(fd_path, "r") as the_file:
    strline = the_file.readline()
    the_file.close()

  strtokens = strline.split()
  if len(strtokens) != 4:
    print ("fd format is not collect, do fd again {}".format(imgpath))
    return None
  x, y, w, h = int(strtokens[0]), int(strtokens[1]), int(strtokens[2]), int(strtokens[3])
  # 256 x 256 from 260 x 260
  ecode1, x_1by1, y_1by1, w_1by1, h_1by1 = genXbyYcorrdinate(x, y, w, h, pilimg.width, pilimg.height, "1by1", imgpath)
  # 240 x 320 from 244 x 324
  ecode2, x_4by3, y_4by3, w_4by3, h_4by3 = genXbyYcorrdinate(x, y, w, h, pilimg.width, pilimg.height, "4by3", imgpath)

  if ecode1 + ecode2 < 2:
    # print ("gen patch error code 1by1:{} 4by3:{}".format(ecode1, ecode2))
    return None


  draw = ImageDraw.Draw(pilimg)
  draw.rectangle([(x,y), (x+w,y+h)], outline="red")
  draw.rectangle([(x_1by1, y_1by1), (x_1by1 + w_1by1, y_1by1 + h_1by1)], outline="blue")
  draw.rectangle([(x_4by3, y_4by3), (x_4by3 + w_4by3, y_4by3 + h_4by3)], outline="green")
  pilimg.show()

def genmydatum(imgpath):
  npimg = genpatch(imgpath)
  if npimg is None:
    return None
  # Siw : live or /spoof/
  # Celeba : live or /spoof/
  # LDRGB / 3007 : real or attack
  mydatum = mydatum_pb2.myDatum()
  mydatum.width = npimg.shape[1]  # im.width
  mydatum.height = npimg.shape[0]  # im.height
  mydatum.channels = npimg.shape[2]
  mydatum.label = 0
  if "live" in imgpath or "real" in imgpath or "emotion" in imgpath:
    mydatum.label = 1
  mydatum.data = npimg.tobytes()
  mydatum.path = imgpath

  return mydatum

def genmydatum_multi(samples):
  pool = Pool()
  datums = pool.map(genmydatum, samples)
  pool.close()
  pool.join()
  datumswNotNone = [sample for sample in datums if sample is not None]
  print ("{}/{}".format(len(datumswNotNone), len(samples)))
  return datumswNotNone

def writedatumtolmdb():
  with open(args.listpath, "r") as the_file:
    strtmplines = the_file.readlines()
    the_file.close()

  strlines = []
  for strline in strtmplines:
    strline = strline.strip()
    strlines.append(strline)

  random.shuffle(strlines)
  batch_size = 3000
  batches = int(len(strlines) / batch_size)
  print (batches)

  listname = getbasenamewoext(os.path.basename(args.listpath))
  dbname = "{}_{}.db".format(listname, args.patchtype)
  usedpathname = "{}_{}.db.path".format(listname, args.patchtype)

  lmdbpath = os.path.join(args.dbpath, "{}".format(dbname))
  usedimagepath = os.path.join(args.dbpath, "{}".format(usedpathname))
  print (lmdbpath)
  print(usedimagepath)

  if os.path.exists(lmdbpath):
    print ("exists lmdb", lmdbpath)
    return

  the_file = open(usedimagepath, "w")
  env = lmdb.open(lmdbpath, map_size=int(1 << 40), lock=False)
  counter = 0
  txn = env.begin(write=True)
  for index in range(batches):
    print ("{}/{}".format(index, batches))
    sub_lines = strlines[index * batch_size:(index + 1) * batch_size]

    datums = genmydatum_multi(sub_lines)

    for datum_id in range(len(datums)):
      if datums[datum_id] is None:
        print ("what???")
        continue
      strid = "{:08}".format(counter)
      the_file.write("{}\n".format(datums[datum_id].path))
      txn.put(strid.encode("ascii"), datums[datum_id].SerializeToString())
      counter +=1

  print ("----------- the rest -----------------")
  sub_lines = strlines[batches*batch_size:]
  print(len(sub_lines))
  datums = genmydatum_multi(sub_lines)
  for datum_id in range(len(datums)):
    if datums[datum_id] is None:
      print("what???")
      continue
    strid = "{:08}".format(counter)
    the_file.write("{}\n".format(datums[datum_id].path))
    txn.put(strid.encode("ascii"), datums[datum_id].SerializeToString())
    counter += 1
  txn.commit()
  print ("total datum size: "+str(counter))
  env.close()
  the_file.close()

def main():
  writedatumtolmdb()
  #genmydatum("/home/user/data1/DBs/antispoofing/CelebA_spoofing/CelebA_Spoof/Data/Test/9727/live/529180.png")
  # showpatch("/home/user/data1/DBs/antispoofing/CelebA_spoofing/CelebA_Spoof/Data/Test/5412/spoof/553892.png")
  # showpatch("/home/user/data1/DBs/antispoofing/CelebA_spoofing/CelebA_Spoof/Data/Test/9727/live/529180.png")

"""
comments
/home/user/work_2022/AntiSpoofing/meta/v220401_01/Test_v220401_01_CelebA.list
'100': 2661, 
'200to300': 23419, 
'300to400': 10824, 
'400to500': 2790
'500to600': 1284, 
'600to': 280,

/home/user/work_2022/AntiSpoofing/meta/v220401_01/Test_v220401_01_LD3007.list
'200to300': 17481,
'300to400': 10935, 
'400to500': 5525, 
'500to600': 694, 
'600to': 8


/home/user/work_2022/AntiSpoofing/meta/v220401_01/Test_v220401_01_SiW.list
'100': 125
'200to300': 3562,
'300to400': 12382,
'400to500': 18618,
'500to600': 15847, 
'600to': 2611

/home/user/work_2022/AntiSpoofing/meta/v220401_01/Test_v220401_01_LDRGB.list
'100': 3
'200to300': 1260, 
'300to400': 5596, 
'400to500': 10095, 
'500to600': 26236,
'600to': 10649, 
"""

if __name__ == '__main__':
  main()
