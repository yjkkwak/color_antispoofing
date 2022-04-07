import os

def sendjobs():
  strcmd = "screen -d -m python -u /home/user/work_2022/AntiSpoofing/train.py --lmdbpath {} --GPU 0".format("/home/user/work_db/v220401_01/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260.db")
  os.system (strcmd)
  strcmd = "screen -d -m python -u /home/user/work_2022/AntiSpoofing/train.py --lmdbpath {} --GPU 1".format("/home/user/work_db/v220401_01/Train_v220401_01_CelebA_SiW_LD3007_1by1_260x260.db")
  os.system(strcmd)
  strcmd = "screen -d -m python -u /home/user/work_2022/AntiSpoofing/train.py --lmdbpath {} --GPU 2".format("/home/user/work_db/v220401_01/Train_v220401_01_CelebA_SiW_LDRGB_1by1_260x260.db")
  os.system(strcmd)
  strcmd = "screen -d -m python -u /home/user/work_2022/AntiSpoofing/train.py --lmdbpath {} --GPU 3".format("/home/user/work_db/v220401_01/Train_v220401_01_SiW_LDRGB_LD3007_1by1_260x260.db")
  os.system (strcmd)
if __name__ == '__main__':
  sendjobs()
