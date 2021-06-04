import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import  DataLoader
import numpy
from torch.utils.data import Subset
from torchvision import datasets, transforms
from PIL import Image
import os
from random import shuffle
from config import Conv,Train

# basic args for our federated dataset is group number, data scale, data and multi-features(or targets)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
def transform(imshape_1,imshape_2):
    return transforms.Compose([
    transforms.Resize((imshape_1,imshape_2)),
    transforms.ToTensor()
    ])
class makeDataset(Dataset):
    def __init__(self,args):

        imgs = []
        picpath = "./"

        if args.name == 'GENKI':
            picpath = "./GENKI-R2009a/files"
            targetpath = './GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt'
            annoInfo = open(targetpath, 'r')
            picfiles = os.listdir(picpath)

            picN = 3987
            for line in annoInfo:
                line = line.strip('\n')
                words = line.split()

                groupN = 0
                for i in range(0,args.groups):
                    for Range in args.groupSplits[i]:
                        if picN -3987 >= Range[0] and picN - 3987 < Range[1]:
                           groupN = i

                target = numpy.array([float(groupN),float(words[0]),float(words[1]),float(words[2]),float(words[3])])
                picname = picfiles[picN]
                imgs.append((picname, target))
                picN += 1

        #
        if args.name == 'UTKface':
            picpath = "./utkface/part1"
            picfiles = os.listdir(picpath)

            picN = 0
            for pics in picfiles:
                words = pics.split('_')

                if len(words) != 4:
                    continue
                groupN = 0

                for i in range(0,args.groups):
                    for Range in args.groupSplits[i]:
                        if picN  >= Range[0] and picN < Range[1]:
                           groupN = i

                target = numpy.array([float(groupN), float(words[1]), float(words[0]) , float(words[2])])
                picname = pics
                imgs.append((picname,target))
                picN += 1


        if args.name == 'celebA':
            picpath = "./celeba/Img/img_align_celeba"
            targetpath = './celeba/Anno/list_attr_celeba.txt'
            annoInfo = open(targetpath, 'r')
            picfiles = os.listdir(picpath)

            picN = 0
            for line in annoInfo:
                line = line.strip('\n')
                words = line.split()
                if words[0].endswith('.jpg') != 1:
                    continue

                groupN = 0
                for i in range(0,args.groups):
                    for Range in args.groupSplits[i]:
                        if picN  >= Range[0] and picN  < Range[1]:
                           groupN = i
                targetclass = 0 if float(words[2]) == -1. else 1

                target = numpy.array([float(groupN),targetclass,float(words[9]),float(words[11])])

                picname = picfiles[picN]
                imgs.append((picname, target))
                picN += 1

        self.imgs = imgs
        self.picpath = picpath
        self.args = args
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        pic,label = self.imgs[index]
        pic = Image.open(self.picpath+'\\'+pic)
        pic = transform(self.args.imshape_1,self.args.imshape_2)(pic)
        label = torch.from_numpy(label).float()
        return pic,label

class DSpriteSet(Dataset):
    def __init__(self,args):
        imgs = []
        picpath = './dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        pics = numpy.load(picpath, allow_pickle=True, encoding="latin1")
        data = pics['imgs']
        anno = pics['latents_values']
        picN = 0
        for img in data:
            groupN = 0
            for i in range(0,args.groups):
                for Range in args.groupSplits[i]:
                    if picN  >= Range[0] and picN  < Range[1]:
                        groupN = i
            target = numpy.array([float(groupN),float(anno[picN][1]),float(anno[picN][4]),float(anno[picN][5])])

            imgs.append((img, target))
            picN += 1

        self.imgs = imgs
        self.picpath = picpath

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        pic,label = self.imgs[index]
        pic = torch.from_numpy(pic).float().unsqueeze(0)
        label = torch.from_numpy(label).float()
        return pic,label


rawdata = []
groupData = []
testLoader = []
contriData = []
trainData = []
def init(args):
    global rawdata
    if args.name == "GENKI":
        print('selected dataset: Genki')
        rawdata = makeDataset(args)
    if args.name == "UTKface":
        print('selected dataset: UTKface')
        rawdata = makeDataset(args)
    if args.name == "celebA":
        print('selected dataset: celebA')
        rawdata = makeDataset(args)
    if args.name == "DSprites":
        print('selected dataset: DSprites')
        rawdata = DSpriteSet(args)
    print(len(rawdata))

    # fetch valid indice
    indices = []
    for i in range(args.groups):
        newindice = []
        indices.append(newindice)
    if args.name != 'celebA':
        index = -1
        for pic in rawdata:
            index += 1
            if args.name == 'GENKI' or args.name == 'UTKface':
                size = pic[0].size()
                if list(size)[0] != 3:
                    continue
            id = int(pic[1][0])
            indices[id].append(index)


        for i in range(args.groups):
            shuffle(indices[i])
    else:
        for i in range(args.groups):
            l = int(args.groupSplits[i][0][0])
            r = int(args.groupSplits[i][0][1])
            for index in range(l,r):
                indices[i].append(index)
            shuffle(indices[i])

    global groupData
    for i in range(args.groups):
        groupData.append(Subset(rawdata,indices[i]))
        print(len(groupData[i]))


    global testData
    global contriData
    global trainbeginer
    global trainData
    for i in range(args.groups):
        testLength = int(args.testSplit * len(groupData[i]))
        tmpLoader = DataLoader(Subset(groupData[i],range(0,testLength)),batch_size = 3000)
        testLoader.append(tmpLoader)
        trainData.append(Subset(groupData[i],range(testLength,len(groupData[i]))))
        contriLength = int(args.contriSplit * len(trainData[i]))
        contriData.append(Subset(trainData[i],range(0,contriLength)))


    fedModel = Conv(out_dim=args.out_dim, imshape_1 = args.imshape_1,
                                                       imshape_2 = args.imshape_2, in_chan=args.in_chan).to(device)
    ftfModel = Conv(out_dim=args.out_dim, imshape_1=args.imshape_1,
                                                       imshape_2=args.imshape_2, in_chan=args.in_chan).to(device)

    facVAEModel = Conv(out_dim=args.out_dim, imshape_1=args.imshape_1,
                                                       imshape_2=args.imshape_2, in_chan=args.in_chan).to(device)
    DFVAEModel = Conv(out_dim=args.out_dim, imshape_1=args.imshape_1,
                                                       imshape_2=args.imshape_2, in_chan=args.in_chan).to(device)
    '''
    preTrainDataset = Subset(trainData[0],range(0,2000))
    for i in range(1, args.groups):
        preTrainDataset += Subset(trainData[i],range(0,2000))
    print('preTrain SetSize', len(preTrainDataset))
    preTrainLoader = DataLoader(preTrainDataset, batch_size = 64, shuffle=True)
    Train(fedModel, preTrainLoader, args, specialTrain = True)
    '''
    ftfModel.load_state_dict(fedModel.state_dict())
    facVAEModel.load_state_dict(fedModel.state_dict())
    DFVAEModel.load_state_dict(fedModel.state_dict())
    return testLoader,contriData,fedModel,ftfModel.to(device),facVAEModel.to(device),DFVAEModel.to(device)

def fedDataset(args):
    trainLoader = []
    for i in range(args.groups):
        trainLength = int(args.Groupscale[i] * len(trainData[i]))
        traindata = Subset(trainData[i],range(0 ,trainLength))
        print('train scale:', len(traindata))
        tmpLoader = DataLoader(traindata,batch_size = args.batch_size,shuffle = True)
        trainLoader.append(tmpLoader)

    return trainLoader


   
