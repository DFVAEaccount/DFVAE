import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy
import torch.optim as optim
from torch.utils.data import DataLoader
import collections
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
# name, test split and train split, scale up step length, target_type, batch_size
# structure for every record is pic, group, target and other sensitive attributes
class args():
    def __init__(self):
        self.groups = 3
        self.Groupscale = [0.9, 0.9, 0.5]
        self.stepLength = 0.001
        self.maxRounds = 500
        self.ordinaryEpoch = int(1)
        self.contriSplit = 0.10
        self.targetLoc = 1
        self.groupLoc = 0
        self.in_chan = 3
        self.lr = 0.001
        self.batch_size = 50
        self.imshape_1 = int(128)
        self.imshape_2 = int(128)
        self.trainEpochs = int(3)
        self.encodingEpochs = int(10)
        self.reLossFactor = 0.01
    def ScaleChange(self, mode):
        if mode == 'Random':
            if numpy.random.rand() > 0.5:
                self.Groupscale[self.groups - 1] -= self.stepLength
            else:
                self.Groupscale[self.groups - 1] += self.stepLength
        if mode == 'Up':
            self.Groupscale[self.groups - 1] += self.stepLength
        if mode == 'Down':
            self.Groupscale[self.groups - 1] -= self.stepLength

# target is attractive, sensitive attr is black hair and  Blurry
class celebA_args(args):
    def __init__(self):
        super(celebA_args,self).__init__()
        self.name = "celebA"
        self.groups = 10
        self.groupSplits = [[(0, 30000)],
                            [(30000,50000)],
                            [(50000,70000)],
                            [(70000,90000)],
                            [(90000,110000)],
                            [(110000,130000)],
                            [(130000,150000)],
                            [(150000,170000)],
                            [(170000,190000)],
                            [(190000,202599)]]
        self.Groupscale = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.5]
        self.testSplit = 0.1
        self.contriSplit = 0.05
        self.stepLength = 0.004
        self.maxRounds = 5000
        self.out_dim = 2

#target is smile or not, sentive attr is pose X,Y,Z
#total records < 4000
class GENKI_args(args):
    def __init__(self):
        super(GENKI_args,self).__init__()
        self.name = "GENKI"
        self.groups = 3
        self.groupSplits = [[(0,1000),(2000,2500)],
                            [(1000,1800),(2500,3200)],
                            [(1800,2000),(3200,3999)]]
        self.Groupscale = [0.9, 0.9, 0.5]
        self.testSplit = 0.2
        self.stepLength = 0.004
        self.out_dim = 2
#target is gender, sentive attr is age and race
class UTKface_args(args):
    def __init__(self):
        super(UTKface_args, self).__init__()
        self.name = "UTKface"
        self.groups = 4
        self.groupSplits = [[(0,500),(1000,1500),(2000,2500),(4000,4500),(6000,6500),(9000,9500)],
                            [(500,1000),(2500,3000),(4500,5000),(7000,7500),(8000,9000)],
                            [(3000,3500),(5000,6000),(6500,7000),(7500,8000)],
                            [(1500,2000),(3500,4000),(9500,10137)]]
        self.Groupscale = [0.6, 0.6, 0.6, 0.3]
        self.testSplit = 0.2
        self.stepLength = 0.004
        self.out_dim = 3
        self.pretrainEpochs = int(1)
#target is shape, sentive attr is Xpos and Ypos
class DSprites_args(args):
    def __init__(self):
        super(DSprites_args,self).__init__()
        self.name = "DSprites"
        self.groups = 10
        self.groupSplits = [[(0, 30000),(700000,737280)],
                            [(30000, 50000),(650000,700000)],
                            [(50000, 70000),(600000,650000)],
                            [(70000, 90000),(550000,600000)],
                            [(90000, 110000),(500000,550000)],
                            [(110000, 130000),(450000,500000)],
                            [(130000, 150000),(400000,450000)],
                            [(150000, 170000),(350000,400000)],
                            [(170000, 190000),(250000,350000)],
                            [(190000, 250000)]]
        self.Groupscale = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.5]
        self.contriSplit = 0.05
        self.testSplit = 0.1
        self.stepLength = 0.004
        self.maxRounds = 5000
        self.in_chan = 1
        self.out_dim = 4
        self.imshape_1 = int(64)
        self.imshape_2 = int(64)

class Conv(nn.Module):
    def __init__(self, out_dim ,imshape_1, imshape_2,in_chan = 3):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True))
        self.mlp = nn.Sequential(
            nn.Linear(64 * imshape_1 * imshape_2 // 256, 128),
            nn.ReLU(True),
            nn.Linear(128, out_dim))
        self.imshape_1 = imshape_1
        self.imshape_2 = imshape_2
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,64 * self.imshape_1 * self.imshape_2 // 256)
        x = self.mlp(x)
        out = F.log_softmax(x,dim = 1)
        return out

def normal(weights):
    sum = 0.0
    for i in range(len(weights)):
        sum += weights[i]
    for i in range(len(weights)):
        weights[i] = weights[i] / sum
    return weights



def attachlabel(args,targets,attrnum = 1):
    if attrnum > 2:
        raise Exception('too much attributes selected!',attrnum)

    sensitivecol = targets[:,2:2 + attrnum]
    length = len(targets)
    labelarr = numpy.zeros([length,attrnum])
    if args.name == 'GENKI':
        for index in range(length):
            for col in range(attrnum):
                labelarr[index][col] = -1.0 if sensitivecol[index][col] < 0 else 1.0

    if args.name == 'UTKface':
        for index in range(length):
            for col in range(attrnum):
                if col == 0:
                    labelarr[index][col] = -1.0 if sensitivecol[index][col] < 50 else 1.0
                if col == 1:
                    labelarr[index][col] = -1.0 if sensitivecol[index][col] < 3 else 1.0

    if args.name == 'DSprites':
        for index in range(length):
            for col in range(attrnum):
                labelarr[index][col] = -1.0 if sensitivecol[index][col] < 0.5 else 1.0

    if args.name == 'celebA':
        for index in range(length):
            for col in range(attrnum):
                labelarr[index][col] = -1.0 if sensitivecol[index][col] < 0 else 1.0


    return torch.from_numpy(labelarr).float()

def Train(Model,trainloader,args,specialTrain = False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = optim.Adam(Model.parameters(), lr=args.lr)
    maxEpoch = args.trainEpochs if specialTrain == True else args.ordinaryEpoch
    for epoch in range(maxEpoch):
        #print('Epoch:',epoch)
        for data, anno in trainloader:
            optimizer.zero_grad()
            data, anno = data.to(device), anno.to(device)
            out = Model(data)
            target = anno[:, 1]
            loss = F.nll_loss(out, target.long(),reduction='sum')
            loss.backward()
            optimizer.step()
    print('a training process ends')
        #print(loss.item())

def testBench(model,args,testLoader,weights):
    def testDeltaDP(pred,anno,deltaDp):
        binaryLabel = attachlabel(args,anno)
        for i in range(len(pred)):
            if binaryLabel[i] == -1.0:
                deltaDp[0][1] += 1
                if pred[i] == anno[i][1]:
                    deltaDp[0][0] += 1
            else:
                deltaDp[1][1] += 1
                if pred[i] == anno[i][1]:
                    deltaDp[1][0] += 1

    accuracyCollection = []
    lossCollection = []
    dpDisCollection = []

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    with torch.no_grad():
        totLoss = 0
        totCorrect = 0
        totDeltaDp = 0
        for i in range(args.groups):
            deltaDp = [[0,0],[0,0]] #the first line is class_1,the second is class_2
            testLoss = 0
            correct = 0

            for data, anno in testLoader[i]:
                data, anno = data.to(device), anno.to(device)
                target = anno[:, 1]

                output = model(data)
                testLoss += F.nll_loss(output, target.long(), reduction='sum').item()  # sum up batch loss
                pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                testDeltaDP(pred,anno,deltaDp)

            testLoss /= len(testLoader[i].dataset)
            accuracy = 100. * correct / len(testLoader[i].dataset)
            delta = abs(deltaDp[0][0] / deltaDp[0][1] - deltaDp[1][0] / deltaDp[1][1])

            accuracyCollection.append(accuracy)
            lossCollection.append(testLoss)
            dpDisCollection.append(delta)

            print('Testset_{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%),DeltaDp: class1_{}/{} class2_{}/{}'.format(
                i,testLoss, correct, len(testLoader[i].dataset),accuracy,deltaDp[0][0],deltaDp[0][1],deltaDp[1][0],deltaDp[1][1]))

            totLoss += weights[i] * testLoss
            totCorrect += weights[i] * accuracy
            totDeltaDp += weights[i] * delta

        print('Wholeset: Average loss: {:.4f}, Accuracy: {:.0f}%, DeltaDp: {:.4f}'.format(
            totLoss, totCorrect,totDeltaDp))

    return totLoss,totCorrect,totDeltaDp,lossCollection,accuracyCollection,dpDisCollection

def fineTuneFedModel(ftfModel, contriSet, args):
    publicDataset = contriSet[0]
    for i in range(1, args.groups):
        publicDataset += contriSet[i]
    publicTrainLoader = DataLoader(publicDataset, batch_size = 128, shuffle = True)
    Train(ftfModel, publicTrainLoader, args, specialTrain = True)

def parametersAvg(workerStateDictCurr, tarModel, prop):
    workerStateDictTar  = tarModel.state_dict()
    weight_keys = list(workerStateDictCurr.keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        keyVal = prop * workerStateDictCurr[key] + (1.0 - prop) * workerStateDictTar[key]
        fed_state_dict[key] = keyVal
    #### update fed weights to fl model
    tarModel.load_state_dict(fed_state_dict)
