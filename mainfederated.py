#ALL RIGHTS RESERVED BY YYR SAMA
from config import celebA_args,GENKI_args,UTKface_args,DSprites_args,testBench,normal,Train,weights_init,fineTuneFedModel
from Federation import federation
from makeDataset import fedDataset,init
import matplotlib.pyplot as plt
from vae import adversary,VAE,VAE_FC
from vision import dynamicpainting
import os,scipy,argparse
from scipy.stats import norm
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required = True, help = 'celebA | GENKI | UTKface | DSprites')
parser.add_argument('--mode',required = True, help = 'Random | Up | Down')
parser.add_argument('--samrounds',required = True, help = 'Up to 500')
opt = parser.parse_args()
dataset = opt.dataset
mode = opt.mode
rounds = int(opt.samrounds)

print("Start:")
args = celebA_args()
if dataset == "GENKI":
    args = GENKI_args()
if dataset == "UTKface":
    args = UTKface_args()
if dataset == "DSprites":
    args = DSprites_args()
testLoader, contriData, fedModel, ftfModel,facVAEModel, DFVAEModel = init(args)

#init experiment results
deltaDpResultDataTot = [[],[],[],[]]
accuracyResultDataTot = [[],[],[],[]]
deltaDpResultDataPart = [[],[],[],[]]
accuracyResultDataPart = [[],[],[],[]]
baslines_label = ['Raw Federated','Fine-tuned Federated','Factor-VAE Adversarial','DFVAE Adversarial']

# begin for simulating

for round in range(rounds):
    print('Sampling Round:',round)
    args.ScaleChange(mode)
    trainLoader = fedDataset(args)

    weights = [len(x.dataset) for x in trainLoader]
    weights = normal(weights)
    print(weights)
    # predictor
    probs = []
    for index in range(args.groups - 1):
        probs.append(0.5)
    if mode == 'Random':
        probs.append(0.5)
    elif mode == 'Up':
        probs.append(norm.cdf(0.0, 51.0, 20.82))
    elif mode == 'Down':
        probs.append(1.0 - norm.cdf(0.0, 51.0, 20.82))

    totLoss,totCorrect,totDeltaDp,lossCollection,accuracyCollection,deltaDpCollection = testBench(fedModel, args,
                                                                                                  testLoader, weights)
    deltaDpResultDataTot[0].append(totDeltaDp)
    accuracyResultDataTot[0].append(totCorrect)
    deltaDpResultDataPart[0].append(deltaDpCollection[args.groups - 1])
    accuracyResultDataPart[0].append(accuracyCollection[args.groups - 1])

    ftfResult = testBench(ftfModel, args, testLoader, weights)
    deltaDpResultDataTot[1].append(ftfResult[2])
    accuracyResultDataTot[1].append(ftfResult[1])
    deltaDpResultDataPart[1].append(ftfResult[5][args.groups - 1])
    accuracyResultDataPart[1].append(ftfResult[4][args.groups - 1])

    facVAEResult = testBench(facVAEModel, args, testLoader, weights)
    deltaDpResultDataTot[2].append(facVAEResult[2])
    accuracyResultDataTot[2].append(facVAEResult[1])
    deltaDpResultDataPart[2].append(facVAEResult[5][args.groups - 1])
    accuracyResultDataPart[2].append(facVAEResult[4][args.groups - 1])

    DFVAEResult = testBench(DFVAEModel, args, testLoader, weights)
    deltaDpResultDataTot[3].append(DFVAEResult[2])
    accuracyResultDataTot[3].append(DFVAEResult[1])
    deltaDpResultDataPart[3].append(DFVAEResult[5][args.groups - 1])
    accuracyResultDataPart[3].append(DFVAEResult[4][args.groups - 1])

    fedModel, models = federation(fedModel, args, trainLoader, weights)
    ftfModel.load_state_dict(fedModel.state_dict())
    fineTuneFedModel(ftfModel, contriData, args)
    DFVAEModel, DFModels = federation(DFVAEModel, args, trainLoader, weights)
    facVAEModel, facModels = federation(facVAEModel, args, trainLoader, weights)

    # next is adversarial models

    if round % max(rounds // 10, int(1)) == 0:
        facVAEModel = adversary(args, contriData, facVAEModel, rSet = [1.0 for i in range(args.groups)])
        DFVAEModel = adversary(args, contriData, DFVAEModel, rSet = probs)

    #testBench(models[args.groups - 1], args, testLoader, weights=[1.0 / args.groups for i in range(args.groups)]) test for every local model,do not delete


print(accuracyResultDataTot)
print(deltaDpResultDataTot)
print(accuracyResultDataPart)
print(deltaDpResultDataPart)
accuracyWholesetCurves = dynamicpainting('accuracyWholeset',accuracyResultDataTot,baslines_label,rounds)
deltaDpWholesetCurves = dynamicpainting('deltadpWholeset',deltaDpResultDataTot,baslines_label,rounds)
accuracyPartsetCurves = dynamicpainting('accuracyPartset',accuracyResultDataPart,baslines_label,rounds)
deltaPartsetCurves = dynamicpainting('deltadpPartset',deltaDpResultDataPart,baslines_label,rounds)