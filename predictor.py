import torch
import torch.nn as nn
import numpy
import torch.optim as optim
import os
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class mlPridictor(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(mlPridictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, out_dim))
        self.out_dim = out_dim
    def forward(self, x):
        x = self.mlp(x)
        return x

LinearPretrainedData = numpy.zeros([500,200,51])
LinearPretrainedTarget = numpy.zeros([500,200,2])
# only for up and down
for d in range(500):
    for st in range(100):
        for j in range(51):
            LinearPretrainedData[d][2 * st][j] = st * d + j * d
            LinearPretrainedData[d][2 * st + 1][50 - j] = st * d + j * d
        LinearPretrainedTarget[d][2 * st][0] = 51.0 * d
        LinearPretrainedTarget[d][2 * st][1] = 20.82 * d
        LinearPretrainedTarget[d][2 * st + 1][1] = -51.0 * d
        LinearPretrainedTarget[d][2 * st + 1][1] = 20.82 * d
Datatensor = torch.from_numpy(LinearPretrainedData).float()
Targetensor = torch.from_numpy(LinearPretrainedTarget).float()
predictor = mlPridictor(51,2).to(device)

for epoch in range(50):
    Loss = nn.MSELoss()
    optimizer = optim.Adam(predictor.parameters(),lr = 0.005)
    for i in range(500):
        data = Datatensor[i][:][:].to(device)
        target = Targetensor[i][:][:].to(device)
        out = predictor(data)
        loss = Loss(out,target)
        loss.backward()
        optimizer.step()
    print(epoch)
torch.save(predictor.state_dict(),"./predictor.pt")

def predict():
    if os.path.exists("./predictor.pt"):
        p = mlPridictor(51,2)
        p.load_state_dict(torch.load("./predictor.pt"))
        p.to(device)
        return p
    else:
        print('please pretrain predictor')