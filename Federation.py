import collections
from config import Conv,testBench
import threading
from config import Train, parametersAvg
import torch
from copy import deepcopy
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
def federation(fedModel,args,trainLoader,weights):
    models = []
    thread = []
    for trainloader in trainLoader:
        Model = Conv(out_dim=args.out_dim, imshape_1=args.imshape_1, imshape_2=args.imshape_2,
                            in_chan=args.in_chan)
        Model.load_state_dict(fedModel.state_dict())
        Model.to(device)
        models.append(Model)
        worker = threading.Thread(target = Train,args=(Model,trainloader,args))
        worker.start()
        thread.append(worker)

    for t in thread:
        t.join()
    print('Federation Complete!')

    worker_state_dict = [x.state_dict() for x in models]
    fedCurrDict = deepcopy(fedModel.state_dict())
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0.0
        for i in range(len(models)):
            key_sum += weights[i] * worker_state_dict[i][key]
        fed_state_dict[key] =  key_sum
    #### update fed weights to fl model
    fedModel.load_state_dict(fed_state_dict)
    parametersAvg(fedCurrDict, fedModel, 0.9)
    return fedModel,models