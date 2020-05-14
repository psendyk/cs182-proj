import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model.forward(x)

class EnsembleNet(nn.Module):
    def __init__(self, num_classes, models):
        super(EnsembleNet, self).__init__()
        self.num_classes = num_classes
        self.models = {}
        for model_path in models:
            model = Net(num_classes)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.models[model_path] = model
    
    def forward(self, x):
        outputs = torch.empty((len(self.models), x.size(0), self.num_classes))
        for idx, (model_path, model) in enumerate(self.models.items()):
            outputs[idx,:,:] = model(x)
        # compute probabilities
        probs = F.softmax(outputs, dim=2)
        avg_probs = torch.mean(probs, dim=0)
        return avg_probs
