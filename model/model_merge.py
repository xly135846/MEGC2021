import torch
import torch.nn as nn
import torch.nn.functional as F

class BN_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)

class merge_Net(nn.Module):
    def __init__(self, init_weights=False, batch_size=32, dataset="SAMM", mode="Macro"):
        super(merge_Net, self).__init__()
        self.batch_size =batch_size
        self.branch1 = nn.Sequential(
                    BN_Conv2d(3, 16, 3, 1, 0, bias=False),
                    BN_Conv2d(16, 16, 3, 2, 0, bias=False))

        self.branch4 = nn.Sequential(
                    BN_Conv2d(16, 16, 3, 1, 0, bias=False),
                    BN_Conv2d(16, 16, 3, 2, 0, bias=False))

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        if dataset=="SAMM" and mode=="Macro":
            self.conv1d_1 = nn.Conv1d(256, 128, 3, stride=2, padding=0)
            self.conv1d_2 = nn.Conv1d(128, 64, 3, stride=2, padding=0)
            self.conv1d_3 = nn.Conv1d(64, 32, 3, stride=2, padding=0)
            self.fc1 = nn.Linear(416, 256)
            self.fc2 = nn.Linear(256, 2)
        elif dataset=="SAMM" and mode=="Micro" or (dataset=="CAS" and mode=="Macro"):
            self.conv1d_1 = nn.Conv1d(256, 128, 3, stride=2, padding=0)
            self.conv1d_2 = nn.Conv1d(128, 64, 3, stride=2, padding=0)
            self.conv1d_3 = nn.Conv1d(64, 32, 3, stride=2, padding=0)
            self.fc1 = nn.Linear(32, 32)
            self.fc2 = nn.Linear(32, 2)
        elif dataset=="CAS" and mode=="Micro":
            self.conv1d_1 = nn.Conv1d(256, 128, 3, stride=1, padding=1)
            self.conv1d_2 = nn.Conv1d(128, 64, 3, stride=1, padding=1)
            self.conv1d_3 = nn.Conv1d(64, 32, 3, stride=1, padding=0)
            self.fc1 = nn.Linear(128, 128)
            self.fc2 = nn.Linear(128, 2)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x1):
        x1 = x1.view(-1, 3, 42, 42)
        out1 = self.branch1(x1)
        out2 = self.branch4(out1)
        w = F.avg_pool2d(out2, 2)

        out = w.view(w.size(0), 1,  w.size(1)*w.size(2)*w.size(3) , -1)
        out = out.view(self.batch_size, -1, out.size(2))
        out = out.permute(0,2,1)
        
        out = self.conv1d_1(out)
        out = self.relu(out)
        out = self.conv1d_2(out)
        out = self.relu(out)
        out = self.conv1d_3(out)
        out = self.relu(out)
        conv_out = out.view(out.size(0), -1)
        
        out = self.fc1(conv_out)
        out = self.relu(out)
        out = self.fc2(out)
        return out, conv_out
