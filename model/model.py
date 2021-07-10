import torch
import torch.nn as nn
import torch.nn.functional as F

class BN_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
 
    def forward(self, x):
        return F.relu(self.seq(x))

class spot_Net(nn.Module):
    def __init__(self, init_weights=False):
        super(spot_Net, self).__init__()
        self.branch1 = nn.Sequential(
                    nn.AvgPool2d(3, 3, 1),
                    BN_Conv2d(1, 3, 1, 1, 0, bias=False))

        self.branch2 = nn.Sequential(
                    BN_Conv2d(1, 5, 1, 1, 0, bias=False),
                    BN_Conv2d(5, 5, 3, 3, 1, bias=False))
        
        self.branch3 = nn.Sequential(
                   BN_Conv2d(1, 8, 1, 1, 0, bias=False),
                   BN_Conv2d(8, 8, 3, 1, 1, bias=False),
                   BN_Conv2d(8, 8, 3, 3, 1, bias=False))

        self.branch4 = nn.Sequential(
                    nn.AvgPool2d(3, 3, 1),
                    BN_Conv2d(16, 16, 1, 1, 0, bias=False))

        self.branch5 = nn.Sequential(
                    BN_Conv2d(16, 16, 1, 1, 0, bias=False),
                    BN_Conv2d(16, 16, 3, 3, 1, bias=False))
        
        self.branch6 = nn.Sequential(
                   BN_Conv2d(16, 16, 1, 1, 0, bias=False),
                   BN_Conv2d(16, 16, 3, 1, 1, bias=False),
                   BN_Conv2d(16, 16, 3, 3, 1, bias=False))

        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.fc1 = nn.Linear(48*2*2, 256)
        self.fc2 = nn.Linear(256, 1)
        
        self.se_fc1 = nn.Conv2d(48, 3, kernel_size=1)
        self.se_fc2 = nn.Conv2d(3, 48, kernel_size=1)

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
    
    def forward(self, x1, x2, x3):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out3 = self.branch3(x3)

        merge1 = torch.cat((out1,out2,out3), 1)

        out4 = self.branch4(merge1)
        out5 = self.branch5(merge1)
        out6 = self.branch6(merge1)

        merge2 = torch.cat((out4,out5,out6), 1)

        #print(merge2.size(2))
        w = F.avg_pool2d(merge2, merge2.size(2))
        w = F.relu(self.se_fc1(w))
        w = F.sigmoid(self.se_fc2(w))
        merge2 = merge2 * w
        
        out = self.maxpool(merge2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out
if __name__=="__main__":
    from torchsummary import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = spot_Net(init_weights=True)
    net.to(device)
    #net.load_state_dict(torch.load("../net_min_loss.pt"))
    summary(net, [(1,42,42), (1,42,42), (1,42,42)])