import torch
import torch.nn as nn

class ConvNet(nn.Module):

    def __init__(self, padding, drop_p, pool_size):


        super().__init__()

        self.conv1 = self._convBlock(12, 32, 9, padding, drop_p) #16 filters- should have input [1000,1]
        self.conv2 = self._convBlock(32, 48, 7, padding, drop_p) # 16 filters
        self.conv3 = self._convBlock(48, 64, 5, padding, drop_p) # 16 filters prev had only 2 layers, added 2 more with improved results
        self.conv4 = self._convBlock(64, 96, 5, padding, drop_p) # 16 filters
        self.conv5 = self._convBlock(96, 128, 3, padding, drop_p) # 64 filters
        self.conv6 = self._convBlock(128, 192, 3, padding, drop_p) # 64 filters

        self.max_pool = nn.MaxPool1d(kernel_size = pool_size)
        self.fc1 = nn.Linear(24000, 4000)
        self.fc2 = nn.Linear(4000, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500,250)
        self.fc5 = nn.Linear(250,5)
        self.dropout = nn.Dropout(p=drop_p)
        self.relu = nn.ReLU(inplace =True)
        self.flatten = nn.Flatten()

        self.transform =  nn.Conv1d(12, 48, 1,  stride=1)
        self.transform2 =  nn.Conv1d(48, 96, 1,  stride=2)
        self.transform3 =  nn.Conv1d(96, 192, 1,  stride=1)


    def _convBlock(self, in_size, out_size, kernel, padding, drop_p):
        return nn.Sequential(
            nn.Conv1d(in_size, in_size, kernel_size = kernel, padding=padding),
            nn.BatchNorm1d(in_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_size, out_size, kernel_size = kernel, padding=padding),
            nn.BatchNorm1d(out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),

            )


    def forward(self, x):

        skip = self.transform(x)

        x = self.conv1(x) # [500,192]

        x = skip + self.conv2(x) # [500, 128]

        skip = self.transform2(skip)

        x = self.conv3(x) #500, 96

        x = self.max_pool(x)

        x = skip + self.conv4(x) #500, 64

        skip = self.transform3(skip)

        x = self.conv5(x) # 256 256

        x = skip + self.conv6(x) # 250, 256

        x = self.max_pool(x)

        x = self.flatten(x) # [15*256] = [3840]

        x = self.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.relu(self.fc2(x))

        x = self.dropout(x)

        x = self.relu(self.fc3(x))

        x = self.dropout(x)


        x = self.relu(self.fc4(x))


        return self.fc5(x)