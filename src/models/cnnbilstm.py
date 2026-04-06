import torch
import torch.nn as nn
class CNNBiLSTM(nn.Module):

    def __init__(self, padding, drop_p, pool_size, lstm_hidden, lstm_layers):
        super().__init__()

        self.conv1 = self._convBlock(12, 32, 15, padding,drop_p,pool_size) #64 filters- should have input [500,12]

        self.conv2 = self._convBlock(32, 48, 9, padding,drop_p,pool_size) # 64 filters

        self.conv3 = self._convBlock(48, 64, 5, padding,drop_p,pool_size) # 64 filters

        self.conv4 = self._convBlock(64, 96, 5, padding,drop_p,pool_size) # 64 filters

        self.conv5 = self._convBlock(96, 128, 3, padding,drop_p,pool_size)

        self.conv6 = self._convBlock(128, 256, 3, padding,drop_p,pool_size)


        self.max_pool = nn.MaxPool1d(kernel_size = pool_size)

        self.bilstm = nn.LSTM(
            input_size = 256,
            hidden_size = lstm_hidden,
            num_layers = lstm_layers,
            batch_first = True,
            bidirectional = True
        )

        self.fc1 = nn.Linear(2*lstm_hidden, 128)
        self.fc2 = nn.Linear(128, 5)

        self.dropout = nn.Dropout(p=drop_p)

        self.relu = nn.ReLU(inplace=True) # inplace relu prevents in from allocating memory to a new tensor with the result of this operation. Saving memory.

        #self.bn = nn.BatchNorm1d(256)

        self.transform = nn.Conv1d(12, 48 , kernel_size=1, stride=1)

        self.transform2 = nn.Conv1d(48, 96 , kernel_size=1, stride=2)

        self.transform3 = nn.Conv1d(96, 256 , kernel_size=1, stride=1)



    def _convBlock(self, in_size, out_size, kernel, padding, drop_p, pool_size):
        return nn.Sequential(
            #nn.Conv1d(in_size, in_size, kernel_size = kernel, padding=padding),
            #nn.BatchNorm1d(in_size),
            #nn.ReLU(inplace=True),
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

        x = skip + self.conv6(x)

        x, _ = self.bilstm(x.transpose(1,2))

        x = x[:,-1,:]

        x = self.relu(self.fc1(x))

        x = self.dropout(x)

        return self.fc2(x)