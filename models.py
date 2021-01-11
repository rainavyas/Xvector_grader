import torch
import torch.nn.functional as F

class Xvector(torch.nn.Module):
    def __init__(self, num_features=13):
        super(Xvector, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_features)
        self.fc1 = torch.nn.Linear(num_features, 200)
        self.fc2 = torch.nn.Linear(200, 200)
        self.fc3 = torch.nn.Linear(200, 200)
        self.fc4 = torch.nn.Linear(200, 200)
        self.fc5 = torch.nn.Linear(200, 512)
        self.fc6 = torch.nn.Linear(512, 200)
        self.fc7 = torch.nn.Linear(200, 1)

    def forward(self, X, M):
        '''
        X = [N x F x n]
        M = [N x F x n]

        N = batch size
        F = number of frames
        n = number of features e.g. 13 for mfcc

        M is a mask with 1 in positions of value and 0 otherwise
        '''

        # Pass all frames through same 5-layer DNN
        X_reshaped = torch.reshape(X, (-1, X.size(2)))
        X_norm = self.bn1(X_reshaped)
        h1 = F.relu(self.fc1(X_norm))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))

        # Calculate X-vectors by mean over frames
        h5_reshaped = torch.reshape(h5, (X.size(0), X.size(1), h5.size(1)))
        print("h5_reshaped ", h5_reshaped.size())
        M_resized = M[:,:,0].repeat(1,1,h5_reshaped.size(2))
        print("M cut ", M[:,:,0].size())
        print("M_resized ", M_resized.size())
        h5_masked = h5_reshaped * M_resized
        frame_length = torch.sum(M[:,:,0].squeeze(), dim=1)
        xvector = torch.sum(h5_masked, dim=1)/frame_length

        # Pass through rest of DNN for score prediction
        h6 = F.relu(self.fc6(xvector))
        y = self.fc7(h6)
        return y
