import torch
import torch.nn.functional as F
from tdnn import TDNN

class Xvector_TDNN(torch.nn.Module):
    def __init__(self, num_features=13):
        super(Xvector_TDNN, self).__init__()
        out_vector_dim = 1500
        self.frame1 = TDNN(input_dim=num_features, output_dim=512, context_size=5, dilation=1)
        self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.frame5 = TDNN(input_dim=512, output_dim=out_vector_dim, context_size=1, dilation=1)

        self.fc1 = torch.nn.Linear(2*out_vector_dim, 500)
        self.fc2 = torch.nn.Linear(500, 1)

    def stats_pooling(self, X, M):
        '''
        Calculate mean and variances across frames
        '''
        M_resized = M[:,:X.size(1),0].unsqueeze(dim=2).repeat(1,1,X.size(2))
        masked = X * M_resized
        frame_length = torch.sum(M[:,:,0].squeeze(), dim=1).unsqueeze(dim=1).repeat(1,masked.size(2))
        xvector_mean = (torch.sum(masked, dim=1)/frame_length).squeeze()
        xvector_std = (torch.sum((masked**2), dim=1)/frame_length) - (xvector_mean**2)
        xvector = torch.cat((xvector_mean, xvector_std), dim=1)
        return x_vector

    def forward(self, X, M):
        '''
        X = [N x F x n]
        M = [N x F x n]

        N = batch size
        F = number of frames
        n = number of features e.g. 13 for mfcc

        M is a mask with 1 in positions of value and 0 otherwise
        '''
        # Pass through TDNN layers
        l1 = self.frame1(X)
        l2 = self.frame2(l1)
        l3 = self.frame3(l2)
        l4 = self.frame4(l3)
        l5 = self.frame5(l4)

        # Stats pooling to get x-vector
        x_vect = self.stats_pooling(l5, M)

        # Predict speaker grade
        h1 = F.relu(self.fc1(x_vect))
        y = self.fc2(h1)
        return y.squeeze()




class Xvector_naive(torch.nn.Module):
    def __init__(self, num_features=13):
        super(Xvector_naive, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(num_features=num_features)
        self.fc1 = torch.nn.Linear(num_features, 500)
        self.fc2 = torch.nn.Linear(500, 500)
        self.fc3 = torch.nn.Linear(500, 500)
        self.fc4 = torch.nn.Linear(500, 500)
        self.fc5 = torch.nn.Linear(500, 512)
        self.fc6 = torch.nn.Linear(512, 600)
        self.fc7 = torch.nn.Linear(600, 600)
        self.fc8 = torch.nn.Linear(600, 600)
        self.fc9 = torch.nn.Linear(600, 1)

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
        M_resized = M[:,:,0].unsqueeze(dim=2).repeat(1,1,h5_reshaped.size(2))
        h5_masked = h5_reshaped * M_resized
        frame_length = torch.sum(M[:,:,0].squeeze(), dim=1).unsqueeze(dim=1).repeat(1,h5_masked.size(2))
        xvector = (torch.sum(h5_masked, dim=1)/frame_length).squeeze()

        # Pass through rest of DNN for score prediction
        h6 = F.relu(self.fc6(xvector))
        h7 = F.relu(self.fc7(h6))
        h8 = F.relu(self.fc8(h7))
        y = self.fc9(h8).squeeze()
        return y
