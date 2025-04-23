import torch
from ConnectFour import ConnectFour

class SmallBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, X):
        return self.model(X)

class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.model = torch.nn.Sequential(
            SmallBlock(in_channels, mid_channels),
            torch.nn.ReLU(),
            SmallBlock(mid_channels, in_channels)
        )

    def forward(self, X):
        Y = self.model(X)
        Y = Y+X
        Y = torch.nn.ReLU()(Y)
        return Y


class DropoutBlock(torch.nn.Module):
    def __init__(self, in_units, out_units, dropout_rate = .5):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_units, out_units),
            torch.nn.BatchNorm1d(out_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_rate)
        )
        
    def forward(self, X):
        return self.model(X)

class CFNet(torch.nn.Module):
    def __init__(self, H=[200,100], num_channels = 32, dropout_rate = .5):

        # input shape: batch_size x 7 x args.M x args.N

        super().__init__()
        self.epoch = None
        self.initial_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_channels),
            torch.nn.ReLU()
        )
        
        self.middle_blocks = torch.nn.Sequential(
            *[ResnetBlock(num_channels, num_channels) for _ in range(5)]
        )

        """
        Das muss ich noch ausarbeiten!
        """
        
        self.dropout_blocks = torch.nn.Sequential(
            DropoutBlock(num_channels * 6 * 7, H[0]),
            DropoutBlock(H[0], H[1])
        )
        
        self.model = torch.nn.Sequential(
            self.initial_block,
            self.middle_blocks,
            torch.nn.Flatten(start_dim=1),  #flatten only along channel dimension
            self.dropout_blocks
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(H[1], H[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(H[1], 1),
            torch.nn.Tanh()
        )

        """
        Hier auch jeweils args.N und .M austauschen ... muss mir Gedanken machen.
        """
        
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(H[1], H[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(H[1], 7), # 7 mögliche Züge
            torch.nn.Softmax(dim =-1)
        )

    """
    Defines the loss
    """
    
    def alphaloss(self, y, mcts_val, mcts_policy):
        loss1 = torch.nn.MSELoss()(
            y['value'].reshape(-1),
            mcts_val
            )
        loss2 = torch.nn.CrossEntropyLoss()(
            y['policy'].reshape(-1,7),
            mcts_policy.reshape(-1,7)
            )
        loss = loss1 + loss2
        
        return loss

    
    def forward(self, X):
        """
        Falls ich dem Modell eine ConnectFour-Instanz übergebe, soll es das hier so
        anpassen, dass es nutzbar ist. D.h. das board wird als erster Kanal verwendet,
        und als zweiter Kanal wird ein 6x7 Tensor mit 1en bei Spieler 1's Zug uns
        -1en bei Spieler -1's Zug benutzt.
        """
        
        if isinstance(X, ConnectFour):
            X = state_to_tensor(X)
            
        if X.dim() == 3:
            X = X.unsqueeze(0)
        Y = self.model(X)
        v = self.value_head(Y)
        p = self.policy_head(Y).reshape(-1, 7)
        return {'value': v, 'policy': p}


def state_to_tensor(state = None) -> torch.Tensor:
    """
    Transforms boardstate into a tensor that can be feed into the NN.
    """

    assert isinstance(state, ConnectFour)
    player_tensor = torch.ones(6, 7, dtype=int)
    input_tensor = torch.stack((torch.Tensor(state.board), state.get_current_player() * player_tensor))

    return input_tensor

def tensor_to_state(tensor = None):
    """
    Transforms tensor to be fed into NN into a boardstate.
    """
    
    board = tensor[0].numpy()
    player = 1 if tensor[1].numpy()[0][0] == 1 else -1

    CF = ConnectFour(board = board, currentPlayer=player)

    return CF






















