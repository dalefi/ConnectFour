import numpy as np
import torch

from src.ConnectFour import ConnectFour


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
    def __init__(self, in_units, out_units, dropout_rate = .0):
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
    def __init__(self, H=[200,100], num_channels = 32, dropout_rate = .0):
        """
        dropout_rate ist standardmaessig 0: AlphaZero regularisiert ueber den
        staendig erneuerten Replay-Buffer, nicht ueber Dropout. Zwei gestapelte
        Schichten mit p=0.5 haben vor allem den Value-Head gedaempft.
        """

        # input shape: batch_size x 2 x 6 x 7

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
        
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(H[1], H[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(H[1], 7),
            # torch.nn.Softmax(dim =-1)
        )

    """
    Defines the loss
    """
    def alphaloss(self, nn_value, nn_policy, result, mcts_policy):
        """
        Gibt (Gesamtloss, Value-Loss, Policy-Loss) zurueck.

        Die Aufteilung ist wichtig fuer die Diagnose: der Cross-Entropy-Term
        enthaelt die Entropie des MCTS-Ziels als nicht reduzierbaren Sockel
        (bei Gleichverteilung ~1.95), der Gesamtwert sieht dadurch aus, als
        wuerde das Training stagnieren. Fuer den echten Fortschritt siehe policy_kl.
        """
        # Value loss (MSE)
        value_loss = torch.nn.MSELoss()(
            nn_value.reshape(-1),
            result.reshape(-1)
        )

        # Policy loss (Cross-Entropy)
        policy_loss = torch.nn.CrossEntropyLoss()(
            nn_policy.reshape(-1, 7),
            mcts_policy.reshape(-1, 7)
        )

        return value_loss + policy_loss, value_loss, policy_loss

    @staticmethod
    def policy_kl(nn_policy, mcts_policy):
        """
        KL(MCTS || Netz): der Teil des Policy-Losses, der tatsaechlich auf 0
        gedrueckt werden kann.
        """
        log_predicted = torch.nn.functional.log_softmax(nn_policy.reshape(-1, 7), dim=-1)
        target = mcts_policy.reshape(-1, 7)
        target_entropy = -(target * torch.log(target.clamp_min(1e-9))).sum(dim=-1)
        cross_entropy = -(target * log_predicted).sum(dim=-1)
        return (cross_entropy - target_entropy).mean()

    def forward(self, X):
        """
        Falls ich dem Modell eine ConnectFour-Instanz übergebe, wird sie hier in
        die kanonische Zwei-Kanal-Darstellung uebersetzt (siehe encode_board).
        """

        if isinstance(X, ConnectFour):
            X = state_to_tensor(X)

        if X.dim() == 3:
            X = X.unsqueeze(0)
        Y = self.model(X)
        v = self.value_head(Y)
        p = self.policy_head(Y).reshape(-1, 7)
        return {'value': v, 'policy': p}


def encode_board(board, current_player) -> torch.Tensor:
    """
    Kanonische Eingabe fuer das Netz: zwei binaere Ebenen aus Sicht des Spielers
    am Zug - eigene Steine, gegnerische Steine.

    Dadurch ist eine Stellung und ihr farbvertauschtes Gegenstueck fuer das Netz
    dieselbe Eingabe. Es muss die Vorzeichensymmetrie nicht mehr lernen, und
    Value/Policy beziehen sich immer auf den Spieler am Zug - genau wie die
    Trainingsziele.

    Das ist die *einzige* Stelle, an der die Eingabe definiert wird. Suche und
    Training muessen zwingend dieselbe Kodierung benutzen.
    """
    canonical = np.asarray(board, dtype=np.int8) * current_player
    own_stones = torch.from_numpy((canonical == 1).astype(np.float32))
    opponent_stones = torch.from_numpy((canonical == -1).astype(np.float32))
    return torch.stack((own_stones, opponent_stones))


def encode_boards(boards, current_players) -> torch.Tensor:
    """
    Vektorisierte Fassung von encode_board fuer einen ganzen Replay-Buffer.
    Muss exakt dasselbe liefern wie encode_board - dafuer gibt es einen Test.
    """
    boards = np.asarray(boards, dtype=np.int8).reshape(-1, 6, 7)
    players = np.asarray(current_players, dtype=np.int8).reshape(-1, 1, 1)

    canonical = boards * players
    own_stones = (canonical == 1).astype(np.float32)
    opponent_stones = (canonical == -1).astype(np.float32)

    return torch.from_numpy(np.stack((own_stones, opponent_stones), axis=1))


def state_to_tensor(state=None) -> torch.Tensor:
    # Sicherstellen, dass wir ein Objekt mit einem .board Attribut haben
    if state is None or not hasattr(state, 'board'):
        raise TypeError(f"Expected ConnectFour-like object, got {type(state)}")

    return encode_board(state.board, state.get_current_player())


def mirror_board(board):
    """
    Connect Four ist links-rechts-symmetrisch. Das Spiegeln liefert eine gueltige,
    aber andere Trainingsstellung - kostenlos doppelt so viele Daten.
    """
    return np.ascontiguousarray(np.asarray(board)[:, ::-1])


def mirror_policy(policy):
    return np.ascontiguousarray(np.asarray(policy)[::-1])


def load_model(model_path=None, model_tag=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CFNet()
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    model.tag = model_tag

    return model
