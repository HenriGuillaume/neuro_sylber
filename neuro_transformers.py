import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import HubertConfig, HubertModel
from utils import SplitDataset, split_data, open_pickle
import os


class ECoGFrontend(nn.Module):
    def __init__(self, n_electrodes, hidden_size, n_branches=3, kernel_size=32, stride=16, dropout_p=0.5):
        super().__init__()
        branch_hidden = hidden_size // n_branches

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=n_electrodes,
                out_channels=branch_hidden,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2
            ) for _ in range(n_branches)
        ])

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):  # x: [batch, frames, electrodes]
        x = x.transpose(1, 2)  # [batch, electrodes, frames]
        feats = [conv(x).transpose(1, 2) for conv in self.convs]  # each: [batch, new_frames, branch_hidden]
        x = torch.cat(feats, dim=-1)  #  [batch, new_frames, hidden_size]
        x = self.layer_norm(x)
        return self.dropout(x)


class ECoGHuBERT(nn.Module):
    def __init__(
        self,
        n_electrodes,
        hidden_size=768,
        output_size=768,
        max_frames=16000,
        kernel_size=32,
        stride=16,
        n_branches=32
    ):
        super().__init__()
        self.stride = stride

        self.config = HubertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=12,
            num_attention_heads=12,  # 12 to match hidden size (768) with 64-dim heads
            intermediate_size=2 * hidden_size,
            hidden_dropout=0.5,
            attention_probs_dropout_prob=0.5,
            max_position_embeddings=max_frames,
            vocab_size=output_size
        )

        self.frontend = ECoGFrontend(
            n_electrodes=n_electrodes,
            hidden_size=hidden_size,
            n_branches=n_branches,
            kernel_size=kernel_size,
            stride=stride,
            dropout_p=0.5
        )

        self.encoder = HubertModel(self.config).encoder
        self.pos_emb = nn.Parameter(torch.zeros(1, max_frames, hidden_size))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.projector = nn.Linear(hidden_size, output_size)

    def forward(self, x, attention_mask=None):
        x = self.frontend(x)                         # → [batch, reduced_frames, hidden]
        x = x + self.pos_emb[:, :x.size(1), :]       # add learned positional embedding
        x = self.encoder(x, attention_mask=attention_mask).last_hidden_state
        x = self.layer_norm(x)
        y = self.projector(x)                        # → [batch, reduced_frames, output_size]
        return y



def train_ecoghubert(sub_num,
                    hidden_states,
                    mode='hg',
                    train_ratio=0.05,
                    n_epochs=10,
                    batch_size=16,
                    lr=1e-4,
                    save_folder="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_folder, exist_ok=True)

    dataset = split_data(sub_num=sub_num,
                         hidden_states=hidden_states,
                         train_ratio=train_ratio,
                         mode=mode,
                         target_freq=None) # no resampling of sylber features, so sr=50Hz
    # add batch dimension
    train_X = torch.tensor(dataset.train_X).float().unsqueeze(0).to(device)  # [1, frames, electrodes]
    train_y = torch.tensor(dataset.train_y).float().unsqueeze(0).to(device)
    test_X = torch.tensor(dataset.test_X).float().unsqueeze(0).to(device)
    test_y = torch.tensor(dataset.test_y).float().unsqueeze(0).to(device)   # [1, frames, sylber_dim]

    # our input has a sample rate of 256Hz, our output 50Hz, by taking strides of 5 we can match them
    model = ECoGHuBERT(n_electrodes=train_X.shape[2], output_size=train_y.shape[2], stride=5)
    model = model.to(device)
    

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)

    model.train()
    losses = []
    best_loss = float('inf')
    plateau_counter = 0
    max_plateau = 6
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            plateau_counter = 0
        else:
            plateau_counter += 1

        if plateau_counter >= max_plateau:
            print("Early stopping due to plateau.")
            break

    # Save loss plot
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"loss_curve_subject_{sub_num}.png"))
    plt.close()

    # Save model
    model_path = os.path.join(save_folder, f"ecoghubert_subject_{sub_num}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model, (test_X, test_y)


if __name__ == "__main__":
    import numpy as np
    SYLBER_FEAT_DIR = './pickled_podcast/'
    full_sylber_features = open_pickle(os.path.join(SYLBER_FEAT_DIR,
    'outputs.pkl'))

    HIDDEN_STATES = full_sylber_features['hidden_states']
    del full_sylber_features

    subject_number = 1
    output_folder = "checkpoints"

    model, (test_X, test_y) = train_ecoghubert(sub_num=subject_number,
                                                save_folder=output_folder,
                                                hidden_states=HIDDEN_STATES)

    model.eval()
    with torch.no_grad():
        predictions = model(test_X)

    predictions_np = predictions.cpu().numpy()
    test_y_np = test_y.cpu().numpy()

    np.save(os.path.join(output_folder, f"model_predictions_subject_{subject_number}.npy"),
            {'y_pred': predictions_np, 'y_true': test_y_np})
    print(f"Predictions and ground truth saved to model_predictions_subject_{subject_number}.npy")


