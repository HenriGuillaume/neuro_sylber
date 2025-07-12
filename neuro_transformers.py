import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import HubertConfig, HubertModel
from utils import SplitDataset, split_data, open_pickle
import os
from matplotlib import pyplot as plt
from tqdm import tqdm


class ECoGFrontend(nn.Module):
    def __init__(
        self,
        n_electrodes,
        hidden_size,
        n_branches=3,
        kernel_size=32,
        stride=1,
        dropout_p=0.5
    ):
        super().__init__()
        branch_hidden = hidden_size // n_branches

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=n_electrodes,
                out_channels=branch_hidden,
                kernel_size=kernel_size,
                stride=stride,
                padding='same'
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
        kernel_size=10, # one 5th of a second at 50Tok/s
        stride=1,
        n_branches=16
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

        # Feature projection layer (like BERT)
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )

        self.encoder = HubertModel(self.config).encoder
        self.pos_emb = nn.Parameter(torch.zeros(1, max_frames, hidden_size))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.projector = nn.Linear(hidden_size, output_size)

    def forward(self, x, attention_mask=None):
        x = self.frontend(x)  # [batch, frames', hidden]
        x = x + self.pos_emb[:, :x.size(1), :]  # positional embedding

        x = self.feature_projection(x)  # project

        x = self.encoder(x, attention_mask=attention_mask).last_hidden_state
        x = self.layer_norm(x)
        y = self.projector(x)  # [batch, frames', output_size]

        return y


def chunk_data(X, y, chunk_len, stride=None):
    '''
    Splits X and y into chunks along time dimension for batching
    '''
    if stride is None:
        stride = chunk_len

    chunks_X = []
    chunks_y = []
    for start in range(0, X.shape[0] - chunk_len + 1, stride):
        chunks_X.append(X[start:start+chunk_len])
        chunks_y.append(y[start:start+chunk_len])

    return torch.stack(chunks_X), torch.stack(chunks_y)


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
                         y=hidden_states,
                         train_ratio=train_ratio,
                         mode=mode) # we match ECoG data to sylber token frequency
    # add batch dimension
    chunk_len = 100 # 2second chunks at 50Hz
    stride = chunk_len // 2
    
    train_X_chunks, train_y_chunks = chunk_data(
        torch.tensor(dataset.train_X).float(),
        torch.tensor(dataset.train_y).float(),
        chunk_len=chunk_len,
        stride=stride
    )
    train_X = train_X_chunks  # [B, frames, electrodes]
    train_y = train_y_chunks # [B, frames, sylber_dim]
    print(f'train_X: {train_X.shape}\n train_y: {train_y.shape}\n')
    # define model
    model = ECoGHuBERT(n_electrodes=train_X.shape[2], output_size=train_y.shape[2], stride=1)
    model = model.to(device)
    

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(train_X_chunks, train_y_chunks),
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)

    model.train()
    losses = []
    best_loss = float('inf')
    plateau_counter = 0
    max_plateau = 6
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        total_loss = 0
        for batch_X, batch_y in tqdm(train_loader):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
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
        torch.cuda.empty_cache()

    # Save loss plot
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
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
    
    test_X = torch.tensor(dataset.test_X).float()
    test_y = torch.tensor(dataset.test_y).float()

    return model, (test_X, test_y)


def ecoghubert_predict(model: ECoGHuBERT, test_X, test_y, chunk_len=100, batch_size=8):
    '''
    Performs predictions on chunked data to save memory during inference
    '''
    device = next(model.parameters()).device
    # MAKE SURE THERE IS NO OVERLAP IN CHUNKING
    test_X_chunks, test_y_chunks = chunk_data(test_X, test_y, chunk_len, stride=chunk_len)

    preds = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, test_X_chunks.size(0), batch_size), desc="Predicting"):
            batch_chunks = test_X_chunks[i:i+batch_size].to(device)  # [batch_size, chunk_len, channels]
            batch_preds = model(batch_chunks)  # [batch_size, chunk_len, output_dim]
            preds.append(batch_preds.cpu())

            del batch_chunks, batch_preds
            torch.cuda.empty_cache()

    preds = torch.cat(preds, dim=0)  # [num_chunks, chunk_len, output_dim]
    return preds


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
                                                hidden_states=HIDDEN_STATES,
                                                n_epochs=2)

    

    predictions_np = ecoghubert_predict(model, test_X, test_y).numpy()
    test_y_np = test_y.cpu().numpy()

    np.save(os.path.join(output_folder, f"model_predictions_subject_{subject_number}.npy"),
            {'y_pred': predictions_np, 'y_true': test_y_np})
    print(f"Predictions and ground truth saved to model_predictions_subject_{subject_number}.npy")


