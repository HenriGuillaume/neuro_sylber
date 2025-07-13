import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import HubertConfig, HubertModel
from utils import CONFIG, SplitDataset, split_data, open_pickle
import os
from matplotlib import pyplot as plt
from tqdm import tqdm



class ECoGFrontend(nn.Module):
    def __init__(
        self,
        n_electrodes: int,
        hidden_size: int,
        kernel_sizes=(1, 5, 10),
        n_filters_per_branch=32,
        dropout_p=0.5
    ):
        super().__init__()

        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=n_electrodes,
                        out_channels=n_filters_per_branch,
                        kernel_size=k,
                        stride=1,
                        padding=k // 2
                    ),
                    nn.GELU(),
                    nn.BatchNorm1d(n_filters_per_branch),
                    nn.Dropout(dropout_p)
                )
            )

        total_filters = n_filters_per_branch * len(kernel_sizes)
        self.projection = nn.Conv1d(total_filters, hidden_size, kernel_size=1)

    def forward(self, x):  # x: [B, T, E]
        x = x.transpose(1, 2)  # [B, E, T]
        x_cat = torch.cat([branch(x) for branch in self.branches], dim=1)  # [B, F, T]
        x_proj = self.projection(x_cat)  # [B, H, T]
        return x_proj.transpose(1, 2)  # [B, T, H]


class ShallowHubert(nn.Module):
    def __init__(self, hidden_size=768, num_layers=2, n_heads=12, dropout=0.5):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask=None):  # [B, T, D]
        if attention_mask is not None:
            # Convert to bool mask for TransformerEncoder
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.layer_norm(x)


class ECoGHuBERT(nn.Module):
    def __init__(
        self,
        n_electrodes,
        speech_upstream=CONFIG['model']['base_model'],
        sylber_ckpt=CONFIG['model']['sylber_checkpoint'],
        hidden_size=768,
        output_size=768,
        max_frames=16000,
        kernel_size=10,
        n_conv_layers=3,
        num_last_layers=2, # number of deep layers we keep
        freeze=[-1], # freeze pretrained model
        device='cuda'
    ):
        super().__init__()

        self.frontend = ECoGFrontend(
            n_electrodes=n_electrodes,
            hidden_size=hidden_size,
            dropout_p=0.5
        )

        self.config = HubertConfig.from_pretrained(speech_upstream)
        self.encoder = HubertModel(self.config).encoder
        try:
            state_dict = torch.load(sylber_ckpt, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=False)
            print("Loaded Sylber HuBERT weights")
        except Exception as e:
            print(f"Failed to load Sylber weights: {e}")
            exit()
        

        self.truncated_encoder = nn.ModuleList(list(self.encoder.layers)[-4:])

        if freeze:
            for i in freeze:
                for param in self.truncated_encoder[i].parameters():
                    param.requires_grad = False

        self.pos_emb = nn.Parameter(torch.zeros(1, max_frames, hidden_size))

        # Add shallow transformer to adapt ECoG to HuBERT space
        self.shallow = ShallowHubert(
            hidden_size=hidden_size,
            num_layers=2,
            n_heads=12,
            dropout=0.5
        )

    def forward(self, x, attention_mask=None):  # x: [B, T, electrodes]
        x = self.frontend(x)  # [B, T, hidden_size]
        x = x + self.pos_emb[:, :x.size(1), :]

        x = self.shallow(x, attention_mask=attention_mask)

        for layer in self.truncated_encoder:
            x = layer(x, attention_mask=attention_mask)[0]

        return x


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
                     model=None,
                     mode='hg',
                     train_ratio=0.5,
                     val_ratio=0.25,
                     n_epochs=10,
                     batch_size=16,
                     lr=1e-4,
                     chunk_len=100,
                     max_plateau=15,
                     save_folder="checkpoints",
                     loss_type="cosine",            # 'mse', 'cosine', or 'mse->cosine'
                     loss_switch_epoch=None         # int: epoch to switch from MSE to Cosine
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_folder, exist_ok=True)

    dataset = split_data(sub_num=sub_num,
                         y=hidden_states,
                         train_ratio=train_ratio,
                         val_ratio=val_ratio,
                         mode=mode)

    stride = chunk_len // 2

    def to_chunks(X, y):
        return chunk_data(torch.tensor(X).float(), torch.tensor(y).float(), chunk_len=chunk_len, stride=stride)

    train_X_chunks, train_y_chunks = to_chunks(dataset.train_X, dataset.train_y)
    val_X_chunks, val_y_chunks = to_chunks(dataset.val_X, dataset.val_y)

    if model is None:
        model = ECoGHuBERT(n_electrodes=train_X_chunks.shape[2],
                           output_size=train_y_chunks.shape[2])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    mse_loss_fn = nn.MSELoss()
    cosine_loss_fn = nn.CosineSimilarity(dim=2, eps=1e-6)

    train_loader = DataLoader(TensorDataset(train_X_chunks, train_y_chunks),
                              batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_X_chunks, val_y_chunks),
                            batch_size=batch_size, shuffle=False, pin_memory=True)

    losses, val_losses = [], []
    best_val_loss = float('inf')
    plateau_counter = 0

    def compute_loss(output, target, epoch):
        if loss_type == 'mse':
            return mse_loss_fn(output, target)
        elif loss_type == 'cosine':
            return -cosine_loss_fn(output, target).mean()
        elif loss_type == 'mse->cosine':
            if loss_switch_epoch is None:
                raise ValueError("`loss_switch_epoch` must be set when using 'mse->cosine'")
            if epoch < loss_switch_epoch:
                return mse_loss_fn(output, target)
            else:
                return -cosine_loss_fn(output, target).mean()
        else:
            raise ValueError(f"Unknown loss_type '{loss_type}'")

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        model.train()
        total_loss = 0

        for batch_X, batch_y in tqdm(train_loader,
                                     desc=f"Training Epoch {epoch+1}",
                                     leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = compute_loss(output, batch_y, epoch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc="Validation", leave=False):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = compute_loss(output, batch_y, epoch)
                val_total_loss += loss.item()
        avg_val_loss = val_total_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            plateau_counter = 0
            model_path = os.path.join(save_folder, f"ecoghubert_subject_{sub_num}_best.pt")
            torch.save(model.state_dict(), model_path)
        else:
            plateau_counter += 1

        if plateau_counter >= max_plateau and max_plateau:
            print("Early stopping due to plateau on validation.")
            break

        torch.cuda.empty_cache()

    # Save final loss curve
    plt.figure()
    plt.plot(losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve (subject {sub_num})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"loss_curve_subject_{sub_num}.png"))
    plt.close()

    return model, (torch.tensor(dataset.test_X).float(), torch.tensor(dataset.test_y).float())


    # Plot loss
    plt.figure()
    plt.plot(losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Cosine Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"loss_curve_subject_{sub_num}.png"))
    plt.close()

    print(f"Best model saved to {model_path}")
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
    preds = preds.reshape(-1, preds.shape[-1])
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
                                               train_ratio=0.8,
                                               save_folder=output_folder,
                                               hidden_states=HIDDEN_STATES,
                                               n_epochs=100)

    

    predictions_np = ecoghubert_predict(model, test_X, test_y).numpy()
    test_y_np = test_y.cpu().numpy()

    np.save(os.path.join(output_folder, f"model_predictions_subject_{subject_number}.npy"),
            predictions_np)
    print(f"Predictions and ground truth saved to model_predictions_subject_{subject_number}.npy")


