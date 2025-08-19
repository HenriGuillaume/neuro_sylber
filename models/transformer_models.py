import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import HubertModel, Wav2Vec2Model, HubertConfig, Wav2Vec2Config
import sys
sys.path.append('../utils')
from data_utils import CONFIG, SplitDataset, split_data, open_pickle, save_predictions
import os
import re
from matplotlib import pyplot as plt
from tqdm import tqdm


class BinnedECoGFrontend(nn.Module):
    '''
    This convolutional layer does not change the sampling rate,
    so it requires binning of the input signal.
    '''
    def __init__(
        self,
        n_electrodes: int,
        hidden_dim: int,
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
                        padding='same'
                    ),
                    nn.GELU(),
                    nn.BatchNorm1d(n_filters_per_branch),
                    nn.Dropout(dropout_p)
                )
            )

        total_filters = n_filters_per_branch * len(kernel_sizes)
        self.projection = nn.Conv1d(total_filters, hidden_dim, kernel_size=1)

    def forward(self, x):  # x: [B, T, E]
        x = x.transpose(1, 2)  # [B, E, T]
        x_cat = torch.cat([branch(x) for branch in self.branches], dim=1)  # [B, F, T]
        x_proj = self.projection(x_cat)  # [B, H, T]
        return x_proj.transpose(1, 2)  # [B, T, H]


class DownsampleFrontend(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch_per_branch: int,
                 strides: int,
                 kernel_sizes=(3, 5, 11),
                 dropout_p=0.5):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=in_ch,
                          out_channels=out_ch_per_branch,
                          kernel_size=k,
                          stride=strides,
                          padding=k//2),
                nn.GELU(),
                nn.BatchNorm1d(out_ch_per_branch),
                nn.Dropout(dropout_p)
            )
            for k in kernel_sizes
        ])
    def forward(self, x):
        # x: [B, C_in, T_in]
        out = torch.cat([b(x) for b in self.branches], dim=1)
        # out: [B, C_out = branches*filters, T_in/stride]
        return out


class ECoGFrontend(nn.Module):
    '''
    This version of the convolutional block takes in the full signal (512Hz),
    BE CAREFUL to feed it chunks of 500frames. The number of samples in the
    output is brought down to 50 through strided convolution.
    '''
    def __init__(self,
                 n_electrodes: int,
                 hidden_dim: int=768,
                 filters_per_branch=32,
                 kernel_sizes=(3,5,11),
                 dropout_p=0.5):
        super().__init__()
        # Block 1: downsample ×5
        self.block1 = DownsampleFrontend(
            in_ch=n_electrodes,
            out_ch_per_branch=filters_per_branch,
            strides=5,
            kernel_sizes=kernel_sizes,
            dropout_p=dropout_p
        )
        # Block 2: downsample ×2
        total_ch = filters_per_branch * len(kernel_sizes)
        self.block2 = DownsampleFrontend(
            in_ch=total_ch,
            out_ch_per_branch=filters_per_branch,
            strides=2,
            kernel_sizes=kernel_sizes,
            dropout_p=dropout_p
        )
        # Final projection to M dims
        total_ch2 = filters_per_branch * len(kernel_sizes)
        self.project = nn.Conv1d(total_ch2, hidden_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, T=500, E]
        assert x.shape[1] == 500, "Input must have precisely 500 samples"
        x = x.transpose(1,2)            # [B, E, 500]
        x = self.block1(x)              # [B, C1, 100]
        x = self.block2(x)              # [B, C2, 50]
        x = self.project(x)             # [B, M, 50]
        return x.transpose(1,2)         # [B, 50, M]


class ShallowHubert(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=1, n_heads=12, dropout=0.5):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, attention_mask=None):  # [B, T, D]
        if attention_mask is not None:
            # Convert to bool mask for TransformerEncoder
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.layer_norm(x)


class ECoGHuBERT_classifier(nn.Module):
    def __init__(
        self,
        n_electrodes,
        speech_upstream=CONFIG['model']['hubert_base_model'],
        sylber_ckpt=CONFIG['model']['sylber']['checkpoint'],
        hidden_dim=768,
        max_frames=16000,
        dropout=0.5,
    ):
        super().__init__()

        self.frontend = ECoGFrontend(
            n_electrodes=n_electrodes,
            hidden_dim=hidden_dim,
            dropout_p=dropout
        )
        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))

        # Shallow transformer to map ECoG → speech model hidden space
        self.shallow = ShallowHubert(dropout=dropout)
        
        self.classifier = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x, attention_mask=None):  # x: [B, T, electrodes]
        assert x.shape[1] == 500, "Input must have precisely 500 samples"
        x = self.frontend(x)  # [B, T, hidden_dim]
        x = x + self.pos_emb[:, :x.size(1), :]

        x = self.shallow(x, attention_mask=attention_mask)

        x = self.classifier(x)  # [B, T, 1]
        x = self.activation(x)  # [B, T, 1]
        x = x.squeeze(-1)       # [B, T] 
        return x


class ECoGHuBERT(nn.Module):
    def __init__(
        self,
        n_electrodes,
        speech_upstream=CONFIG['model']['hubert_base_model'],
        sylber_ckpt=CONFIG['model']['sylber']['checkpoint'],
        hidden_dim=768,
        output_size=768,
        max_frames=16000,
        pretrained_layer_indices=[-1, -2],  # which pretrained layers to keep
        freeze=[],                    # which to freeze
    ):
        super().__init__()

        self.frontend = ECoGFrontend(
            n_electrodes=n_electrodes,
            hidden_dim=hidden_dim,
            dropout_p=0.3
        )

        # Detect whether model is HuBERT or wav2vec2
        if 'hubert' in speech_upstream.lower():
            self.config = HubertConfig.from_pretrained(speech_upstream)
            base_model = HubertModel.from_pretrained(speech_upstream)
        elif 'wav2vec' in speech_upstream.lower():
            self.config = Wav2Vec2Config.from_pretrained(speech_upstream)
            base_model = Wav2Vec2Model.from_pretrained(speech_upstream)
        else:
            raise ValueError("Unsupported model type: must be HuBERT or Wav2Vec2")

        self.encoder = base_model.encoder

        # Optionally load checkpoint (Sylber finetuned weights)
        if sylber_ckpt:
            try:
                state_dict = torch.load(sylber_ckpt, map_location='cpu')
                self.encoder.load_state_dict(state_dict, strict=False)
                print("Loaded Sylber HuBERT/Wav2Vec2 weights")
            except Exception as e:
                print(f"Failed to load Sylber weights: {e}")
                exit()

        # Select only specified pretrained layers
        pretrained_layers = list(self.encoder.layers)
        self.truncated_encoder = nn.ModuleList([pretrained_layers[i] for i in pretrained_layer_indices])

        # Optionally freeze layers
        if freeze:
            for i in freeze:
                for param in self.truncated_encoder[i].parameters():
                    param.requires_grad = False

        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))

        # Shallow transformer to map ECoG → speech model hidden space
        self.shallow = None #ShallowHubert()

    def forward(self, x, attention_mask=None):  # x: [B, T, electrodes]
        assert x.shape[1] == 500, "Input must have precisely 500 samples"

        x = self.frontend(x)  # [B, T, hidden_dim]
        x = x + self.pos_emb[:, :x.size(1), :]
        if self.shallow:
            x = self.shallow(x, attention_mask=attention_mask)

        for layer in self.truncated_encoder:
            x = layer(x, attention_mask=attention_mask)[0]

        return x


#-----------------CHUNKING UTILS---------------#

def chunk_data(X, y, chunk_len_X=500, chunk_len_y=50, x_rate=512, y_rate=50, stride_X=None):
    '''
    Splits X and y into chunks along time dimension for batching
    It is advised to provide chunk_len_y to avoid rounding errors
    '''
    if stride_X is None:
        stride_X = chunk_len_X
    
    if chunk_len_y is None:
        # Duration of each chunk in seconds
        chunk_duration = chunk_len_X / x_rate
        stride_duration = stride_X / x_rate

        # Corresponding chunk and stride lengths in y

        chunk_len_y = int(round(chunk_duration * y_rate))
    
    stride_y = int(round(chunk_len_y * stride_X / chunk_len_X))

    chunks_X = []
    chunks_y = []

    max_chunks = min(
        (X.shape[0] - chunk_len_X) // stride_X + 1,
        (y.shape[0] - chunk_len_y) // stride_y + 1
    )

    for i in range(max_chunks):
        start_X = i * stride_X
        start_y = i * stride_y
        chunks_X.append(X[start_X:start_X + chunk_len_X])
        chunks_y.append(y[start_y:start_y + chunk_len_y])

    return torch.stack(chunks_X), torch.stack(chunks_y)


#---------------LOSSES------------------#
def dot_product_loss(y_pred, y_true):
    # BEWARE, THIS IS A TERRIBLE LOSS, YOU
    # CAN CHEAT IT BY OUTPUTING LARGE VECTORS
    # Ensure batch dimension is first: [B, D]
    dot = torch.sum(y_pred * y_true, dim=-1)
    loss = -dot
    return loss.mean()

def cossim_loss(y_pred, y_true):
    return -F.cosine_similarity(y_pred, y_true, dim=-1).mean()


def combined_loss(y_pred, y_true, loss_fn1, loss_fn2, alpha):
    loss1 = loss_fn1(y_pred, y_true)
    loss2 = loss_fn2(y_pred, y_true)
    return alpha * loss1 + (1 - alpha) * loss2


#------------TRAINING LOOPS--------------#


def train_model(model, 
                dataset: SplitDataset,
                loss_fn1, 
                loss_fn2, # set to null function if none
                alpha_schedule,
                chunk_len=500, # leave as is unless you know what youre doing
                n_epochs=128,
                batch_size=32,
                lr=1e-4,
                max_plateau=100,
                save_path="weights",
                model_id="unknown",
                loss_plot_path="loss_curve.png"):
    # check save path
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=16)

    # CHUNK DATA
    stride = chunk_len // 2

    def to_chunks(X, y):
        return chunk_data(torch.tensor(X).float(), torch.tensor(y).float(),
                          chunk_len_X=chunk_len, stride_X=stride)

    train_X_chunks, train_y_chunks = to_chunks(dataset.train_X, dataset.train_y)
    val_X_chunks, val_y_chunks = to_chunks(dataset.val_X, dataset.val_y)

    train_loader = DataLoader(TensorDataset(train_X_chunks, train_y_chunks),
                              batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_X_chunks, val_y_chunks),
                            batch_size=batch_size, shuffle=False, pin_memory=True)

    losses, val_losses = [], []
    best_val_loss = float('inf')
    plateau_counter = 0
    
    alpha = alpha_schedule[0]
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}", flush=True)
        model.train()
        total_loss = 0
        alpha = alpha_schedule.get(epoch, alpha)

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = combined_loss(output, batch_y, loss_fn1, loss_fn2, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        losses.append(avg_train_loss)

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                val_loss = combined_loss(output, batch_y, loss_fn1, loss_fn2, alpha)
                val_total_loss += val_loss.item()
        avg_val_loss = val_total_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            plateau_counter = 0
            model_path = os.path.join(save_path, model_id + "_best.pt")
            torch.save(model.state_dict(), model_path)
        else:
            plateau_counter += 1

        if plateau_counter >= max_plateau and max_plateau:
            print("Early stopping due to plateau on validation.")
            break

        torch.cuda.empty_cache()
    print('Report')
    print(f"Best model saved to {model_path}")
    print(f"End train Loss: {avg_train_loss:.4f} | End val Loss: {avg_val_loss:.4f} | Best val loss {best_val_loss:.4f}\n")
    #model_path = os.path.join(save_path, model_id + "_last.pt")
    #torch.save(model.state_dict(), model_path)
    #print(f"Last model saved to {model_path}")

    # Return model and test data (unchunked)
    test_X = torch.tensor(dataset.test_X).float()
    test_y = torch.tensor(dataset.test_y).float()
    # return best model
    model.load_state_dict(torch.load(model_path))
    return {'model':model, 
            'test_X':test_X,
            'test_y':test_y,
            'losses':losses,
            'val_losses':val_losses}


#------INFERENCE LOOPS-------------------------#



def inference(model: ECoGHuBERT, test_X, test_y, chunk_len=500, batch_size=8):
    '''
    Performs predictions on chunked data to save memory during inference
    '''
    device = next(model.parameters()).device
    # MAKE SURE THERE IS NO OVERLAP IN CHUNKING
    test_X_chunks, test_y_chunks = chunk_data(test_X, test_y, chunk_len, stride_X=chunk_len)

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

    SYLBER_FEAT_DIR = CONFIG['model']['sylber_outputs']
    full_sylber_features = open_pickle(SYLBER_FEAT_DIR)

    HIDDEN_STATES = full_sylber_features['hidden_states']
    del full_sylber_features

    subject_number = 9
    output_folder = "transformer_outputs"
    train_ratio = 0.8
    n_epochs = 64
    pass
