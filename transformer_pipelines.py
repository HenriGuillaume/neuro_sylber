from neuro_transformers import *
import torch.nn as nn
from utils import Subject, SplitDataset, split_data


SYLBER_FEAT_DIR = CONFIG['model']['sylber_outputs']
full_sylber_features = open_pickle(SYLBER_FEAT_DIR)

HIDDEN_STATES = full_sylber_features['hidden_states']
del full_sylber_features

output_folder = "transformer_outputs"

def next_avg_frame_pred(sub_num,
                        model=None,
                        train_ratio=0.8,
                        val_ratio=0.1,
                        loss_fn1=nn.MSELoss(),
                        loss_fn2=lambda y_pred, y_true: 0,
                        ):
    dataset = split_data(sub_num, HIDDEN_STATES)
    if model is None:
        model = ECoG_HuBERT_classifier(dataset.train_X.shape[1])
    model, (test_X, test_y) = train_model(
                model,
                dataset,
                loss_fn1,
                loss_fn2,
                alpha_schedule={0:1})


if __name__ == "__main__":
    subject_number = 1
    next_avg_frame_pred(subject_number)
