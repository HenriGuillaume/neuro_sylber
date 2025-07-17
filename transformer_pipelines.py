from neuro_transformers import *
import torch.nn as nn
from utils import CONFIG, Subject, SplitDataset, split_data, tsv_to_VAD
from pathlib import Path


SYLBER_FEAT_DIR = CONFIG['model']['sylber_outputs']
full_sylber_features = open_pickle(SYLBER_FEAT_DIR)

HIDDEN_STATES = full_sylber_features['hidden_states']
del full_sylber_features

output_folder = "transformer_outputs"

def VAD_ssl(sub_num,
            model=None,
            train_ratio=0.8,
            val_ratio=0.1,
            loss_fn1=nn.BCELoss(),
            loss_fn2=lambda y_pred, y_true: 0,
            ):
    '''
    Build a binary signal of voice activity during the audio using the provided
    transcript, to pretrain the shallow hubert layer on a VAD task.
    '''
    tsv_path = Path(CONFIG['data']['dataset_dir']) / 'stimuli' / 'phonetic' / 'transcript.tsv'
    VAD = tsv_to_VAD(tsv_path) 
    print('gathered VAD')
    dataset = split_data(sub_num, VAD)
    print(dataset.train_X.shape)
    if model is None:
        model = ECoGHuBERT_classifier(dataset.train_X.shape[1],
                                    dropout=0.1)
    model_id = f'VAD_sub{sub_num}_T{int(train_ratio * 10):02}_V{int(val_ratio * 10):02}'
    model, (test_X, test_y) = train_model(
                model,
                dataset,
                loss_fn1,
                loss_fn2,
                max_plateau=0,
                model_id = model_id,
                alpha_schedule={0:1})


if __name__ == "__main__":
    for subject_number in range(1, 10): 
        VAD_ssl(subject_number)
