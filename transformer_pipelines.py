from neuro_transformers import *
import torch.nn as nn
from utils import CONFIG, Subject, SplitDataset, split_data, tsv_to_VAD
from pathlib import Path


SYLBER_FEAT_DIR = CONFIG['model']['sylber_outputs']
full_sylber_features = open_pickle(SYLBER_FEAT_DIR)

SYLBER_HIDDEN_STATES = full_sylber_features['hidden_states']
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


def train_sylber_pred(sub_num,
                      hidden_states=SYLBER_HIDDEN_STATES,
                      base_model='hubert', # or 'wav2vec'
                      model=None,
                      pretrained_shallow=None,
                      train_ratio=0.8,
                      val_ratio=0.1,
                      n_epochs=100,
                      loss_fn1=nn.MSELoss(),
                      loss_fn2=dot_product_loss
                      ):
    '''
    pretrained_shallow: path to pretrained weights for the shallow
    hubert layer
    '''
    if base_model == 'hubert':
        speech_upstream=CONFIG['model']['hubert_base_model']
    elif base_model == 'wav2vec':
        speech_upstream=CONFIG['model']['wav2vec_base_model']
    dataset = split_data(sub_num, hidden_states)
    if model is None:
        model = ECoGHuBERT(dataset.train_X,shape[1],
                           speech_upstream=speech_upstream)
    model_id = f'TOKPRED_base{base_model}_sub{sub_num}_T{int(train_ratio * 10):02}_V{int(val_ratio * 10):02}'
    schedule={0:1, 50:0.5, 100:0}
    model, (test_X, test_y) = train_model(model,
                                        dataset,
                                        loss_fn1,
                                        loss_fn2,
                                        alpha_schedule=schedule,
                                        n_epochs=n_epochs,
                                        max_plateau=0,
                                        model_id = model_id)

if __name__ == "__main__":
    for subject_number in range(1, 10): 
        VAD_ssl(subject_number)
