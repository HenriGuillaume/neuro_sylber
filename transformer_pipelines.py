import numpy as np
from neuro_transformers import *
import torch.nn as nn
import torchaudio
from utils import CONFIG, Subject, SplitDataset, split_data, tsv_to_VAD
from pathlib import Path


SYLBER_FEAT_DIR = CONFIG['model']['sylber_outputs']
full_sylber_features = open_pickle(SYLBER_FEAT_DIR)

SYLBER_HIDDEN_STATES = full_sylber_features['hidden_states']
del full_sylber_features

output_folder = "transformer_outputs"


def generate_wav2vec_hidden(wav_pth=CONFIG['data']['podcast_audio'], end_layer=6):
    from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(CONFIG['model']['wav2vec_base_model'])
    model = Wav2Vec2Model.from_pretrained(CONFIG['model']['wav2vec_base_model'])
    model.eval()
    # Load audio
    waveform, sample_rate = torchaudio.load(wav_pth)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # Tuple of (layer0, layer1, ..., layerN)
    layer6_activations = hidden_states[end_layer]  # Shape: (batch_size, sequence_length, hidden_size)
    
    np.save('wav2vec_hidden.pkl', layer6_activations, allow_pickle=True)
    return layer6_activations


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
                      mode='hg', # or 'clean'
                      model=None,
                      pretrained_shallow=None,
                      freeze=[],
                      train_ratio=0.8,
                      val_ratio=0.1,
                      n_epochs=64,
                      loss_fn1=nn.MSELoss(),
                      loss_fn2=dot_product_loss,
                      schedule={0:1, 16:0}
                      ):
    '''
    pretrained_shallow: path to pretrained weights for the shallow
    hubert layer
    '''
    if base_model == 'hubert':
        speech_upstream=CONFIG['model']['hubert_base_model']
    elif base_model == 'wav2vec':
        speech_upstream=CONFIG['model']['wav2vec_base_model']
    dataset = split_data(sub_num, hidden_states, mode=mode)
    if model is None:
        model = ECoGHuBERT(dataset.train_X.shape[1],
                           speech_upstream=speech_upstream,
                           freeze=freeze)
    model_id = f'TOKPRED_base{base_model}_mod_{mode}_freeze{bool(freeze)}_sub{sub_num}_T{int(train_ratio * 10):02}_V{int(val_ratio * 10):02}'
    model, (test_X, test_y) = train_model(model,
                                        dataset,
                                        loss_fn1,
                                        loss_fn2,
                                        alpha_schedule=schedule,
                                        n_epochs=n_epochs,
                                        max_plateau=0,
                                        model_id = model_id)
    preds = inference(model, test_X, test_y)
    save_predictions(preds, test_y, model_id)
    return model

if __name__ == "__main__":
    generate_wav2vec_hidden()
    exit()
    model = None
    for sub_num in range(1, 2):
        model = train_sylber_pred(sub_num,
                                  mode='hg',
                                  freeze=[]) # reuse previous weights when possible
    #for subject_number in range(1, 10): 
    #    VAD_ssl(subject_number)
