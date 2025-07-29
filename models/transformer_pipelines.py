import numpy as np
from neuro_transformers import *
import torch
import torch.nn as nn
import torchaudio
import soundfile
from utils import CONFIG, Subject, SplitDataset, split_data, tsv_to_VAD
from pathlib import Path


SYLBER_FEAT_DIR = CONFIG['model']['sylber']['outputs']
full_sylber_features = open_pickle(SYLBER_FEAT_DIR)

SYLBER_HIDDEN_STATES = full_sylber_features['hidden_states']
del full_sylber_features

WAV2VEC_FEAT_DIR = CONFIG['model']['wav2vec_outputs']
WAV2VEC_HIDDEN_STATES = np.load(WAV2VEC_FEAT_DIR)


def generate_wav2vec_hidden(wav_pth=CONFIG['data']['podcast_audio'], end_layer=6, chunk_sec=10):
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(CONFIG['model']['wav2vec_base_model'])
    model = Wav2Vec2Model.from_pretrained(CONFIG['model']['wav2vec_base_model'])
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load audio
    waveform, sample_rate = torchaudio.load(wav_pth)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze()
    sample_rate = 16000
    chunk_size = chunk_sec * sample_rate

    all_hidden = []

    for i in range(0, len(waveform), chunk_size):
        print(f'chunk {i}/{len(waveform) // chunk_size}')
        chunk = waveform[i:i+chunk_size]
        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            h = outputs.hidden_states[end_layer].squeeze(0).cpu()  # [seq_len, hidden_dim]
            all_hidden.append(h)

    layer6_activations = torch.cat(all_hidden, dim=0)  # [total_seq_len, hidden_dim]
    np.save('wav2vec_hidden.pkl', layer6_activations.numpy(), allow_pickle=True)
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
                      n_epochs=256,
                      loss_fn1=nn.MSELoss(),
                      loss_fn2=cossim_loss,
                      schedule={0:0.2}
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
    model_id = f'TOKPRED_base{base_model}_mod_{mode}_freeze{bool(freeze)}_sub0{sub_num}_T{int(train_ratio * 10):02}_V{int(val_ratio * 10):02}_ep{n_epochs}'
    model, (test_X, test_y) = train_model(model,
                                        dataset,
                                        loss_fn1,
                                        loss_fn2,
                                        alpha_schedule=schedule,
                                        n_epochs=n_epochs,
                                        max_plateau=32,
                                        model_id = model_id)
    preds = inference(model, test_X, test_y)
    save_predictions(preds, None, model_id)
    return model

if __name__ == "__main__":
    import sys
    # keep pretrained layers in mind across patients
    #mode_list = ('alpha', 'gamma', 'theta', 'beta', 'highgamma') 
    sub_num = int(sys.argv[1])
    base_model = sys.argv[2] # 'wav2vec' or 'hubert'
    mode = sys.argv[3]
    if base_model == 'hubert':
        hidden_states=SYLBER_HIDDEN_STATES
    elif base_model == 'wav2vec':
        hidden_states=WAV2VEC_HIDDEN_STATES
    else:
        raise ValueError("Unknown base model")
    print((sub_num, mode, base_model))
    model = train_sylber_pred(sub_num,
                              hidden_states=hidden_states,
                              base_model=base_model,
                              mode=mode,
                              freeze=[])
    torch.cuda.empty_cache()
    del model
