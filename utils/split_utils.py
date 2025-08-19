import os
import sys
import pickle
import yaml
import numpy as np
from pydub import AudioSegment
import soundfile as sf
from textgrid import TextGrid, IntervalTier
import h5py

with open("../config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

sys.path.append('../' + CONFIG['model']['sylber']['dir'])
from sylber import Segmenter

def split_audio_by_timepoints(
    input_file, split_points, output_prefix='part', target_sr=16000
):
    dir_path = os.path.dirname(input_file)
    # convert hh:mm:ss or mm:ss to milliseconds
    def time_to_ms(time_str):
        parts = list(map(int, time_str.split(":")))
        if len(parts) == 2:
            m, s = parts
            return (m * 60 + s) * 1000
        elif len(parts) == 3:
            h, m, s = parts
            return (h * 3600 + m * 60 + s) * 1000
        else:
            raise ValueError("Invalid time format")

    # resample
    audio = AudioSegment.from_file(input_file)
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)

    # Convert split points to ms and add 0 at the beginning
    split_ms = [0] + [time_to_ms(t) for t in split_points] + [len(audio)]
    
    output_files = []
    for i in range(len(split_ms) - 1):
        start = split_ms[i]
        end = split_ms[i + 1]
        segment = audio[start:end]
        filename = f"{output_prefix}{i + 1}.wav"
        filepath = os.path.join(dir_path, filename)
        segment.export(filepath, format="wav") # defaults to pcm_16le
        print(f"Saved: {filename} ({(end-start)/1000:.2f}s)")
        output_files.append(filepath)
    
    # save resampled audio
    resampled_audio_path = os.path.join(dir_path, f'{output_prefix}_resampled.wav')
    audio.export(resampled_audio_path, format='wav')

    return output_files


def run_sylber_on_file(filename):
    print(f"Processing {filename} ...")

    segmenter = Segmenter(model_ckpt="sylber", in_second=False)
    outputs = segmenter(filename, in_second=False)

    # Prepare output path
    base = os.path.splitext(os.path.basename(filename))[0]
    os.makedirs("pickle", exist_ok=True)
    pickle_path = os.path.join("pickle", f"{base}.pickle")

    with open(pickle_path, "wb") as f:
        pickle.dump(outputs, f)

    print(f"Saved output to {pickle_path}")
    return pickle_path


def merge_sylber_dicts(pickle_paths, audio_part_paths, hdf5_path, audio_sr=16000):
    """
    Merge sylber dicts into a single HDF5 file.
    Matches original reshape: (time, layers, hidden_dim).
    """
    # ===== PASS 1: find dimensions =====
    total_time = 0
    total_segments = 0
    with open(pickle_paths[0], "rb") as f:
        first_d = pickle.load(f)
    
    with open(pickle_paths[0], "rb") as f:
        first_d = pickle.load(f)

    # all_hidden_states: (num_layers, time, hidden_dim)
    num_layers, time_len, hidden_dim = first_d["all_hidden_states"].shape
    feature_dim = first_d["segment_features"].shape[1]

    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as f:
            d = pickle.load(f)
        total_time += d["all_hidden_states"].shape[1]  # seq_len
        total_segments += len(d["segments"])

    # ===== Create HDF5 datasets =====
    with h5py.File(hdf5_path, "w") as h5f:
        dset_hidden = h5f.create_dataset(
            "all_hidden_states",
            shape=(total_time, num_layers, hidden_dim),
            dtype="float32"
        )
        dset_segments = h5f.create_dataset(
            "segments",
            shape=(total_segments, 2),
            dtype="int64"
        )
        dset_features = h5f.create_dataset(
            "segment_features",
            shape=(total_segments, feature_dim),
            dtype="float32"
        )

        # ===== PASS 2: write reshaped data bit-by-bit =====
        time_offset = 0
        seg_offset = 0
        for pickle_path, audio_path in zip(pickle_paths, audio_part_paths):
            with open(pickle_path, "rb") as f:
                d = pickle.load(f)

            seq_len = d["all_hidden_states"].shape[1]
            seg_len = len(d["segments"])

            # store segments and features
            dset_segments[seg_offset:seg_offset+seg_len] = d["segments"] + time_offset
            dset_features[seg_offset:seg_offset+seg_len] = d["segment_features"]

            # write hidden states in (time, layer, hidden) order
            for layer_idx, layer_data in enumerate(d["all_hidden_states"]):
                dset_hidden[time_offset:time_offset+seq_len, layer_idx, :] = layer_data


            time_offset += seq_len
            seg_offset += seg_len

    return hdf5_path


def segments_to_textgrid(segments, output_path, sample_rate=16000, tier_name="segments"):
    if segments.ndim != 2 or segments.shape[1] != 2:
        raise ValueError("segments should be an Nx2 array of start and end sample indices")

    # Convert to seconds
    segments_sec = segments

    # Create a TextGrid and IntervalTier
    tg = TextGrid(minTime=0.0, maxTime=float(np.max(segments_sec)))
    tier = IntervalTier(name=tier_name, minTime=0.0, maxTime=float(np.max(segments_sec)))
    
    last_end = 0.0
    for i, (start, end) in enumerate(segments_sec):
        start = float(start)
        end = float(end)

        # If overlap with previous, snap start to midpoint between last_end and current end
        if start < last_end:
            midpoint = (last_end + end) / 2
            start = last_end
            end = max(midpoint, start + 1e-6)

        label = f"seg_{i+1}"
        tier.add(minTime=start, maxTime=end, mark=label)
        last_end = end

    tg.append(tier)

    # Save TextGrid
    tg.write(output_path)
    return output_path
    

if __name__ == "__main__":
    ## Split points chosen within silent intervals
    #split_points = ["00:05:00", "00:09:55", "00:15:28", "00:19:36", "00:25:22"]
    audio_files = []
    audio_pth = '/home/bigh/prog/neuro_sylber/pickled_podcast/samples'
    for i in os.listdir(audio_pth):
        if os.path.isfile(os.path.join(audio_pth,i)) and 'podcast_nomusic_' in i:
            audio_files.append(os.path.join(audio_pth, i))
    audio_files.sort()
    print(audio_files)
    pickle_outputs = [run_sylber_on_file(f) for f in audio_files]
    merged = merge_sylber_dicts(pickle_outputs, audio_files, 'pickle/merged.h5')
    
    print(f'Merged and saved to {merged}')