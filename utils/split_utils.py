import os
import sys
import pickle
import numpy as np
from sylber import Segmenter
from pydub import AudioSegment
import soundfile as sf
from textgrid import TextGrid, IntervalTier

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
    outputs = segmenter(filename)

    # Prepare output path
    base = os.path.splitext(os.path.basename(filename))[0]
    os.makedirs("pickle", exist_ok=True)
    pickle_path = os.path.join("pickle", f"{base}.pickle")

    with open(pickle_path, "wb") as f:
        pickle.dump(outputs, f)

    print(f"Saved output to {pickle_path}")
    return pickle_path


def merge_sylber_dicts(pickle_paths, audio_part_paths, sample_rate=16000):
    merged_segments = []
    merged_features = []
    merged_hidden = []

    offset = 0  # in samples

    for pickle_path, audio_path in zip(pickle_paths, audio_part_paths):
        with open(pickle_path, 'rb') as f:
            d = pickle.load(f)

        # shift segments by current offset
        seg = d['segments'] + offset
        merged_segments.append(seg)
        merged_features.append(d['segment_features'])
        merged_hidden.append(d['hidden_states'])

        # update offset using number of samples in this audio part
        num_samples = sf.info(audio_path).frames
        offset += num_samples

    merged = {
        'segments': np.concatenate(merged_segments),
        'segment_features': np.concatenate(merged_features),
        'hidden_states': np.concatenate(merged_hidden),
    }

    return merged


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
    #audio_file = "./samples/podcast.wav"

    #print("Splitting audio...")
    #part_files = split_audio_by_timepoints(audio_file, split_points)

    #print("\nRunning Sylber on each part...")
    #pickle_paths = [run_sylber_on_file(f) for f in part_files]

    #print("\nMerging Sylber outputs...")
    #merged_output = merge_sylber_dicts(pickle_paths, part_files)

    ## Optional: save merged result
    #with open("pickle/merged_output.pickle", "wb") as f:
    #    pickle.dump(merged_output, f)

    #print("Merged output saved to pickle/merged_output.pickle")

    # save to TextGrid
    with open("outputs.pkl", "rb") as f:
        merged_output = pickle.load(f)

    segments = merged_output['segments']
    
    segments_to_textgrid(segments, "segments.TextGrid")

