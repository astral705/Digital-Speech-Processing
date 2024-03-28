import os
import sys
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from pathlib import Path


#get VAD repo path
VAD_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(VAD_PATH))
print(f"VAD_PATH:{VAD_PATH}")

suffix = ".wav"
subdirectory = 'wavs/dev'


from vad_utils import read_label_from_file


def frame_waveform(waveform, sample_rate, frame_size: float = 0.032, frame_shift: float = 0.008):
    # Convert frame length and shift from seconds to samples
    frame_length_samples = int(frame_size * sample_rate)
    frame_shift_samples = int(frame_shift * sample_rate)
    
    frames = []
    num_samples = len(waveform)
    idx = 0
    
    # Iterate over the waveform and segment into frames
    while idx + frame_length_samples <= num_samples:
        frame = waveform[idx:idx + frame_length_samples]
        frames.append(frame)
        idx += frame_shift_samples
    
    # Zero-pad the last frame if its length is less than frame length
    if idx < num_samples:
        last_frame = waveform[idx:]
        padding = np.zeros(frame_length_samples - len(last_frame))
        frames.append(np.concatenate([last_frame, padding]))
    
    return frames


def load_dev_data():
    label_data = read_label_from_file()
    speech_dict = {}
    dev_path = VAD_PATH / subdirectory
    
    for name, label in tqdm(label_data.items()):
        file_path = os.path.join(dev_path, name + suffix)
        sample_rate, speech_data = wavfile.read(file_path)
        frame_data = frame_waveform(speech_data, sample_rate)
        speech_dict[name] = frame_data
    
    return label_data, sample_rate, speech_dict


def extract():
    feature_list = []
    label_list = []
    data, sample_rate, speech_dict = load_dev_data()
    for name, value in tqdm(data.items()):
        speech_data = speech_dict[name]
        feature = extract_feature(speech_data, sample_rate)
        label = extract_label(feature, value)
        feature_list.append(feature)
        label_list.append(label)
    feature_list = np.concatenate(feature_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    return feature_list, label_list

def extract_feature(data, sample_rate):
    """
    each data is constructed with lots of frame
    data = [frame1, frame2, ...]
    """
    feature1 = short_time_energy(data)
    feature2 = short_time_zcr(data)
    feature3 = spectral_centroid(data, sample_rate)
    feature4 = estimate_pitch(data, sample_rate)
    #print(f"shape of feature 1 {feature1.shape}")
    return np.stack((feature1, feature2, feature3, feature4), axis=1)
"""    for frame in data:
        f1 = short_time_energy(frame)
        f2 = shor_time_zcr(frame)
        f3 = spectral_centroid(frame, sample_rate)
        f4 = estimate_pitch(frame, sample_rate)
        feature.append(np.concatenate([f1, f2, f3, f4], axis=0))"""


def short_time_energy(data):
    return np.sum(np.square(data), axis=1)


def short_time_zcr(data):
    return 1/2 * np.sum(np.diff(np.sign(data), axis = 1), axis = 1)


def spectral_centroid(data, sample_rate):
    """Calculate spectral centroid for each frame in the data.

    Args:
        data (list of ndarrays): List of frames, where each frame is a 1D numpy array.
        sample_rate (int): Sampling rate of the audio signal.

    Returns:
        centroids (ndarray): Array of spectral centroids corresponding to each frame.
    """
    data = np.array(data)
    data_fft = np.fft.fft(data, axis=1)
    freq_axis = np.fft.fftfreq(data.shape[1], d=1/sample_rate)
    spectrum = np.abs(data_fft)

    #check if there exists all 0 data
    valid_data = np.sum(spectrum, axis=1) != 0
    spectrum[~valid_data] = np.finfo(float).eps
    
    weighted_freq = freq_axis * spectrum
    centroids = np.sum(weighted_freq, axis=1) / np.sum(spectrum, axis=1)
    
    #constrain the centroids within a specified range
    return np.clip(centroids, None, sample_rate / 2)


def autocorr(x):
    return np.correlate(x, x, mode='full')[len(x)-1:]


"""
def estimate_pitch(data, sample_rate):
    data = np.array(data)
    autocorrs = np.apply_along_axis(autocorr, axis=1, arr=data)
    autocorrs = autocorrs[:, len(data[0])//2:]  # 取一半长度，因为自相关函数是对称的

    # 计算自相关函数中的峰值
    peak_indices = np.where((autocorrs[:, :-1] > autocorrs[:, 1:]) & 
                            (autocorrs[:, :-1] > autocorrs.mean(axis=1, keepdims=True)))

    # 根据峰值计算基频
    fundamental_freqs = np.zeros(len(data))
    for i in range(len(data)):
        indices = peak_indices[1][i]
        if isinstance(indices, np.ndarray) and len(indices) > 0:
            peak_index = indices[0]  # 选择第一个峰值
            fundamental_period = peak_index / sample_rate  # 基频对应的周期
            fundamental_freq = 1 / fundamental_period  # 基频对应的频率
            fundamental_freqs[i] = fundamental_freq

    return fundamental_freqs
    """
def estimate_pitch(data, sample_rate):
    freq_list = []
    for frame in data:
        # 计算帧的自相关函数
        autocorr = np.correlate(frame, frame, mode='full')
        
        # 取自相关函数在20Hz到2kHz范围内的部分
        min_freq = int(sample_rate / 2000)  # 最小频率对应的索引
        max_freq = int(sample_rate / 20)    # 最大频率对应的索引
        autocorr = autocorr[min_freq:max_freq]
        
        # 找到自相关函数中的最大值点
        fundamental_index = np.argmax(autocorr)
        
        # 将最大值点索引转换为基频（频率）
        fundamental_freq = sample_rate / (min_freq + fundamental_index)

        freq_list.append(fundamental_freq)
        
    return freq_list

def extract_label(feature, value):
    pad_length = max(0, len(feature) - len(value))
    label_pad = np.pad(value, (0, pad_length), 'constant', constant_values=(0, 0))
    return label_pad

if __name__ == "__main__":
    data, sample_rate, speech_dict = load_dev_data()
    print(list(data.keys())[0])


    for name, value in tqdm(data.items()):
        print(name)
        speech_data = speech_dict[name]
        feature = extract_feature(speech_data, sample_rate)
        print("=== feature extraction done===")
        for i in range(100):
            if feature[i][3]:
                print(f"feature of the {i}th frame: {feature[i]}")
        label = extract_label(feature, value)
        print("=== label extraction done===")
        break