import os
import sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler
from tqdm import tqdm

#get VAD repo path
VAD_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(VAD_PATH))

test_subdirectory = 'wavs/test'

from feature_extraction import extract, extract_feature, frame_waveform
from feature_extraction import subdirectory, suffix
from feature_extraction import load_dev_data
from vad_utils import mean_filtering
from evaluate import get_metrics


def load_test_data():
    speech_dict = {}
    test_path = VAD_PATH / test_subdirectory
    for filename in os.listdir(test_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(test_path, filename)
            sample_rate, speech_data = wavfile.read(file_path)
            frame_data = frame_waveform(speech_data, sample_rate)
            speech_dict[filename.split('.')[0]] = frame_data
            #print(len(speech_data))
            #xsprint(len(frame_data))
            #print(f"frame: {frame_data[:1]}")
            print(f"label length {len(label)}")
            #print(f"label: {label}")
            #print(list(speech_dict.keys())[0])
            break
    return sample_rate, speech_dict


def feature_normalize(feature_list):
    first_two_features = feature_list[:, :2]
    third_feature = feature_list[:, 2]
    fourth_feature = feature_list[:, 3]

    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler2 = StandardScaler()
    scaler3 = RobustScaler()

    normalized_first_two_features = scaler1.fit_transform(first_two_features)

    standardized_third_feature = scaler2.fit_transform(third_feature.reshape(-1, 1)).flatten()
    robust_scaled_fourth_feature = scaler3.fit_transform(fourth_feature.reshape(-1, 1)).flatten()

    feature_list[:, :2] = normalized_first_two_features
    feature_list[:, 2] = standardized_third_feature
    feature_list[:, 3] = robust_scaled_fourth_feature

    return feature_list


def estimate(classifier, label):
    predict_list = []
    #data represents label data
    data, sample_rate, speech_dict = load_dev_data()
    for name, value in tqdm(data.items()):
        speech_data = speech_dict[name]
        feature = extract_feature(speech_data, sample_rate)
        feature = feature_normalize(feature)
        probs = mean_filtering(classifier.predict_proba(feature)[:, 1])
        states = get_states(probs)
        predict_list.append(states)

    predict_list = np.concatenate(predict_list, axis=0)
    print(f"length is {len(predict_list)}")
    print("=== predict done ===")
    auc ,eer = get_metrics(predict_list, label)
    print(f"eer is {eer}")
    print("==== evaluate done ====")


def get_states(probs, k = 6):
    """
    prob is an array from classifer.predict_proba pass through mean_filtering
    we define state
    1 represents speech 
    0 represents silence
    """
    state = []
    for i in range(len(probs)):
        if i < k:
            state.append(int(np.mean(probs[:k]) > 0.5))
        elif i > len(probs) - k - 1:
            state.append(int(np.mean(probs[-k:]) > 0.5))
        else:
            state.append(int(np.mean(probs[i-k:i+k]) > 0.5))
    return state


if __name__ == "__main__":
    feature, label = extract()

    print("=== extract done===")

    count_0 = np.sum(label == 0)
    count_1 = np.sum(label == 1)

    feature = feature_normalize(feature)
    print(f"the length of feature is {len(feature)}")
    print(f"feature is: {feature[:100]}")
    
    print("=== normalize done===")

    #train step
    linear_model = LogisticRegression(class_weight="balanced")
    linear_model.fit(feature, label)

    estimate(linear_model, label)

    print(f"0 count is {count_0}")
    print(f"1 count is {count_1}")
