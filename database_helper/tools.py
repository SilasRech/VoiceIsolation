import numpy as np
import pandas as pd
import os
import soundfile as sf
import random


def walk_through_files(path):
    return_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".flac"):
                return_list.append(os.path.join(root, file))

    list_possible_audio_files = []
    for path in return_list:
        minimum_file_length = 96000
        audiofile, sample_rate = sf.read(path)

        if len(audiofile) >= minimum_file_length:
            list_possible_audio_files.append(path)

    random_sample = random.choice(list_possible_audio_files)

    return random_sample


def transform_speaker_text(text, subset):

    text_list = []
    del text[:12]

    for line in text:
        text_line = str.split(line, '|')
        text_line = [x.strip(' ') for x in text_line]
        text_list.append(text_line)

    speaker_frame = pd.DataFrame(text_list).iloc[:, :3]
    speaker_frame.columns = ['ID', 'SEX', 'SUBSET']
    return speaker_frame.loc[speaker_frame['SUBSET'] == subset]


def subtract_elements_in_list(list1, list2):
    return list(np.asarray(list1) - np.asarray(list2))
