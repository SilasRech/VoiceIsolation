import numpy as np
import pandas as pd
import os


def walk_through_files(path):
    return_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                return_list.append(os.path.join(root, file))

    for path in return_list:
        longest_file = 0
        with open(path)as f:
            text = f.readlines()
            length_audiofile = len(max(text, key=len))

            if length_audiofile > longest_file:
                longest_file = length_audiofile
                audiofile_idx = text.index(max(text, key=len))

    return os.path.join(root, files[audiofile_idx])


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
