from tools import splitStrings

import os
import torchaudio
import glob
from tqdm import tqdm
import random

def walk_through_files(path):
    test = os.path.join(path, "**" , '*.flac')
    return_list = glob.glob(test, recursive=True)

    list_possible_audio_files = []
    for path in return_list:
        minimum_file_length = 96000
        audiofile, sample_rate = torchaudio.load(path)

        if len(audiofile.T) >= minimum_file_length:
            list_possible_audio_files.append(path)

    #random_sample = random.choice(list_possible_audio_files)

    return list_possible_audio_files


def generate_meta_files(root, datapath, database_length, overwrite, num_speakers=2):

    root_dir = datapath
    root_dir_list_files = os.path.join(datapath, '**', '*.flac')
    print(root_dir_list_files)

    speaker_files = 'AudioFiles.txt'
    if not os.path.exists('AudioFiles.txt') or overwrite:
        files = glob.glob(root_dir_list_files, recursive=True)
        print("Number of files found: {}".format(len(files)))
        with open(speaker_files, 'w') as fp:
            for item in files:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('List of All Audio Files Saved')
    else:
        with open(speaker_files) as f:
            files = f.read().splitlines()

    with open(os.path.join(root, "SPEAKERS.txt")) as f:
        speakers = f.read().splitlines()

    list_female = []
    list_male = []
    for speaker in speakers:
        if 'dev-clean' in speaker:
            res = splitStrings(speaker, '|')
            if res[1] == ' F ':
                list_female.append(res[0])
            else:
                list_male.append(res[0])

    if not os.path.exists(os.path.join(root, 'PossibleFemaleSpeakers.txt')) or overwrite:
        all_files_female = []
        for m in tqdm(range(len(list_male))):
            possible_file_female = walk_through_files(os.path.join(root_dir, list_female[m].strip()))
            all_files_female = all_files_female + possible_file_female

        with open(os.path.join(root, 'PossibleFemaleSpeakers.txt'), 'w') as fp:
            for item in all_files_female:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('List of Female Speakers Saved')
    else:
        with open(os.path.join(root, 'PossibleFemaleSpeakers.txt')) as f:
            all_files_female = f.read().splitlines()

    if not os.path.exists(os.path.join(root, 'PossibleMaleSpeakers.txt')) or overwrite:
        all_files_male = []
        for m in tqdm(range(len(list_male))):
            possible_file_male = walk_through_files(os.path.join(root_dir, list_male[m].strip()))
            all_files_male = all_files_male + possible_file_male
        with open(os.path.join(root, 'PossibleMaleSpeakers.txt'), 'w') as fp:
            for item in all_files_male:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('List of Male Speakers Saved')
    else:
        with open(os.path.join(root, 'PossibleMaleSpeakers.txt')) as f:
            all_files_male = f.read().splitlines()

    # Get Distribution
    print("Number of available male files: {}".format(len(all_files_male)))
    print("Number of available female files: {}".format(len(all_files_female)))

    target_male = random.choices(all_files_male, k=database_length // 2)
    target_female = random.choices(all_files_female, k=database_length // 2)

    target_all = target_male + target_female
    interferer_all = []
    for k in range(num_speakers-1):

        interferer_male1 = random.choices(all_files_male, k=database_length // 4)
        interferer_male2 = random.choices(all_files_male, k=database_length // 4)
        interferer_female1 = random.choices(all_files_female, k=database_length // 4)
        interferer_female2 = random.choices(all_files_female, k=database_length // 4)

        interferer_all.append(interferer_male1 + interferer_female1 + interferer_male2 + interferer_female2)

    with open(os.path.join(root, "training_list.txt"), 'w') as fp:
        for i in range(len(target_all)):
            # write each item on a new line

            interferers = ' '.join([speaker[i] for speaker in interferer_all])
            fp.write(target_all[i]+ " " + interferers + '\n')

        print('Training List Saved and Finished')