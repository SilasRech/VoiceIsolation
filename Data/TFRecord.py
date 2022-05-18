import math
import os
import numpy as np
import tensorflow as tf
import glob


_BASE_DIR = 'C:/Users/silas/NoisyOverlappingSpeakers/example.tfrecords'

_DEFAULT_OUTPUT_DIR = 'C:/Users/silas/NoisyOverlappingSpeakers/example.tfrecords'

_DEFAULT_DURATION = 6  # seconds
_DEFAULT_SAMPLE_RATE = 16000

_DEFAULT_NUM_SHARDS_TRAIN = 73

_SEED = 2020


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class TFRecordsConverter:
    """Convert audio to TFRecords."""
    def __init__(self, meta, output_dir, n_shards_train,
                 duration, sample_rate):
        self.output_dir = output_dir
        self.n_shards_train = n_shards_train
        self.duration = duration
        self.sample_rate = sample_rate

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.target_path = meta[0]
        self.far_end_path = meta[1]
        self.mixture_path = meta[2]

        self.n_train = 27000

    def _get_shard_path(self, split, shard_id, shard_size):
        return os.path.join(self.output_dir,
                            '{}-{:03d}-{}.tfrecord'.format(split, shard_id,
                                                           shard_size))

    def _write_tfrecord_file(self, shard_path, indices):
        """Write TFRecord file."""
        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
            for index in range(len(indices)):
                file_path_clean = self.target_path[index]
                file_path_far_end = self.far_end_path[index]
                file_path_mixture = self.mixture_path[index]

                raw_audio_clean = tf.io.read_file(file_path_clean)
                audio_clean, sample_rate = tf.audio.decode_wav(
                    raw_audio_clean,
                    desired_channels=1,  # mono
                    desired_samples=self.sample_rate * self.duration)

                raw_audio_far_end = tf.io.read_file(file_path_far_end)
                audio_far_end, sample_rate = tf.audio.decode_wav(
                    raw_audio_far_end,
                    desired_channels=1,  # mono
                    desired_samples=self.sample_rate * self.duration)

                raw_audio_mixture = tf.io.read_file(file_path_mixture)
                audio_mixture, sample_rate = tf.audio.decode_wav(
                    raw_audio_mixture,
                    desired_channels=1,  # mono
                    desired_samples=self.sample_rate * self.duration)

                # Example is a flexible message type that contains key-value
                # pairs, where each key maps to a Feature message. Here, each
                # Example contains two features: A FloatList for the decoded
                # audio data and an Int64List containing the corresponding
                # label's index.
                example = tf.train.Example(features=tf.train.Features(feature={
                    'target': _float_feature(audio_clean.numpy().flatten().tolist()),
                    'far_end': _float_feature(audio_far_end.numpy().flatten().tolist()),
                    'mix': _float_feature(audio_mixture.numpy().flatten().tolist())}))

                out.write(example.SerializeToString())

    def convert(self):
        """Convert to TFRecords.
        Partition data into training, testing and validation sets. Then,
        divide each data set into the specified number of TFRecords shards.
        """
        split = 'train'
        size = self.n_train
        n_shards = self.n_shards_train

        offset = 0

        print('Converting {} set into TFRecord shards...'.format(split))
        shard_size = math.ceil(size / n_shards)
        cumulative_size = offset + size
        for shard_id in range(1, n_shards + 1):
            step_size = min(shard_size, cumulative_size - offset)
            shard_path = self._get_shard_path(split, shard_id, step_size)
            # Generate a subset of indices to select only a subset of
            # audio-files/labels for the current shard.
            file_indices = np.arange(offset, offset + step_size)
            self._write_tfrecord_file(shard_path, file_indices)
            offset += step_size

        print('Number of training examples: {}'.format(self.n_train))
        print('TFRecord files saved to {}'.format(self.output_dir))


def main():
    path_train = 'C:/Users/silas/NoisyOverlappingSpeakers/Database/'

    list_samples = glob.glob(path_train + "/*.wav")

    clean_speaker_path = []
    target_speaker_path = []
    speech_mix_path = []

    for i in range((int(len(list_samples)/3))):
        clean_speaker_path.append(os.path.join(path_train, 'SpeakerSet_CleanSpeech{}.wav'.format(i)))
        target_speaker_path.append(os.path.join(path_train, 'SpeakerSet_FarEnd{}.wav'.format(i)))
        speech_mix_path.append(os.path.join(path_train, 'SpeakerSet_NearEnd{}.wav'.format(i)))

    meta_csv = [clean_speaker_path, target_speaker_path, speech_mix_path]

    output_dir = 'C:/Users/silas/NoisyOverlappingSpeakers/'
    n_shards_train = 73
    duration = 6
    sample_rate = 16000

    converter = TFRecordsConverter(meta_csv,
                                   output_dir,
                                   n_shards_train,
                                   duration,
                                   sample_rate)
    converter.convert()


if __name__ == '__main__':
    main()
