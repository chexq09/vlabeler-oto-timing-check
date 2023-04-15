#!/usr/bin/env python
import os
import wave
import struct
import argparse
from typing import Tuple, List
from collections import namedtuple


WavInfo = namedtuple('WavInfo', ['num_channels', 'sample_width', 'frame_rate', 'num_frames'])
SampleWidthProp = namedtuple('SampleWidthProp',
                             ['format_str', 'offset', 'norm_base', 'min_value', 'max_value'])

SAMPLE_WIDTH_DICT = {
    1: SampleWidthProp('B', -128, 128, 0, 255),
    2: SampleWidthProp('h', 0, 32768, -32768, 32767),
    # 4: ('i', 0, 2147483648),  # Unsupported by wave library
}


def read_wav_file(input_file: str,
                  start: float=None,
                  end: float=None) -> Tuple[list, WavInfo]:
    """Load a WAV file.

    Args:
        input_file (str): The input filename.
        start (float, optional): The start position in milisecond.
            Defaults to None (means from 0.0).
        end (float, optional): The end position in milisecond.
            Defaults to None (means to the end of the file).

    Raises:
        ValueError: The sample width is not supported,
            or there are no frame between start and end.

    Returns:
        Tuple[list, WavInfo]: The WAV data and the information.
            The WAV data is a 2D list, the first dimension is the channel and the second is the data.
            The data will always be normalized to [-1, 1].
    """
    # load file
    with wave.open(input_file, 'rb') as input_wav:
        num_channels = input_wav.getnchannels()
        sample_width = input_wav.getsampwidth()
        frame_rate = input_wav.getframerate()
        num_frames = input_wav.getnframes()
        # get the start and end position
        start = 0 if start is None else round(start / 1000 * frame_rate)
        end = num_frames if end is None else round(end / 1000 * frame_rate)
        if start >= end:
            raise ValueError('No frame between start and end.')
        # seek the file and read frames
        input_wav.setpos(start)
        bytes_data = input_wav.readframes(end - start)
        num_frames = end - start
    # get information from sample_width
    prop = SAMPLE_WIDTH_DICT.get(sample_width)
    if prop is None:
        raise ValueError('Unsupported sample width: {}'.format(sample_width))
    # decode and to float
    length = num_channels * num_frames
    data = [(float(x) + prop.offset) / prop.norm_base
            for x in struct.unpack('<%d' % length + prop.format_str, bytes_data)]
    # split to different channels
    data = [[data[j * num_channels + i] for j in range(num_frames)] for i in range(num_channels)]
    return data, WavInfo(num_channels, sample_width, frame_rate, num_frames)


def merge_metronome_wav(voice_data: List[List],
                        voice_info: WavInfo,
                        metronome_data: List[List],
                        metronome_info: WavInfo,
                        metronome_count: int,
                        metronome_bpm: float,
                        metronome_weight: float,
                        metronome_end_pos: float) -> List[List]:
    # check args
    assert metronome_count >= 0, 'Metronome count is minus.'
    assert metronome_bpm > 0, 'BPM must more than 0.'
    assert 1 > metronome_weight > 0, 'Metronome weight must in (0, 1)'
    assert metronome_end_pos > 0, 'Metronome end position must > 0'
    if voice_info.frame_rate != metronome_info.frame_rate:
        raise NotImplementedError('Difference frame rate of voice and metronome is unsupported.')
    if len(metronome_data) != 1:
        raise NotImplementedError('Multi-channel metronome is unsupported.')
    # merge
    # TODO
    return voice_data


def write_wav_file(output_file: str,
                   data: List[List],
                   wav_info: WavInfo) -> None:
    """Write a WAV file.

    Args:
        output_file (str): The output filename.
        data (List[List]): The WAV data, same format as read_wav_file.
        wav_info (WavInfo): The information of the WAV.

    Raises:
        ValueError: The sample width is not supported.
    """
    # get information from sample_width
    prop = SAMPLE_WIDTH_DICT.get(wav_info.sample_width)
    if prop is None:
        raise ValueError('Unsupported sample width: {}'.format(wav_info.sample_width))
    # different channels to 1 list
    length = wav_info.num_channels * wav_info.num_frames
    data = [data[i % wav_info.num_channels][i // wav_info.num_channels] for i in range(length)]
    # to int and encode
    data = [max(min(round(x * prop.norm_base - prop.offset), prop.max_value), prop.min_value)
            for x in data]
    bytes_data = struct.pack('<%d' % length + prop.format_str, *data)
    with wave.open(output_file, 'wb') as output_wav:
        output_wav.setnchannels(wav_info.num_channels)
        output_wav.setsampwidth(wav_info.sample_width)
        output_wav.setframerate(wav_info.frame_rate)
        output_wav.setnframes(wav_info.num_frames)
        output_wav.setcomptype('NONE', 'Uncompressed')
        output_wav.writeframes(bytes_data)


def main():
    parser = argparse.ArgumentParser(description='Merge metronome to a WAV file.')
    parser.add_argument('metronome_file', metavar='METRONOME_FILE', type=str,
                        help='the path to the input metronome WAV file')
    parser.add_argument('metronome_count', metavar='METRONOME_COUNT', type=int,
                        help='the number of metronome repeats before the last one,\
                              (it will has METRONOME_COUNT + 1 repeats)')
    parser.add_argument('bpm', metavar='BPM', type=float,
                        help='the BPM of metronomes')
    parser.add_argument('metronome_weight', metavar='METRONOME_WEIGHT', type=float,
                        help='the weight of metronomes in (0, 1), \
                              the weight of voice will be 1-METRONOME_WEIGHT')

    parser.add_argument('voice_file', metavar='VOICE_FILE', type=str,
                        help='the path to the input voice WAV file')
    parser.add_argument('entry_start', metavar='ENTRY_START', type=float,
                        help='the entry start time in milisecond')
    parser.add_argument('entry_end', metavar='ENTRY_END', type=float,
                        help='the entry end time in milisecond')
    parser.add_argument('entry_offset', metavar='ENTRY_OFFSET', type=float,
                        help='the offset of entry in milisecond, \
                              the last metronomes will end at ENTRY_START-ENTRY_OFFSET')

    parser.add_argument('output_file', metavar='OUTPUT_FILE', type=str,
                        help='the path to the output WAV file')
    args = parser.parse_args()

    # check args
    assert os.path.exists(args.metronome_file), 'METRONOME_FILE not found'
    assert args.metronome_count > 0, 'METRONOME_COUNT must > 0'
    assert args.bpm > 0, 'BPM must > 0'
    assert 1 > args.metronome_weight > 0, 'METRONOME_WEIGHT must in (0, 1)'
    assert os.path.exists(args.voice_file), 'METRONOME_FILE not found'
    assert args.entry_start > 0, 'ENTRY_START must > 0'
    assert args.entry_end > 0, 'ENTRY_END must > 0'
    assert args.entry_offset < 0, 'ENTRY_OFFSET must > 0'
    assert args.metronome_file.lower().endswith('.wav') and args.voice_file.lower().endswith('.wav') \
        and args.output_file.lower().endswith('.wav'), 'input and output must by .WAV file'

    # load WAV files
    voice_data, voice_info = read_wav_file(args.voice_file, args.entry_start, args.entry_end)
    metronome_data, metronome_info = read_wav_file(args.metronome_file)

    # process
    merged_data = merge_metronome_wav(voice_data,
                                      voice_info,
                                      metronome_data,
                                      metronome_info,
                                      args.metronome_count + 1,
                                      args.bpm,
                                      args.metronome_weight,
                                      -args.entry_offset)

    # write WAV file
    write_wav_file(args.output_file, merged_data, voice_info)


if __name__ == '__main__':
    main()
