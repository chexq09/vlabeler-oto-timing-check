#!/usr/bin/env python
import os
import wave
import struct
import argparse
from collections import namedtuple
from typing import Tuple, List, Union


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
                  end: float=None) -> Tuple[List[List[float]], WavInfo]:
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
        Tuple[List[List[float]], WavInfo]: The WAV data and the information.
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


def _merge_single_channel(voice_data: List[float],
                          metronome_data: List[float],
                          metronome_weight: float,
                          merge_idx_list: List[int],
                          alignment: Union[int, str]=0,
                          padding: Union[int, str]='auto') -> List[float]:
    """Merge single channel.

    Args:
        voice_data (List[float]): The voice data.
        metronome_data (List[float]): The metronome data.
        metronome_weight (float): The weight of the metronome in (0, 1).
        merge_idx_list (List[int]): Index list of the voice data before padding.
            Metronome will be merged to each index.
        alignment (Union[int, str], optional): The aligment of the metronome data.
            metronome_data[alignment] will be merged at index in merge_idx_list.
            If 'center', align at the center of the metronome data.
            Defaults to 0.
        padding (Union[int, str], optional): Pad on the left of the voice.
            If 'auto', pad to the start of the first click.
            Must >= 0, Defaults to 'auto'.

    Raises:
        NotImplementedError: The alignment or padding is not implemented.

    Returns:
        List[float]: Merged data.
    """
    # get alignment index
    if alignment == 'center':
        alignment = len(metronome_data) // 2
    elif not isinstance(alignment, int):
        raise NotImplementedError(alignment)

    # apply weight
    voice_weight = 1.0 - metronome_weight
    merge_data = [x * voice_weight for x in voice_data]
    weighted_data = [x * metronome_weight for x in metronome_data]

    # padding
    if padding == 'auto':
        padding = max(0, -(min(merge_idx_list) - alignment))
    elif not isinstance(padding, int):
        raise NotImplementedError(padding)
    assert padding >= 0, 'padding must >= 0'
    if padding > 0:
        merge_data = [0.0 for _ in range(padding)] + merge_data
        merge_idx_list = [x + padding for x in merge_idx_list]

    # merge
    for merge_idx in merge_idx_list:
        voice_start = merge_idx - alignment
        voice_end = voice_start + len(metronome_data)
        metronome_start = 0

        # move to inside
        if voice_start < 0:
            metronome_start = -voice_start
            voice_start = 0
        if voice_end > len(merge_data):
            voice_end = len(merge_data)

        # check position
        if voice_end <= voice_start:
            continue

        # merge
        for i in range(voice_start, voice_end):
            j = metronome_start + i - voice_start
            merge_data[i] = merge_data[i] + weighted_data[j]
            merge_data[i] = min(max(merge_data[i], -1.0), 1.0)

    return merge_data


def merge_metronome_wav(voice_data: List[List[float]],
                        voice_info: WavInfo,
                        metronome_data: List[List[float]],
                        metronome_info: WavInfo,
                        metronome_count: int,
                        metronome_bpm: float,
                        metronome_weight: float,
                        metronome_end_pos: float) -> Tuple[List[List[float]], WavInfo]:
    """Merge metronome to voice.

    Args:
        voice_data (List[List[float]]): The voice data.
        voice_info (WavInfo): The WAV data of voice data.
        metronome_data (List[List[float]]): The metronome data.
        metronome_info (WavInfo): The WAV data of metronome data.
        metronome_count (int): The count of all metronome clicks.
        metronome_bpm (float): The BMP of metronome.
        metronome_weight (float): The weight of the metronome in (0, 1).
        metronome_end_pos (float): The position of the last metronome click.

    Raises:
        NotImplementedError: Difference frame rate of voice and metronome is unsupported.
        NotImplementedError: Multi-channel metronome is unsupported.

    Returns:
        Tuple[List[List[float]], WavInfo]: Merged data and the new WAV data.
    """
    # check args
    assert metronome_count >= 0, 'Metronome count is minus.'
    assert metronome_bpm > 0, 'BPM must more than 0.'
    assert 1 > metronome_weight > 0, 'Metronome weight must in (0, 1)'
    assert metronome_end_pos > 0, 'Metronome end position must > 0'
    if voice_info.frame_rate != metronome_info.frame_rate:
        # TODO
        raise NotImplementedError('Difference frame rate of voice and metronome is unsupported.')
    if len(metronome_data) != 1:
        raise NotImplementedError('Multi-channel metronome is unsupported.')
    metronome_data = metronome_data[0]

    # calculate the position of the metronome
    interval = 60000.0 / metronome_bpm  # milisecond
    merge_pos_list = [metronome_end_pos - interval * float(i) for i in range(metronome_count)]
    merge_idx_list = [round(x / 1000 * voice_info.frame_rate) for x in merge_pos_list]

    # merge
    merge_data = [_merge_single_channel(data,
                                        metronome_data,
                                        metronome_weight,
                                        merge_idx_list,
                                        padding='auto')
                  for data in voice_data]
    merge_info = WavInfo(voice_info.num_channels,
                         voice_info.sample_width,
                         voice_info.frame_rate,
                         len(merge_data[0]))
    return merge_data, merge_info


def write_wav_file(output_file: str,
                   data: List[List[float]],
                   wav_info: WavInfo) -> None:
    """Write a WAV file.

    Args:
        output_file (str): The output filename.
        data (List[List[float]]): The WAV data, same format as read_wav_file.
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

    # output
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
    merged_data, merge_info = merge_metronome_wav(voice_data,
                                                  voice_info,
                                                  metronome_data,
                                                  metronome_info,
                                                  args.metronome_count + 1,
                                                  args.bpm,
                                                  args.metronome_weight,
                                                  -args.entry_offset)

    # write WAV file
    write_wav_file(args.output_file, merged_data, merge_info)


if __name__ == '__main__':
    main()
