import numpy as np
from collections import OrderedDict
import math

chn = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6',
  'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

def calculate_eucl_distances(montage):

    eucl_distances = {}
    ch_pos = montage.get_positions()["ch_pos"]
    for i in range(len(chn)):
        dist_dict = OrderedDict()
        for j in range(len(chn)):
            if i != j:
                xi = ch_pos[chn[i]][0]
                xj = ch_pos[chn[j]][0]
                yi = ch_pos[chn[i]][1]
                yj = ch_pos[chn[j]][1]
                zi = ch_pos[chn[i]][2]
                zj = ch_pos[chn[j]][2]
                eucl_dist = np.sqrt( ((xi - xj) ** 2) + ((yi - yj) ** 2) + ((zi - zj) ** 2))
                dist_dict[chn[j]] = eucl_dist
        dist_dict = sorted(dist_dict.items(), key=lambda x: x[1], reverse=True)
        eucl_distances[chn[i]] = dist_dict
    return eucl_distances


def calculate_interpolated_signal(data, index, eucl_distances, channel_to_interpolate,  chunksize, power_parameter, neighboring_channels):

    interpolated_signal = []
    beg_idx = index * chunksize
    channel = chn[channel_to_interpolate]
    for idx in range(beg_idx, beg_idx + chunksize):
        numerator = 0
        dominator = 0
        counter = 0
        for key, value in eucl_distances[channel]:
            other_channel_idx = None
            for _ in range(len(chn)):
                if chn[_] == key:
                    other_channel_idx = _
                    break

            _weight = (1/(value ** power_parameter))
            numerator += _weight * data[other_channel_idx][idx]
            dominator += _weight
            if counter > neighboring_channels:
                break
        interpolated_signal.append(numerator / dominator)

    return interpolated_signal


def drop_bad_intervals(data, frequency, starts, durations):

    data_copy = data
    data_return = []

    for idx, st in enumerate(starts):
        beg_idx = int(st*frequency)
        end_idx = beg_idx + int(durations[idx]*frequency)
        length = end_idx - beg_idx

        for d in data_copy:
            d[beg_idx: end_idx] = [None] * length

    for da in data_copy:
        data_return.append(np.array([d for d in da if not math.isnan(d)]))

    return np.array(data_return)


def interpolate_signals(raw, montage, chunk_fraction=16, power_parameter=5, neighboring_channels=2, std_limit=0.00001):
    """
    If the signal of a channel in a chunk is bad based on the chunk's std it interpolates from its neighboring channels.
    If any of the neighboring channel is also bad in that chunk, it cuts the entire slice out from the data.
    :param raw:  raw signal
    :param montage: montage
    :param chunk_fraction: how many chucks per second, default 16
    :param power_parameter: power parameter, default 5
    :param neighboring_channels: number of closest channel to interpolate a bad signal, default 2
    :param std_limit: if the std of a chunk is bigger than this limit, the signal is bad, default 0.00001
    :return: filtered and cutted signals
    """
    "Signal filtering has started!"
    "\tEucledian distance calculation has started!"
    eucl_distances = calculate_eucl_distances(montage)
    max_eucledian_weights = {}

    for i in range(len(chn)):
        chname = chn[i]
        distances = eucl_distances[chname]
        counter = 0
        valsum = 0
        for key, val in distances:
            valsum += 1/(val ** power_parameter)
            counter += 1
            if counter > neighboring_channels:
                break
        max_eucledian_weights[chname] = valsum

    data = raw.get_data()
    max_length = len(data[0])
    raw_freq = raw.info['sfreq']
    chunksize = int(raw_freq // chunk_fraction)

    chunks = [range(x, x + chunksize) for x in range(0, max_length, chunksize)]

    correct_intervals = {}
    print("\t Calculating of signal quality has started!")
    #calculates if the quality of the signal is acceptable or not at each interval per channel
    for i in range(len(chn)):
        intervals = []
        for idx, c in enumerate(chunks):
            data_std = np.std(data[i][c[0]: c[-1]])
            if data_std < std_limit:
                intervals.append(True)
            else:
                intervals.append(False)
        correct_intervals[chn[i]] = intervals

    chunks_to_cut = [False] * len(correct_intervals[chn[0]])
    print(f"\t Interpolating of bad signals has started!")
    for i in range(len(chn)):
        chname = chn[i]
        interval_arr = correct_intervals[chname]
        for idx, interv in enumerate(interval_arr):
            if interv is False:
                max_weight = max_eucledian_weights[chname]
                distances = eucl_distances[chname]
                summed_weight = 0
                counter = 0
                for key, val in distances:
                    counter += 1
                    if correct_intervals[key][idx] is True:
                        _weight = val
                        summed_weight += 1/(_weight ** power_parameter)
                    if counter > neighboring_channels:
                        break
                if summed_weight == max_weight:
                    interp_sign = calculate_interpolated_signal(data, idx, eucl_distances, i, chunksize, power_parameter=power_parameter, neighboring_channels=neighboring_channels)
                    data[i][chunksize * idx: chunksize * idx + chunksize] = interp_sign
                else:
                    if chunks_to_cut[idx] == False:
                        chunks_to_cut[idx] = True

    data_to_filter = data
    print("\t Cutting of bad signals has started!")
    for idx, chunk in enumerate(chunks_to_cut):
        if chunk is True:
            b_idx = chunksize * idx
            e_idx = chunksize * idx + chunksize

            for i in range(len(chn) + 1):  ## + 1 because of the 'stimulus' channel
                data_to_filter[i][b_idx:e_idx] = [None] * chunksize

    data_cutted = []

    for i in range(len(data_to_filter)):
        temp_data = [d for d in data_to_filter[i] if not math.isnan(d)]
        data_cutted.append(np.array(temp_data))

    data_cutted = np.array(data_cutted)

    dclen = len(data_cutted[0])
    data_cutted = np.reshape(data_cutted, (len(chn) + 1,dclen))
    print("\t New data is ready!")
    return data_cutted