import mne
import numpy as np
import matplotlib.pyplot as plt
import mne_microstates
import os
import pandas as pd
import matplotlib
import preprocesser as preprocesser
from pathlib import Path
import scipy
import fnmatch
import sys
matplotlib.use('TkAgg')

def _categorize_spatial_correlation(data, peaks, reference_df):


    data_temp = data[:32]

    microstates = []
    states = [st for st in reference_df.columns.tolist() if "Unnamed:" not in st]
    for p in range(len(data[0])):
        abs_diff = [0] * 5
        for idx, state in enumerate(states):
            print(f"{p} - {state}")
            electrode_values = [abs(d[p]) for d in data_temp]
            electrode_global_values = abs(reference_df[state].values)
            abs_diff[idx] = scipy.spatial.distance.correlation(electrode_values, electrode_global_values)

        min_index = abs_diff.index(min(abs_diff))
        microstates.append(states[min_index])

    change_dict = {"microstate_A": 1, "microstate_B": 2, "microstate_C": 3, "microstate_D": 4, "microstate_D2": 5}

    return [change_dict[m] for m in microstates]


def _plot_maps(maps, info, segmentation, outfile=None):
    """Plot prototypical microstate maps.

    Parameters
    ----------
    maps : ndarray, shape (n_channels, n_maps)
        The prototypical microstate maps.
    info : instance of mne.io.Info
        The info structure of the dataset, containing the location of the
        sensors.
    """

    row = len(maps) // 4 + 1
    col = 4
    plt.figure(figsize=(2 * col, row * 2))
    seg_len = len(segmentation)
    for i, map in enumerate(maps):
        print(i)
        print(map)
        letter = "A" if i == 0 else "B" if i == 1 else "C" if i == 2 else "D" if i == 3 else i
        cluster_count = np.count_nonzero(segmentation == i)
        plt.subplot(row, col, i + 1)
        plt.title(f"{i} - {cluster_count} == {round((cluster_count / seg_len) * 100, 2)}%")
        if outfile:
            mne.viz.plot_topomap(map, info, show=False, ch_type="eeg")
            plt.savefig(outfile + f"_{letter}.png")
        else:
            mne.viz.plot_topomap(map, info, show=True, ch_type="eeg")




def connect_microstates(maps_dirpath=None, reference_df=None, outdir=None):
    """
    Connects the maps in maps_dirpath in a .csv file if there's a reference file.
    Reference file is a must!
    If there is no reference file, it has to be done by hand.
    :param maps_dirpath: directory for maps
    :param reference_df: directory to reference maps
    :param outdir: directory for the outfile
    :return: None
    """

    ref_df = reference_df

    change_dict = {"1": "A", "2": "B", "3": "C", "4": "D"}
    list_to_write = []
    for maps in os.listdir(maps_dirpath):
        if maps.endswith(".csv"):
            f_id = "_".join(maps.split("_")[:-1])
            maps_df = pd.read_csv(f"{maps_dirpath}/{maps}")
            best_dict = {}
            for column_name, column_data in maps_df.iteritems():
                if column_name != "Unnamed: 0":
                    best_score = 9999999
                    best_microstate = None
                    inverse = False

                    for ref_name, ref_data in ref_df.iteritems():
                        if "microstate" in ref_name:
                            spatial_corr =  scipy.spatial.distance.correlation(column_data, ref_data)
                            spatial_corr_inv =  scipy.spatial.distance.correlation(column_data * - 1, ref_data)
                            if spatial_corr < best_score:
                                best_score = spatial_corr
                                best_microstate = ref_name
                                inverse = False

                            if spatial_corr_inv < best_score:
                                best_score = spatial_corr_inv
                                best_microstate = ref_name
                                inverse = True

                    best_dict["id"] = f_id
                    best_dict[best_microstate] = change_dict[column_name]
                    best_dict[f"invert_{best_microstate}_polarity"] = inverse
            list_to_write.append(best_dict)

    df = pd.DataFrame(list_to_write)
    df = df.fillna("MISSING")
    df.to_csv(f"{outdir}/connecting_microstates.csv", index=False)


def calc_average_microstates(maps_dir=None, connecting_df=None, outfile=None):
    """
    Calculates the average maps of a directory of maps, via the connecting dataframe.
    Connecting dataframe is a must before using this function!
    :param maps_dir: Directory of the maps.
    :param connecting_df: The dataframe where the connections are recorded
    :param outfile: Path for the outfile.
    :return: None
    """

    microstates_df = connecting_df
    microstate_columns = [c for c in microstates_df.columns.tolist() if not c.startswith("invert_") and c != "id" and c != "Note"]
    microstate_counters = [0] * len(microstate_columns)
    microstate_avgs = [None] * len(microstate_columns)
    for idx, cn in enumerate(microstate_columns):

        m_idx = idx

        for _, row in microstates_df.iterrows():
            try:
                maps_df = pd.read_csv(f"{maps_dir}/{row['id']}_maps.csv")
            except Exception as e:
                print(e)
                continue
            maps_df.rename(columns={"1": "A", "2": "B", "3": "C", "4":"D"}, inplace=True)
            if row[cn] != "MISSING":
                print(row[cn])
                map_microstate = maps_df[row[cn]]
                polarity_reverse = True if row[f"invert_{cn}"] == "True" else False
                if polarity_reverse:
                    map_microstate = -1 * map_microstate
                if microstate_avgs[m_idx] is None:
                    microstate_avgs[m_idx] = map_microstate
                else:
                    microstate_avgs[m_idx] += map_microstate
                microstate_counters[m_idx] += 1

    micro_avg_df = pd.DataFrame()
    for idx, mavg in enumerate(microstate_avgs):
        try:
            avg_microstate = mavg / microstate_counters[idx]
            micro_avg_df[microstate_columns[idx]] = avg_microstate
        except:
            avg_microstate = 0
            micro_avg_df[microstate_columns[idx]] = avg_microstate
    micro_avg_df.to_csv(outfile)




def plot_avg_microstates(example_bdf_or_edf_file_path, microstate_df_to_plot=None, good_channels=None, bad_channels=None, stim_channel=None,
                         montage_string="biosemi32", outdir=None):

    avg_microstates_df = pd.read_csv(microstate_df_to_plot)

    #just for a sample - to consturct a bdf with the appropriate .info  data.
    try:
        if example_bdf_or_edf_file_path.endswith(".edf"):
            raw = mne.io.read_raw_edf(example_bdf_or_edf_file_path, exclude=bad_channels, eog=good_channels, stim_channel='Status',
                                      preload=True)
        elif example_bdf_or_edf_file_path.endswith(".bdf"):
            raw = mne.io.read_raw_bdf(example_bdf_or_edf_file_path, exclude=bad_channels, eog=good_channels, stim_channel='Status',
                                      preload=True)
        else:
            print("File must be an .edf or a .bdf file!!")
            sys.exit()
    except Exception as e:
        print(e)
        return

    epoch_length = 512
    overlap = 0


    biosemi_montage = mne.channels.make_standard_montage(montage_string)
    channel_types = {ch_name: 'eeg' for ch_name in raw.ch_names}
    print(raw.ch_names)
    if stim_channel:
        channel_types[stim_channel] = 'stim'
    raw.set_channel_types(channel_types)
    raw.set_montage(biosemi_montage, on_missing='ignore')

    raw.pick_types(meg=False, eeg=True, stim=True)
    # raw.notch_filter(freqs=[50, 100], n_jobs=1)
    raw.notch_filter(freqs=[50], n_jobs=1)
    raw.filter(l_freq=0.5, h_freq=40, n_jobs=1)

    raw.set_eeg_reference('average')
    epochs_data = mne.make_fixed_length_epochs(raw, duration=epoch_length / raw.info['sfreq'],
                                               overlap=overlap).get_data()
    events = mne.make_fixed_length_events(raw, duration=epoch_length / raw.info['sfreq'])
    epochs = mne.EpochsArray(epochs_data, raw.info, events=events)
    if stim_channel:
        raw.add_events(events)
        raw.set_annotations(epochs.annotations)

    Path(f"{outdir}").mkdir(parents=True, exist_ok=True)
    for c in avg_microstates_df.columns[1:]:
        mne.viz.plot_topomap(avg_microstates_df[c], raw.info, show=False, ch_type="eeg", res=512, size=3)
        plt.savefig(f"{outdir}/average_{c}.png")

def make_microstates(raw, fname, good_channels=None, reference_df=None,
                      random_state=42, num_comp=10, outdir=None, new_signals_to_csv=False,):
    """
    This function is used for preprocessing .edf/.bdf files, calculating maps for that file, and backfitting them into a reference.
    If there is a reference map, this function also backfittes the calculated maps into that reference.
    This function creates a 'bad' folder, and a subfolders in the 'bad' folder to each individual, with 3 .csv files in it.
    If those 3 .csv files exists, it does the preprocessing automatically via those files, if not, it needs to be done by hand,
    If one wants to redo the preprocessing for an individual by hand, the 3 .csv file needs to be deleted.
    If for example one only wants to redo the ICA part of the preprocessing, it is enough if only the ica_components.csv is deleted.
    :param raw: bdf/edf file
    :fname fnmae: filename, without extension
    :param good_channels: List of name of the good channels that will be used in the file.
    :param reference_df: Dataframe of the reference. The program backfits the maps into this reference.
    :param random_state: Random state to use. Default is 42.
    :param num_comp: Number of components to use in the Indepentent Component Analysis. Can't be more than the number of electrodes used.
    Default is 10.
    :param outdir: Path of the directory where the program will write the files.
    :param new_signals_to_csv: If set to True, the program also writes the new signals out into a .csv file.
    :return: None
    """

    counter=0
    if outdir is None:
        while os.path.exists(f"preprocessed_signals/{counter}"):
            counter += 1
        outdir=f"preprocessed_signals/{counter}"

    #### PREPROCESSING END #####
    Path(f"{outdir}/bads/{fname}").mkdir(parents=True, exist_ok=True)
    #if there's no bad_intervals.csv in the bads/{name} folder, then have mark them by hand
    if not os.path.isfile(f"{outdir}/bads/{fname}/{fname}_bad_intervals.csv") or not os.path.isfile(f"{outdir}/bads/{fname}/{fname}_bad_channels.csv"):
        print(f"There is no {outdir}/bads/{fname}/{fname}_bad_intervals.csv so you need to mark the bad intervals (with 'BAD_' title) and mark the bad channels to interpolate!")
        raw.plot(block=True)
        bad_channels = raw.info['bads']
        bad_intervals = raw.annotations[raw.annotations.description == 'BAD_']
        bad_interval_starts = bad_intervals.onset
        bad_interval_durations = bad_intervals.duration
        bad_interval_descriptions = bad_intervals.description

        # Create a pandas DataFrame with the bad interval information
        bad_intervals_df = pd.DataFrame({'start_time': bad_interval_starts,
                                         'duration': bad_interval_durations,
                                         'description': bad_interval_descriptions})
        bad_channels_df = pd.DataFrame({'bad_channels': bad_channels})


        bad_intervals_df.to_csv(f"{outdir}/bads/{fname}/{fname}_bad_intervals.csv")
        bad_channels_df.to_csv(f"{outdir}/bads/{fname}/{fname}_bad_channels.csv")



    bad_intervals_df = pd.read_csv(f"{outdir}/bads/{fname}/{fname}_bad_intervals.csv")
    bad_channels_df = pd.read_csv(f"{outdir}/bads/{fname}/{fname}_bad_channels.csv")

    bad_intervals_ann = mne.Annotations(onset=bad_intervals_df['start_time'],
                                        duration=bad_intervals_df['duration'],
                                        description=bad_intervals_df['description'])

    raw.set_annotations(bad_intervals_ann)

    raw.info['bads'] = bad_channels_df["bad_channels"].tolist()

    raw.interpolate_bads(reset_bads=True)

    data = raw.get_data()
    new_signals = preprocesser.drop_bad_intervals(data, raw.info['sfreq'],
                                                  bad_intervals_df['start_time'],
                                                  bad_intervals_df['duration'])

    new_raw = mne.io.RawArray(new_signals, info=raw.info)

    #fig = mne.viz.plot_raw(raw, duration=raw.times[-1], block=False, show=False)
    #fig.savefig('ica_plots/raw_plot.png')


    ica = mne.preprocessing.ICA(n_components=num_comp, random_state=random_state)
    ica.fit(new_raw)
    if not os.path.isfile(f"{outdir}/bads/{fname}/{fname}_bad_ica_components.csv"):
        #new_raw.plot(block=False)
        ica.plot_sources(new_raw, start=0, stop=200, picks=list(range(num_comp)), block=True)
        bad_ica_df = pd.DataFrame({"bad_ica_components": ica.exclude, "num_of_components": num_comp, "random_state": random_state})
        bad_ica_df.to_csv(f"{outdir}/bads/{fname}/{fname}_bad_ica_components.csv")


    bad_ica_df = pd.read_csv(f"{outdir}/bads/{fname}/{fname}_bad_ica_components.csv")
    bad_icas = bad_ica_df["bad_ica_components"].tolist()

    ica.exclude = bad_icas
    ica.apply(new_raw)

    if new_signals_to_csv is True:
        dict_to_write = {}
        for z in zip(good_channels, new_raw.get_data()):
            dict_to_write[z[0]] = z[1]
        df_to_write = pd.DataFrame(dict_to_write)
        df_to_write.to_csv(f"{outdir}/new_signals/{fname}.csv")

    new_data = new_raw.get_data()
    gfp = np.std(new_data, axis=0)
    peaks, _ = mne_microstates.find_peaks(gfp)

    new_raw.pick_types(meg=False, eeg=True)
    maps, segmentation = mne_microstates.segment(new_raw.get_data(), n_states=4, random_state=random_state)
    plt.clf()
    plt.cla()
    plt.close()
    Path(f"{outdir}/microstates/plots/{fname}/").mkdir(parents=True, exist_ok=True)
    _plot_maps(maps, new_raw.info, segmentation, f"{outdir}/microstates/plots/{fname}/{fname}")

    maps_dict = {"1": maps[0],
                 "2": maps[1],
                 "3": maps[2],
                 "4": maps[3],}
    maps_df = pd.DataFrame(maps_dict)
    Path(f"{outdir}/microstates/maps/").mkdir(parents=True, exist_ok=True)
    maps_df.to_csv(f"{outdir}/microstates/maps/{fname}_maps.csv")

    if reference_df is not None:
        Path(f"{outdir}/backfitted_microstates/").mkdir(parents=True, exist_ok=True)
        microstates = _categorize_spatial_correlation(new_data, peaks, reference_df)
        microstate_df = pd.DataFrame({"microstates": microstates})
        microstate_df.to_csv(f"{outdir}/backfitted_microstates/microstates_absolute_{fname}.csv")



