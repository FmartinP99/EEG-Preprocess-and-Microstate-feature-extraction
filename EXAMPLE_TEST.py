from file_preprocessing import make_microstates, connect_microstates, calc_average_microstates, plot_avg_microstates
from calculate_characteristics import read_file_characteristics
from transitions_processing import read_file_transitions
from correlation_clustering import make_graph, make_cliques
import os
import pandas as pd
import mne
import sys
if __name__ == "__main__":
    
    ############ PREPARING VARIABLES ###################
    files_dirpath = "test_files/test_repod"
    channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    """
    channel_names_biosemi32 = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
                     'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6',
                     'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']
    """
    stim_channel = None

    _bad_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4',
                 'EXG5', 'EXG6', 'EXG7', 'EXG8']

    outdir = "preprocessed_signals/example"
    montage_string = "standard_1020" # "biosemi_32" for the SZTE files.
    reference_df = pd.read_csv(r"calculated_avg_microstates_TEST.csv")

    ########### PREPROCESSING FILES - SETTING MONTAGE IS MANDATORY IF THERE IS NONE!! ####################

    for f in os.listdir(files_dirpath):
        print(f"{f} is in the making!")

        fname = f.split(".")[0]

        try:
            if f.endswith(".edf"):
                raw = mne.io.read_raw_edf(f"{files_dirpath}/{f}", exclude=_bad_channels, eog=channel_names, stim_channel='Status', preload=True)
            elif f.endswith(".bdf"):
                raw = mne.io.read_raw_bdf(f"{files_dirpath}/{f}", exclude=_bad_channels, eog=channel_names, stim_channel='Status', preload=True)
            else:
                print("File must be an .edf or a .bdf file!!")
                continue
        except Exception as e:
            print(e)
            continue

        raw_freq = raw.info['sfreq']
        resample_freq = 256

        raw.resample(int(resample_freq), npad="auto")

        #artifical epoch, doesnt matter
        epoch_length = 512
        overlap = 0
        montage = mne.channels.make_standard_montage(montage_string)
        channel_types = {ch_name: 'eeg' for ch_name in channel_names}
        if stim_channel:
            channel_types[stim_channel] = 'stim'
        raw.set_channel_types(channel_types)
        raw.set_montage(montage, on_missing='ignore')
        raw.pick_types(meg=False, eeg=True, stim=True)

        raw.notch_filter(freqs=50, n_jobs=1)
        raw.filter(l_freq=0.5, h_freq=40, n_jobs=1)

        raw.set_eeg_reference('average')
        epochs_data = mne.make_fixed_length_epochs(raw, duration=epoch_length / raw.info['sfreq'],
                                                   overlap=overlap).get_data()
        events = mne.make_fixed_length_events(raw, duration=epoch_length / raw.info['sfreq'])
        epochs = mne.EpochsArray(epochs_data, raw.info, events=events)
        if stim_channel:
            raw.add_events(events)
            raw.set_annotations(epochs.annotations)


        ############ PREPROCESSING WITHOUT REFERENCE MAPS ############
        make_microstates(raw=raw,fname=fname, good_channels=channel_names, reference_df=None, outdir=outdir)

        # After the average maps were calculated, use the make_microstates function again, with the created average_maps as reference_df.

        ############ PREPROCESSING WITH REFERENCE MAPS ############
        #make_microstates(raw=raw, fname=fname, good_channels=channel_names, reference_df=reference_df, outdir=outdir)

    ############ CALCULATING AVERAGE MAPS THAT CAN BE USED AS REFERENCE############
    
    calc_average_microstates(maps_dir=r"preprocessed_signals/example/microstates/maps",
                             connecting_df=pd.read_csv(
                                 r"preprocessed_signals/example/microstates/connecting_microstates.csv"),
                             outfile=f"preprocessed_signals/example/avg_microstates.csv")
    

    plot_avg_microstates(r"test_files\test_repod\h01.edf",
                         microstate_df_to_plot=r"preprocessed_signals/example/avg_microstates.csv",
                         good_channels=channel_names, bad_channels=_bad_channels,
                         stim_channel=None,
                         montage_string=montage_string, outdir=f"preprocessed_signals/example/")

    ############ CONNECTING MICROSTATES AUTOMATIZED IF THERE IS A REFERENCE FILE ############
    connect_microstates(maps_dirpath=r"preprocessed_signals/example/microstates/maps", reference_df=reference_df,
                        outdir=r"preprocessed_signals/example/microstates")


    ############ MAKING CHARACTERISTICS ############
    read_file_characteristics(r"preprocessed_signals\example\backfitted_microstates", list_of_categories=[1,2])
    read_file_transitions(sequence_tranisitons_csv=r"characteristics/eeg_sequence_transitions.csv")


    ########### MAKING CLIQUES FOR MACHINE LEARNING ############
    columns_to_drop = ["id", "class", "name", "group", "Unnamed: 0", 'timestamp_beginning', 'zcm_threshold', ]
    make_graph(r"characteristics\eeg_transition_metrics_TEST.csv", "20240104_TEST", limit=0.3, compare_class=2, columns_to_drop=columns_to_drop)
    corr_df = pd.read_csv(r"characteristics/graphs/graph_20240104_TEST.csv")
    make_cliques(corr_df, "20240104_TEST", 3, 10, disjunct_cliques=True)

