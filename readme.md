# REFERENCE ARTICLE
https://www.psychologie.uni-freiburg.de/abteilungen/psychobio/studium/archiv/ws1920/archiv/ws1819/material/Michel17-MicrostatesReview.pdf

# This program's features are:
- uses mne_python to help preprocess .edf/.bdf files based on this article. (the bad intervals/channels and ICA components has to be marked by hand).
- backfits the preprocessed files to reference maps to create a microstate sequence based on this article.
- extracts further features from the created microstate sequences.
- using a correlation matrix of a feature set, it searches for cliques in that correlation matrix, and extracts them for future machine learning. 


### Python version:
- python 3.9.

# Preprocessing:
- **Setting a montage if there is none! This is mandatory so that the electrodes have positions!**
- Resampling the signals to preprocess them faster.
- Notch filter. Default at 50Hz.
- Bnadpass filter. Default is between 0,5Hz and 40Hz.
- Identifying the bad intervals and interpolate the bad channels by hand.
- Removing the bad states using Independent Component Analysis if necessary. 
- Calculating the 4 most present maps at GFP peaks only.
  - If there's reference maps already the function will backfit the signals to that reference.
  - If there is not, then based on the previously calculated maps, it needs to be determined <br> 
  for each individual, which calculated map (1,2,3,4) corresponds to which reference microstate map (A,B,C,D,... etc.). <br>
  After the microstates have been connected in a .csv file, the average of these microstates has to be calculated <br> in order to be used
  as reference maps, and using these reference maps, running the function again, the function will backfit the individual microstates to these maps.


- ![ScreenShot](drawio_etc\creating_backfitted_microstates.png)



### running the make_microstates function

The necessary arguments are the following:
- raw: bdf/edf file waiting to be preprocessed
- fname: filename, without extension. Can be anything.
- good_channels: list of channels that will be used
- reference_df: dataframe of the aforementioned reference maps. If exists, the program backfits the signals to this.
- outdir: the directory where the program will write.

The program first makes a /bads/ folder. As the program goes, in this folder there will be <br>
a subfolder for each individual. <br> <br >In these subfolders there will be 3 .csv files:
  - {name}_bad_channels.csv: list of channel names that were marked to interpolate
  - {name}_bad_intervals.csv: list of intervals that were marked as bad.
  - {name}_bad_ica_components.csv: list of ica components that was marked to be excluded.

If you wish to redo these preprocessing steps for an individual, just simply delete the corresponding .csv file, and run the function again.

The program then will calculate the 4 most present microstate at GFP peaks, makes the plots of the maps, then writes them into a .csv file.

If there's a reference_df the program will also backfit the signals to that reference.
<br>The backfitted signals will be in the {outdir}/backfitted_microstates directory.

If the argument new_signals_to_csv is set to True it will write the new preprocessed signals into a .csv file.

### running the calc_average_microstates function

Before running this function the 4 microstate of each individual needs to be connected uniformly to each other in a .csv file, in the following way for each individual:

| id | microstate_A | invert_A_polarity | microstate_B | invert_microstate_B_polarity | microstate_C | invert_microstate_C_polarity | microstate_D | invert_microstate_D_polarity | microstate_D2 | invert_microstate_D2_polarity | Note                     |
|----|--------------|-------------------|--------------|------------------------------|--------------|------------------------------|--------------|------------------------------|---------------|-------------------------------|--------------------------|
| h1 | D            | True              | MISSING      | MISSING                      | C            | False                        | B            | True                         | A             | False                         | whatever string you want |


***this csv can only contain the following items:***
- id column
- Note column
- All the other columns will be treated as a `microstate`, and has to have an `invert_{microstate_name}_polarity` column with a boolean value.

--->  `microstate_columns = [c for c in microstates_df.columns.tolist() if not c.startswith("invert_") and c != "id" and c != "Note"]`

What this does is, it signals that out of the 4 calculated maps for '**h1**', <br>
the D belongs to the ***reference***'s A map, but with reversed polarity, and so on...<br>
Since the program calculates only 4 map, but the reference has 5 maps, <br>
it means that there will be at least 1 map that wasn't 'found' amongst the calculated map.<br>
At cases like this, it needs to be marked with the 'MISSING' marking.<br>

***In this case this reference is just something to group the 4 microstates, not actual reference maps.<br>***
***That will be calculated later based on this grouping.***

This step needs to be done in order to calculate a reference map that will be used as backfitting.

The necessary arguments are the following:
- maps_dir: this directory contains the signals of the maps that the make_microstates function generated
- connecting_df: Dataframe that contains the aforementioned "connecting" .csv file.
- outfile: path of the file that will be generated
The program will generate the average microstate maps into a .csv file.

# Calculating microstate characteristics

  There are 2 functions in 2 .py files that are responsible for calculating characteristics.
  The first one is calculate_characteristics.py's read_file_characteristics function, the second one is transitions_preprocessing.py's read_file_transitions function.
  They need to be run in this order!
  
### read_file_characteristics function

The necessary arguments are:
- dirpath: the directory with the previously made backfitted signals.
- list_of_categories: a list that contains each individuals' category. For example:
  - let's say there is 5 file in the backfitted folder, and two of them belongs to group 'A', and three of them belongs to group 'B', in this order: [A, B, A, B, B]. 
  Then the list_of_categories' marking order should be the same. (of course you can mark them however you want, or use more than 2 groups as well)

It creates 2 .csv files:
- eeg_sequence_statistical.csv
- eeg_sequence_transitions.csv


### read_file_transitions function

The only necessary argument is:
- sequence_transitions_csv: The filepath of the sequence_transitions_csv made by the read_file_characteristics function.

This one creates the following 3 .csv files that can be used for further work:
- eeg_transition_homogeneity_metric.csv
- eeg_transition_metrics.csv
- eeg_transition_shannon_metric.csv

# Making cliques for machine learning

In this part:
- first the program creates a correlation matrix of a feature dataframe. (for example the characteristics from before)
- it collects all the edges in that matrix below or above of a certain limit
- and it searches all the cliques between 2 limits in that graph.

2 Functions are used in this part.

### make_graph function

Calculates the correlation matrix (lower triangular part) below <limits> and the graph for the make_cliques function.<br>
The necessary arguments are:
- df_path: this is a dataframe with the features to work with
- date_str: this is just a string to mark the output. Can be any string.
- limit: The limit that the function uses for getting the edges in the correlation matrix.
- control_class: Label of the control class. Default is 1.
- ccompare_class: Which class' row to keep besides the control class. Default is 2.
- below limit: Collect the edges either below or above the limit.
- columns_to_drop: these columns will be excluded when calculating the correlation matrix. (for example: ["id", "group"])

### make_cliques function

This function uses the output graph from the aforementioned make_corr function.<br>
It searches for cliques in that graph (between a minimum and maximum length), that can be used for machine learning.
The necessary arguments are:
- graph_df: Dataframe containing a graph.
- date_str: this is just a string to mark the output. Can be any string.
- min_clique: minimum length of cliques to search for.
- max clique: maximum length of cliques to search for.
- disjunct_cliques: whether the program should **NOT** write sub-cliques of bigger cliques.
  For example: there are 2 cliques: (A,B,C,D) and (A,C,D). <br>
  If this is set to True, it doesn't collect the (A,C,D) clique because it is a part of the (A,B,C,D) clique.
- outdir: path of the directory the program should write the output file

# OTHER REFERENCES:
Part of this program's feature extraction uses these:
-  [EEG microstate analyses from Frederic-vW](https://github.com/Frederic-vW/eeg_microstates). The functions were copied and fixed so that it'd work with python 3.9.<br>
-  [Discrete Information Theory](https://dit.readthedocs.io/en/latest/generalinfo.html) python package.
-  [mne_microstates by wmvanvliet](https://github.com/wmvanvliet/mne_microstates)


# Test files:
The .edf test files can be found here:  [RepOD>IBIB PAN - Department of Methods of Brain Imaging and Functional Research of Nervous System](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441)