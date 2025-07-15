# MsTargetPeaker: a quality-aware deep reinforcement learning approach for peak identification in targeted proteomics

MsTargetPeaker incorporates a deep reinforcement learning agent and Monte Carlo tree search to locate target peak regions in targeted mass spectrometry.
The agent was trained with proximal policy optimization on a big collection of targeted MS datasets containing around 1.7M peak groups.
During the training, we established a gymnasium environment for the agent to move peak boundaries to locate target signals. 
To define optimal peaks, we designed a reward function incorporating our previously developed TMSQE quality scoring. 
Thus, the agent can learn autonomously to find high-scoring peak regions without using maunally annotated peaks. 
In the end, the training process took about 200M timesteps to reach performance plateau.

The peak search procedure in MsTargetPeaker was performed using Monte Carlo tree search guided by this agent to enhance the generability, especially for unseen datasets. To further enhance the precision on ambiguous peaks, additional search rounds were appended trying to locate peak regions enclosing true target signals. 

After running the peak search, the generated peak csv file can be imported into Skyline for manual re-evaluation or peak integration.
MsTargetPeaker also provides a peak reporter to generate interpretable peak quality reports. 

Currently, MsTargetPeaker supports peptide MRM/PRM data.


## Installation
MsTargetPeaker was built as a Python package. You can use the following command to install the package.

```
pip install MsTargetPeaker
```

After you install mstargetpeaker, you can use `mstarget-peaker` and `mstarget-reporter` as the command line tools for identification of peak regions and assessment of the peak quality.
Use `--help` or `-h` to see detailed argument descriptions.


## Input Data Format

MsTargetPeaker currently accepts chromatogram data in tab-separated value (TSV) format. This chromatogram file can be exported via **Skyline**.

The required nine column headers for chromatograms are listed as follows.

| FileName |  PeptideModifiedSequence  |  PrecursorCharge | ProductMz | FragmentIon | ProductCharge | IsotopeLabelType | Times | Intensities|
|----------|---------------------------|------------------|-----------|-------------|---------------|------------------|-------|------------|



## Usage

Use the following command to run `MsTargetPeaker` to search peak regions in chromatograms.

```Shell
MsTargetPeaker <chromatogram_tsv> <output_peak_boundary_csv>
```

With this command, MsTargetPeaker takes the first chromatogram TSV file as input and outputs the resulting peak regions to the CSV file specified in the second argument.
The resulting peak CSV file can be imported into Skyline to update peak regions in the chromatograms.

The full arguments are shown below:

```Shell
MsTargetPeaker [-h] [--speed SPEED] [--search SEARCH] [--config CONFIG] [--picked PICKED] [--process_num PROCESS_NUM] [--internal_standard_type {heavy,light}] [--device DEVICE] chromatogram_tsv output_peak_boundary_cs
```

| Argument      |   | Description | Value Type |Default Values<tr><td colspan="5">**INPUT**</td></tr>
|:--------|--|:------------|------------|----------:|
|chromatogram_tsv| |The chromatogram TSV file path| File path|no default<tr><td colspan="5">**OUTPUT**</td></tr>
|output_peak_boundary_csv| |The output peak boundary CSV file path|File path|no default<tr><td colspan="5">**Options**</td></tr>
|--help| -h |Show the detailed argument list|(no value)|unset|
|--version| -v |Display the package version|(no value)|unset|
|--speed  | -s |The speed mode of UltraFast (10X), Faster (5X), Fast (2X), or Standard (1X speed). This can be customized in the config file.| string |UltraFast|
|--mode   | -m |The search mode using the parameter set of MRM or PRM. This can be customized in the config file.| string |MRM|
|--prescreen| -pre |Prescreen peak regions for better peak boundaries as initial state.|int|50|
|--internal_standard_type|-r|Set the internal standard reference to heavy or light ions.|{`heavy`, `light`}|heavy<tr><td colspan="5">**GROUPING**</td></tr>
| --process_num | -p | The parallel process number to search peak regions | integer | 4|
| --device | -d | Use cpu or cuda device for peak picking. | string | auto <tr><td colspan="5">**Incremental Peak Search**</td></tr>
| --picked | | The previously picked boundaries for incremental peak search. | File path | unset |
|--start_round|-sr|Specify the starting MCTS round in the config file. This is useful for incremental peak search.| int| 1|
|--end_round|-er|Specify the ending MCTS round in the config file. This is useful for incremental peak search.| int| 7|

## Incremental Peak Search
MsTargetPeaker supports incremental peak search from a previously identified peak boundary csv file (You may use the peak boundary results from Skyline or other peak identification tools).
To further reduce the search time, users can initially use `--speed=SuperFast` to have a quick result. 
Then, specify `--picked={the peak csv file}` with the `--start_round=4` to start the search with parameters of the 4th to the last round of MCTS.
With this setting we can re-search peak groups which rewards failed to pass the threshold set in the config file.

## Configuration

The default configuration file is MsTargetPeaker.cfg. You may customize this file to suit your preferences.

## Quality Reporter

The reporter can be run independently to generate the following five reports:

1. Transition quality files in a folder.
2. An Excel file containing two sheets: sample quality and replicate group quality.
3. A PDF showing chromatogram plots.
4. A PDF swhoing the probability density functions of peak start and end for each target.

To run the quality reporter, use the following command:

```
MsTargetReporter [-h] [--internal_standard_type {heavy,light}] [--top_n_fragment TOP_N_FRAGMENT] [--group_csv GROUP_CSV] [--output_chromatogram_pdf] [--chromatogram_dpi CHROMATOGRAM_DPI]
                 [--chromatogram_nrow CHROMATOGRAM_NROW] [--chromatogram_ncol CHROMATOGRAM_NCOL] [--chromatogram_fig_w CHROMATOGRAM_FIG_W] [--chromatogram_fig_h CHROMATOGRAM_FIG_H] [--output_mixed_mol] [--reorder_by_group]
                 chromatogram_tsv peak_boundary_csv output_folder

```

The full arguments are shown below:
| Argument      |   | Description | Value Type |Default Values<tr><td colspan="5">**INPUT**</td></tr>
|:--------|--|:------------|------------|----------:|
|chromatogram_tsv| |The chromatogram TSV file path| File path|no default|
|peak_boundary_csv| |The output peak boundary CSV file path|File path|no default<tr><td colspan="5">**OUTPUT**</td></tr>
|output_folder| |The output peak boundary CSV file path|File path|no default<tr><td colspan="5">**Options**</td></tr>
|--help| -h |Show the detailed argument list|(no value)|unset|
|--group_csv|-g|The CSV file containing the replicate group information|File path|unset|
|--top_n_fragment|-n|Automatically select top N transition ions for reporting the quality|integer|5<tr><td colspan="5">**Options for Generating Chromatogram Plots**</td></tr>
|--output_chromatogram_pdf|-pdf|Set for generating chromatogram plots in a file named chromatogram_plots.pdf|File path|unset|
|--output_mixed_mol|-mix|If set, chromatogram plots for each target molecule will be mixed in one PDF page.|(no value)|unset|
|--reorder_by_group|-r|If set, target molecule will be reordered based on the replicate group. Only works if the --group_csv is provided.|(no value)|unset|
|--chromatogram_dpi|-dpi|The dpi of chromatogram plots. Only works when --output_chromatogram_pdf is set.|integer|200|
|--chromatogram_nrow|-nrow| | |
|--chromatogram_ncol|-ncol| | |
|--chromatogram_fig_w|-figw| | |
|--chromatogram_fig_h|-figh| | |

## Utility Functions

### Chromatogram Checking
We noticed that certain exported chromatogram TSV files from Skyline may have unpaired arrays of `Time` and `Intensity` between light and heavy ions.
Also, as we currently rely on the modified peptide sequence and sample file name to recognize each peak group, 
it may cause issues if the chromatogram data contain duplicate peptide-sample names.

We provided `MsTargetChromChecker` to solve these two issues.
For the unaligned data points in light and heavy ions, we apply interpolations to make the same number of retention time and its intensity for both light and heavy ions.
For duplicated names for peak groups, MsTargetChromChecker appends a suffix to the sample file names. The suffix has a pattern of `::n`, where n is a number indicating the duplication number.

Use the following command to run `MsTargetChromChecker`,

```
MsTargetChromChecker [-h] chromatogram_tsv output_chrom_tsv
```

### Parallelism
As it takes time to run `MsTargetPeaker`, we provide `MsTargetChromSplitter` to split the task into smaller ones.
Each splitted task can be run parallelly on different processes or machines. The results from these tasks can then be merged with `MsTargetPeakMerger`.

`MsTargetChromSplitter`

```
MsTargetChromSplitter [-h] -n [number of file] chromatogram_tsv output_folder
```
The default spliting number is the number of target moleculars (without specifying the `-n` argument).


`MsTargetPeakMerger`

```
MsTargetPeakMerger [-h] input_folder output_csv_file
```
This `MsTargetPeakMerger` accepts a folder containing multiple peak csv files and output the merged version of those csv files.
You can use `MsTargetChromSplitter` to split the input chromatogram TSV file, run MsTargetPeaker in parallel on each split file to search for peak regions, and merge the resulting peak CSV files in a folder using `MsTargetPeakMerger`.

