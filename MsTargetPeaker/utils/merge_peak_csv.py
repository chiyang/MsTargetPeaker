from argparse import ArgumentParser
from glob import glob
from numpy import array as np_array, median as np_median, inf as np_inf
from pandas import read_csv as pd_read_csv, concat as pd_concat
from os import stat as os_stat
from os.path import join as os_path_join


def merge_peak_csv(input_folder, output_csv, picked=None, suffix=None):
  if suffix:
    csv_files = glob(os_path_join(input_folder, f'*{suffix}.csv'))
  else:
    csv_files = glob(os_path_join(input_folder, '*.csv'))
  sorted_csv_files = sorted(csv_files)
  print(f'Merge peak boundary csv files: {sorted_csv_files}')
  dataframes = map(pd_read_csv, filter(lambda x: os_stat(x).st_size > 1, sorted_csv_files))
  output_df = pd_concat(dataframes, ignore_index=True)

  print('#########################################################')
  print('Score distributions:')
  total_len = len(output_df)
  if total_len > 0:
    ranges = [-1, 0, 0.01, 1, 2, 4, 6, 8, 10, 12]
    for idx, range_start in enumerate(ranges):
      if idx + 1 < len(ranges):
        range_end = ranges[idx + 1]
      else:
        range_end = np_inf
      in_range_len = len(output_df[(output_df['FinalReward'] >= range_start) & (output_df['FinalReward'] < range_end)])
      print(f'[{range_start}, {range_end}): {in_range_len} ({ round(100*in_range_len/total_len, 2) }%)')
  print('#########################################################')
  print(f'Output merged csv file to {output_csv}')
  output_df.to_csv(output_csv, index=False)

  if picked is not None:
    pre_picked = pd_read_csv(picked)
    if 'FinalReward' in pre_picked:
      print('#########################################################')
      print('Comparing with previous picked peaks:')
      improved_counts = 0
      improved_scores = []
      for idx, row in pre_picked.iterrows():
        fn = row['File Name']
        ps = row['Peptide Modified Sequence']
        r = row['FinalReward']
        matched_row = output_df[(output_df['File Name'] == fn) & (output_df['Peptide Modified Sequence'] == ps)]
        if not matched_row.empty and len(matched_row) == 1:
          new_r = matched_row['FinalReward'].tolist()[0]
        if new_r > r:
          improved_counts += 1
          improved_scores.append(new_r - r)
      if len(improved_scores) > 0:
        improved_scores = np_array(improved_scores)
        improved_mean = improved_scores.mean()
        improved_median = np_median(improved_scores)
        improved_min = improved_scores.min()
        improved_max = improved_scores.max()
        print(f'Improved chromatograms: {improved_counts}')
        print(f'Improved scores => mean: {improved_mean}; median: {improved_median}; min: {improved_min}; max: {improved_max}')
      else:
        print('No improvement')
      print('#########################################################')

  print('Finished.')

def main():
  parser = ArgumentParser()
  parser.add_argument('input_folder', type=str, help='The input folder containing splitted chromatogram files and picked peak csv fils.')
  parser.add_argument('output_csv', type=str, help='The output csv file of the merged peak data.')
  parser.add_argument('--picked', type=str, default=None, help="The previously picked peak boundaries to compare and summarize the differences. (default: None)")
  parser.add_argument('--suffix', type=str, default=None, help="The filename suffix pattern of csv files. (default: None)")
  args = parser.parse_args()
  merge_peak_csv(args.input_folder, args.output_csv, picked=args.picked, suffix=args.suffix)

if __name__ == '__main__':
  main()