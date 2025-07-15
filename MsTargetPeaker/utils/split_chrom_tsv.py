from pandas import read_csv as pd_read_csv
from argparse import ArgumentParser
from numpy import array_split as np_array_split
from os import makedirs as os_makedirs
from os.path import join as os_path_join, basename as os_path_basename, splitext as os_path_splitext

def split_chrom_tsv(chrom_tsv, output_folder, split_num=0):
  chrom = pd_read_csv(chrom_tsv, sep='\t')
  chrom.columns = chrom.columns.str.replace(' ', '')
  pep_list = chrom['PeptideModifiedSequence'].unique()
  by_column = 'PeptideModifiedSequence'
  os_makedirs(output_folder, exist_ok=True)
  prefix = os_path_join(output_folder,os_path_basename(os_path_splitext(chrom_tsv)[0]))
  if split_num == 0:
    splited_list = np_array_split(chrom[by_column].unique(), len(pep_list))
  else:
    splited_list = np_array_split(chrom[by_column].unique(), split_num)
  for i in range(len(splited_list)):
    data = chrom[chrom[by_column].isin(splited_list[i])]
    output_path = prefix + f'_split_{i}.tsv'
    row_num = len(data)
    print(f'Output to {output_path} containing {row_num} rows.')
    data.to_csv(output_path, sep='\t', index=False)

def main():
  parser = ArgumentParser()
  parser.add_argument("chrom_tsv", type=str, help="The chromatogram tsv file path")
  parser.add_argument("output_folder", type=str, help="The output folder of splitted chromatogram files")
  parser.add_argument("--number", '-n', type=int, default=0, help="The desired slice number of the chromatogram tsv files")
  args = parser.parse_args()
  split_chrom_tsv(args.chrom_tsv, args.output_folder, split_num=args.number)

if __name__ == '__main__':
  main()