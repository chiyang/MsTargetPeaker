from os import environ as os_environ
os_environ["OMP_NUM_THREADS"] = "1"
from argparse import ArgumentParser
import pandas as pd
import numpy as np

def check_chrom_tsv(chrom_tsv, output_tsv):
  chrom = pd.read_csv(chrom_tsv, sep='\t')
  chrom.columns = chrom.columns.str.replace(' ', '')
  for mol, mol_data in chrom.groupby('PeptideModifiedSequence'):
    for fn , fn_data in mol_data.groupby('FileName'):
      counter = 0
      ref_times = []
      ref_time_string = ''
      for time_string, data in fn_data.sort_values(['Times', 'IsotopeLabelType', 'PrecursorCharge', 'FragmentIon', 'ProductCharge']).groupby(['Times']):
        counter += 1
        isFragmentValid = True
        if len(ref_times) == 0:
          ref_times = np.array(str(time_string[0]).split(','), dtype=np.float32)
          ref_time_string = time_string[0]
        for _, frag_data in data.groupby(['PrecursorCharge', 'FragmentIon', 'ProductCharge']):
          if len(frag_data) > 2:
            isFragmentValid = False
            break
        if isFragmentValid and counter > 1:
          print(f'Identified {mol} {fn} has multiple chromatograms (count: {counter}) with correct fragments. Trying to use interpolation to align the datasets')
          time_set2 = np.array(str(time_string[0]).split(','), dtype=np.float32)
          chrom.loc[data.index, 'Times'] = ref_time_string
          chrom.loc[data.index, 'Intensities'] = data['Intensities'].str.split(',').apply(lambda x: np.round(np.interp(ref_times, time_set2, np.array(x, dtype=np.float32)), 6)).apply(lambda x: ','.join(map(lambda y: str(y), x)))
        elif not isFragmentValid:
          print(f'Identified {mol} {fn} has multiple chromatograms sharing with the same RT values (count: {counter})')
          added_counter = 0
          for _, frag_data in data.groupby(['PrecursorCharge', 'FragmentIon', 'ProductCharge', 'IsotopeLabelType']):
            pair_counter = counter
            added_counter = len(frag_data)
            for idx, row in frag_data.iterrows():
              chrom.loc[idx, 'FileName'] = chrom.loc[idx, 'FileName'] + f"::{pair_counter}"
              print(row['FileName'] + f'::{pair_counter}', f"{row['PrecursorCharge']}.{row['FragmentIon']}.{row['ProductCharge']} {row['IsotopeLabelType']}")
              pair_counter += 1
          counter = counter + added_counter -1
          
  chrom.to_csv(output_tsv, sep="\t", index=False)
  print(f'Output checked chromatogram tsv file to {output_tsv}')

def main():
  parser = ArgumentParser()
  parser.add_argument("chromatogram_tsv", type=str, help="The chromatogram TSV file path")
  parser.add_argument("output_chrom_tsv", type=str, help="Output chromatogram tsv file with unique filenames for each target sample")
  args = parser.parse_args()
  print("Input Parameters:")
  for arg in vars(args):
    print(arg, '=',getattr(args, arg))
  print('#########################################################')
  check_chrom_tsv(args.chromatogram_tsv, args.output_chrom_tsv)
if __name__ == '__main__':
  main()