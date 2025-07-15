from os import environ as os_environ
os_environ["OMP_NUM_THREADS"] = "1"
from argparse import ArgumentParser
from MsTargetPeaker import PeakQualityReport


def main():
  parser = ArgumentParser()
  parser.add_argument("chromatogram_tsv", type=str, help="The chromatogram TSV file path")
  parser.add_argument("peak_boundary_csv", type=str, help="The Peak Boundary CSV file path")
  parser.add_argument("output_folder", type=str, help="Output folder")
  parser.add_argument('--internal_standard_type', '-s',type=str, default='heavy', choices=['heavy', 'light'] , help="Set the internal standards to heavy or light ions. (default: heavy)") 
  parser.add_argument('--top_n_fragment', '-n', type=int, default=5, help="Automatically select top N transition ion for quality reporting. (default: 5)")
  parser.add_argument('--group_csv', '-g', type=str, default=None, help="The CSV file containing the ReplicateName column indicating the groups/batch of samples")
  parser.add_argument('--output_chromatogram_pdf', '-pdf', action='store_true', help="Set for making chromatogram plots in a chromatogram_plots.pdf file. (default: unset)")
  parser.add_argument('--chromatogram_dpi', '-dpi', type=str, default=200, help="The dpi of chromatogram plots. Only works when --output_chromatogram_pdf is set. (default: 200)")
  parser.add_argument('--chromatogram_nrow', '-nrow', type=int, default=6, help="The number of chromatograms per row in one pdf page. Only works when --output_chromatogram_pdf is set. (default: 6)")
  parser.add_argument('--chromatogram_ncol', '-ncol', type=int, default=6, help="The number of chromatograms per columne in one pdf page. Only works when --output_chromatogram_pdf is set. (default: 6)")
  parser.add_argument('--chromatogram_fig_w', '-figw', type=int, default=None, help="The figure width in inches. Only works when --output_chromatogram_pdf is set. (default: nrow * 7)")
  parser.add_argument('--chromatogram_fig_h', '-figh', type=int, default=None, help="The figure height in inches. Only works when --output_chromatogram_pdf is set. (default: ncol * 5)")
  parser.add_argument('--output_mixed_mol', '-mix', action='store_true', help="If set, chromatogram data for each molecule will be mixed in one pdf page. (default: unset)")
  parser.add_argument('--reorder_by_group', '-r', action='store_true', help="If set, chromatogram data for each molecule will be mixed in one pdf page. (default: unset)")
  args = parser.parse_args()
  print("Input Parameters:")
  for arg in vars(args):
    print(arg, '=',getattr(args, arg))
  print('#########################################################')
  pqr = PeakQualityReport(args.chromatogram_tsv, args.peak_boundary_csv, args.output_folder, internal_standard_type=args.internal_standard_type, top_n_transitions=args.top_n_fragment, group_csv=args.group_csv)
  pqr.run()
  if args.output_chromatogram_pdf:
    pqr.output_chrom_plot(ncol=args.chromatogram_ncol, nrow=args.chromatogram_nrow, dpi=args.chromatogram_dpi, figW=args.chromatogram_fig_w, figH=args.chromatogram_fig_h, mol_separation=not args.output_mixed_mol, reorder=args.reorder_by_group)
  print('Finished')

if __name__ == '__main__':
  main()

