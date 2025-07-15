from argparse import ArgumentParser
import configparser
import json
from time import time
from os import environ as os_environ
from numpy import split as np_split, arange as np_arange, array as np_array, median as np_median, asarray as np_asarray, inf as np_inf
from pandas import DataFrame
from torch import multiprocessing as th_mp
from torch.cuda import is_available as th_cuda_is_available
from torch.backends import mps as torch_mps
from os.path import join as os_path_join, dirname as os_path_dirname
from MsTargetPeaker import MsTargetPeakerEnv
import MsTargetPeaker
os_environ["OMP_NUM_THREADS"] = "1"

def main():
  th_mp.freeze_support()
  parser = ArgumentParser()
  parser.add_argument('--version', '-v', help='show the version of the tmasque package', action='version', version=MsTargetPeaker.__version__)
  parser.add_argument("chromatogram_tsv", type=str, help="The chromatogram tsv file path")
  parser.add_argument("output_peak_boundary_csv", type=str, help="The output peak_boundary csv file path")
  parser.add_argument("--speed", '-s', type=str, default="SuperFast", help="The speed mode of SuperFast (10X), SuperFast (10X), Faster (5X), Fast (2X), or Standard (1X speed). (default: SuperFast).")
  parser.add_argument("--mode", '-m', type=str, default="MRM", help="The search mode defined in the config file. Default search modes can be MRM or PRM. (default: MRM)")
  parser.add_argument("--config", '-c', type=str, default=None, help="The file path for customized config file. If unset, use the default config. (default: None)")
  parser.add_argument('--picked', type=str, default=None, help="The previously picked boundaries to continue picking. (default: None)")
  parser.add_argument('--process_num', '-p', type=int, default=4, help="The parallel processing number to calculate quality feature values for all peak groups (default: 4)")
  parser.add_argument('--prescreen', '-pre', type=int, default=50, help="Prescreen peaks for better peak boundaries as initial state. (default: 50)")
  parser.add_argument('--internal_standard_type', '-r', type=str, default=None, choices=['heavy', 'light'], help="Set the internal standard reference to heavy or light ions. (default: heavy)") 
  parser.add_argument('--start_round', '-sr', type=int, default=1, help="Specify the start MCTS round set in the config file. This can be helpful when using continuous peak search. (default: 1)") 
  parser.add_argument('--end_round', '-er', type=int, default=7, help="Specify the end MCTS round set in the config file. This can be helpful when using continuous peak search. (default: 7)") 
  parser.add_argument('--device', '-d', type=str, default='cpu', help="Use cpu or cuda device for model peak picking. (default: cpu)")
  args = parser.parse_args()
  print('Input Parameters')
  for arg in vars(args):
    print(arg, getattr(args, arg))
  print('#########################################################')
  start = time()
  config = configparser.ConfigParser()
  if args.config:
    config.read(args.config)
  else:
    config.read(os_path_join(os_path_dirname(__file__), 'MsTargetPeaker.cfg'))
  default_conf = config['DEFAULT']
  speed = default_conf['Speed']
  if 'speed' in args or args.speed:
    speed = args.speed
  search_mode = default_conf['SearchMode']
  if 'search' in args or args.mode:
    search_mode = args.mode
  device = default_conf['Device']
  policy_path = None if default_conf['PolicyPath'] == 'None' else default_conf['PolicyPath']
  process_num = float(default_conf['ParallelProcessNumber'])
  internal_standard_type = default_conf['InternalStandardType']
  max_step = int(default_conf['MaxTimeStep'])
  mcts_param_conf = config['MCTSParam']
  mcts_param = {
    "alpha":  float(mcts_param_conf['Alpha']),
    "beta":   float(mcts_param_conf['Beta']),
    "K":    eval(mcts_param_conf['K']),
    "eval_mode": mcts_param_conf['EvalMode']
  }
  MCTS_cycles = json.loads(config[f'Speed.{speed}']['MCTSCycle'])
  
  searchParams = config[f'SearchParam.{search_mode}']
  use_weighted_PBAR = int(searchParams['UseWeightedPBAR'])
  threshold_list = json.loads(searchParams['Threshold'])
  selection_noise_list = json.loads(searchParams['SelectionNoise'])
  pbar_heavy_weight = json.loads(searchParams['PBARHeavyWeight'])
  pair_ratio_factor_list = json.loads(searchParams['PairRatioFactor'])
  top_n_ion_list = json.loads(searchParams['TopNIon'])
  intensity_power_list = json.loads(searchParams['IntensityPower'])
  quality_power_list = json.loads(searchParams['QualityPower'])
  use_consensus_list = json.loads(searchParams['UseConsensus'])
  consensus_threshold_list = json.loads(searchParams['ConsensusThresholds'])
  use_ref_if_no_consensus_list = json.loads(searchParams['UseRefIfNoConsensus'])
  overwrite_list = json.loads(searchParams['Overwrite'])
  
  output_peak_boundary_csv = args.output_peak_boundary_csv
  th_mp.set_start_method('spawn')
  start_idx = args.start_round - 1
  end_idx = args.end_round - 1
  if 'device' in args or args.device:
    device = args.device
  if device == 'auto':
    device = 'cuda' if th_cuda_is_available() else 'cpu'

  if 'process_num' in args or args.process_num:
    process_num = args.process_num
  if 'internal_standard_type' in args or args.internal_standard_type:
    internal_standard_type = args.internal_standard_type

  current_csv_file = args.picked
  canStop = False
  total_steps = len(MCTS_cycles)
  if total_steps > len(threshold_list):
    total_steps = len(threshold_list)
  for step in range(total_steps):
    if step < start_idx:
      continue
    if step > end_idx:
      break
    print(f"[Round {step+1}] Start.")
    env = MsTargetPeakerEnv(args.chromatogram_tsv, policy_path=policy_path, picked_peak_csv=current_csv_file,
                            max_step=max_step, device=device, 
                            internal_standard_type=internal_standard_type,
                            pair_ratio_factor= pair_ratio_factor_list[step], 
                            pbar_heavy_weight = pbar_heavy_weight[step],
                            intensity_power=intensity_power_list[step],
                            quality_power=quality_power_list[step],
                            top_n_ion=top_n_ion_list[step],
                            use_weighted_pbar = True if use_weighted_PBAR == 1 else False,
                            use_kde=current_csv_file if use_consensus_list[step] == 1 else None,
                            threshold_for_kde=consensus_threshold_list[step],
                            use_ref_if_no_kde= True if use_ref_if_no_consensus_list[step] == 1 else False
                          )
    all_samples, samples_to_process, need_repicking, pre_picked = env.get_sample_lists(threshold_list[step])
    sample_batches = np_split(samples_to_process, np_arange(process_num, len(samples_to_process), process_num))
    prescreen = args.prescreen
    sample_batches = [(batch, prescreen) for batch in sample_batches]
    print(f'Reading peak groups with {prescreen} pre-screens ...')
    with th_mp.Pool(args.process_num) as pool:
      chrom_list = pool.starmap(env.get_chrom_list_from_sample, sample_batches)
      pool.close()
      pool.join()
    chrom_list = [ y for x in list(filter(lambda x: x is not None, chrom_list)) for y in x]
    chrom_list = np_asarray(chrom_list, dtype='object')
    print(f'[Round {step+1}] Number of peak groups (reward < {threshold_list[step]})  to be processed: {len(chrom_list)}/{len(all_samples)} ({round(100*len(chrom_list)/len(all_samples), 2) if len(all_samples) > 0 else 0.00}%)')
    if overwrite_list[step]:
      print(f'[Round {step+1}] Ignore pre-selected peaks.')
    print(f"RUN MCTS-DPW with {mcts_param['eval_mode']} evaluation")
    option = {'cycle': MCTS_cycles[step], 'alpha': mcts_param['alpha'], 'beta': mcts_param['beta'],
                'K': mcts_param['K'], 'eval_mode': mcts_param['eval_mode'], 'selection_noise': selection_noise_list[step]}
    print(f'[Round {step+1} Option] {option}')
    chrom_len = len(chrom_list)
    arg_list = [(chrom, option, f"[Round {step+1} ({idx+1}/{chrom_len})]") for idx, chrom in enumerate(chrom_list)]
    results = []
    total_size = len(arg_list)
    poolsize = total_size if process_num > total_size else process_num
    print(f'Pool size: {poolsize}')
    if poolsize > 0:
      pool = th_mp.Pool(poolsize)
      results = pool.starmap(env.run_mcts, arg_list)
      pool.close()
      pool.join()
      print('Finished MCTS')
    else:
      canStop = True
    #results = list(chain.from_iterable(results))
    result_df = DataFrame(results)
    if not result_df.empty:
      result_df.columns = ['File Name', 'Peptide Modified Sequence', 'Min Start Time', 'Max End Time', 'Type1Reward', 'Type2Reward',
                           'FinalReward', 'PBAR', 'PBARFactor','PairRatioConsistencyMedian', 
                           'PairRatioConsistencyFactor', 'PeakModality', 'PeakModalityFactor', 'IntensityQuantile', 
                           'IntensityQuantileFactor', 'PeakStartFactor', 'PeakEndFactor', 'PeakBoundaryFactor', 'Note']
      non_qualified = result_df[result_df['FinalReward'] < threshold_list[step]]
    else:
      non_qualified = []
    print(f"[Round {step+1}] #Peak Groups with Final Reward < {threshold_list[step]}: {len(non_qualified)}")
    print(f'[Round {step+1}] Outputing picked peaks to {output_peak_boundary_csv} ...')
    if need_repicking is None:
      output_df = result_df
    else:
      output_arr = []
      improved_counts = 0
      improved_scores = []
      for file, pep in all_samples:
        if pep in need_repicking and file in need_repicking[pep]:
          previous_result = need_repicking[pep][file]
          if result_df.empty:
            output_arr.append(previous_result)
          else:
            repick_result = result_df[(result_df['Peptide Modified Sequence'] == pep) & (result_df['File Name'] == file)]
            if len(repick_result) >= 1:
              repicked = repick_result.iloc[0]
              previous_reward = previous_result['FinalReward']
              repicked_reward = repicked['FinalReward']
              if repicked_reward > previous_reward:
                improved_counts += 1
                improved_scores.append(repicked['FinalReward'] - previous_result['FinalReward'])
              if overwrite_list[step] or (repicked_reward >= previous_reward):
                output_arr.append(repicked)
              else:
                output_arr.append(previous_result)
            else:
              output_arr.append(previous_result) #Should not exist
        else:
          previous_result = pre_picked[(pre_picked['Peptide Modified Sequence'] == pep) & (pre_picked['File Name'] == file)] if pre_picked is not None else []
          if len(previous_result) >= 1:
            output_arr.append(previous_result.iloc[0])
      output_df = DataFrame(output_arr)
      improved_scores = np_array(improved_scores)
      if len(improved_scores) > 0:
        improved_mean = improved_scores.mean()
        improved_median = np_median(improved_scores)
        improved_min = improved_scores.min()
        improved_max = improved_scores.max()
        print(f'Improved chromatograms: {improved_counts}')
        print(f'Improved scores => mean: {improved_mean}; median: {improved_median}; min: {improved_min}; max: {improved_max}')
      else:
        print('No improvement')
    print('#########################################################')
    print(f'[Round {step+1}] Score distributions:')
    total_len = len(output_df)
    if total_len > 0:
      ranges = [-1, 0, 0.01, 1, 2, 4, 6, 8, 10, 12]
      for idx, range_start in enumerate(ranges):
        if idx + 1 < len(ranges):
          range_end = ranges[idx + 1]
        else:
          range_end = np_inf
        in_range_data = output_df[(output_df['FinalReward'] >= range_start) & (output_df['FinalReward'] < range_end)]
        in_range_len = len(in_range_data)
        print(f'[{range_start}, {range_end}): {in_range_len} ({ round(100*in_range_len/total_len, 2) }%)')
    print('#########################################################')
    print(f'[Round {step+1}] Output peak boundary csv file to ' + output_peak_boundary_csv)
    current_csv_file = output_peak_boundary_csv
    output_df.to_csv(output_peak_boundary_csv, index=False)
    #output_df.to_csv(f'{output_peak_boundary_csv}.step{step+1}.txt', index=False)
    if canStop:
      break
  end = time()
  print('Total execution time: %.2f seconds' % (end - start))

if __name__ == '__main__':
  main()