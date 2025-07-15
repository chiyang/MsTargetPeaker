from pandas import read_csv
from numpy import (arange as np_arange, hsplit as np_hsplit, array as np_array, vstack as np_vstack, sqrt as np_sqrt, where as np_where, quantile as np_quantile, float32 as np_float32, mean as np_mean, average as np_average, power as np_power, sum as np_sum, absolute as np_absolute, unique as np_unique, zeros as np_zeros, max as np_max, argmax as np_argmax, log as np_log, count_nonzero as np_count_nonzero, median as np_median, argsort as np_argsort, abs as np_abs, sort as np_sort, flip as np_flip, vectorize as np_vectorize, zeros_like as np_zeros_like,
                  sign as np_sign, diff as np_diff, around as np_around, clip as np_clip, all as np_all, amin as np_amin, amax as np_amax, apply_along_axis as np_apply_along_axis)
from scipy.stats import (percentileofscore as scipy_percentileofscore, gmean as scipy_gmean)
from numpy.random import shuffle as np_random_shuffle
from torch import no_grad as th_no_grad, Tensor as th_Tensor, tensor as th_tensor, as_tensor as th_as_tensor, load as th_load, FloatTensor as th_FloatTensor, float32 as th_float32
from torch.cuda import is_available as th_cuda_is_available
from torch.nn import Module, Sequential, Conv1d, MaxPool1d, ReLU, Flatten, Tanh, Linear
from gymnasium.spaces import Box as gym_spaces_Box
from .tmsqe import PeakQualityEval
from .tmsqe import ChromatogramDB
from .mcts.MCTS_DPW import MCTS_DPW
from scipy.stats import (norm as scipy_norm, gaussian_kde as scipy_gaussian_kde)
from gc import collect as gc_collect
from os.path import join as os_path_join, split as os_path_split

def _shuffle_ions(obs_mx):
  reshaped_array = obs_mx[2:,:].reshape(-1, 2, obs_mx.shape[1])
  np_random_shuffle(reshaped_array)
  shuffled_obs_mx = reshaped_array.reshape(-1, obs_mx.shape[1])
  return np_vstack([obs_mx[0:2,:], shuffled_obs_mx])

def _chrom_to_current_obs(chrom_data):
  (start, end) = chrom_data[3]
  time = chrom_data[10]
  intensity = chrom_data[11]
  transitions = chrom_data[2][0]
  transition_len = len(transitions)
  random_pick_transition_idx = np_arange(transition_len)
  np_random_shuffle(random_pick_transition_idx)
  shuffled_idx = random_pick_transition_idx[0:20]
  transitions = [transitions[transition_idx] for transition_idx in shuffled_idx]
  (light_inst, heavy_inst) = np_hsplit(intensity, 2)
  light_inst = _normalize_intensity(light_inst[:, shuffled_idx])
  heavy_inst = _normalize_intensity(heavy_inst[:, shuffled_idx])
  binary_mask = 1*(1*(time >= start) & (time <= end))
  step_indicator = np_array([1]*len(time))
  light_heavy_pairs = np_vstack(list(zip(light_inst.T, heavy_inst.T)))
  max_channels = 42
  to_be_padded = max_channels - 2 - light_heavy_pairs.shape[0]
  obs_mx = np_vstack([step_indicator, binary_mask, light_heavy_pairs, np_zeros((to_be_padded, 1024))])
  return _shuffle_ions(obs_mx)

def _reset_obs(obs_mx, chrom_data):
  obs_mx[0,:] = 1
  (start, end) = chrom_data[3]
  time = chrom_data[10]
  obs_mx[1,:] = 1*(1*(time >= start) & (time <= end))
  return _shuffle_ions(obs_mx)

def _normalize_intensity(intensity_values):
  intensity_df = np_sqrt(np_where(intensity_values >= 1, intensity_values, 1)) #2nd_env57
  max = intensity_df.max().max()
  if max == 0.0:
    return np_zeros_like(intensity_df)
  else:
    min = np_quantile(intensity_df, 0.3)
    if max - min == 0:
      return np_zeros_like(intensity_df)
    normalized = 1 * ((intensity_df - min)/(max-min))
    return np_where(normalized >= 0, normalized, 0)

def _get_quantile_factor(chrom_data, top_n = 1):
  intensity = chrom_data[9]
  peak_intensity = chrom_data[4]
  if peak_intensity.shape[0] == 0:
    return 0, 0
  heavy_peak_intensity = np_hsplit(peak_intensity, 2)[1]
  heavy_intensity = np_hsplit(intensity, 2)[1]
  heavy_unique_intensity = np_unique(heavy_intensity)
  if top_n == 1:
    heavy_peak_intensity_max = np_max(heavy_peak_intensity)
    heavy_peakheight_quantile = np_count_nonzero(heavy_unique_intensity <= heavy_peak_intensity_max) / heavy_unique_intensity.size
  else:
    heavy_peak_intensity_max = np_max(heavy_peak_intensity, axis=0)
    heavy_peakheight_quantiles = scipy_percentileofscore(heavy_unique_intensity, heavy_peak_intensity_max)/100
    heavy_sorted_quantiles = -np_sort(-heavy_peakheight_quantiles)
    heavy_peakheight_quantile = np_mean(heavy_sorted_quantiles[:top_n])
  peak_height_quantile = heavy_peakheight_quantile
  intensity_quantile_factor = 1 - ((1-peak_height_quantile)**0.4)
  return peak_height_quantile, intensity_quantile_factor

def _get_pbar_factor(chrom_data, pbar_heavy_weight=2/3, use_weighted_pbar=True):
  (start, end) = chrom_data[3]
  peak_intensity = chrom_data[4]
  area = chrom_data[7]
  intensity = chrom_data[9]
  if peak_intensity.shape[0] < 1:
    return 0, 1.2e-08, 0, np_array([0.0, 1.0, 2.0]), np_array([1.0, 1.0, 1.0])
  (peak_intensity_light, peak_intensity_heavy) = np_hsplit(peak_intensity, 2)
  (area_light, area_heavy) = np_hsplit(area, 2)
  peak_intensity_max_all = peak_intensity_heavy.max(axis=0)
  picked_area_heavy = (end - start) * peak_intensity_max_all # heavy intensity
  picked_area_heavy[picked_area_heavy < 0.01] = 0.01
  pbar_heavy = area_heavy/picked_area_heavy
  picked_area_light = (end - start) * peak_intensity_light.max(axis=0) # light intensity
  picked_area_light[picked_area_light < 0.01] = 0.01
  pbar_light = area_light/picked_area_light
  ordered_transition_list = np_flip(np_argsort(peak_intensity_max_all)[-3:])
  peak_intensity_max_all = np_where(peak_intensity_max_all > 0, peak_intensity_max_all, 1e-8)
  intensity_ratios = peak_intensity_max_all/peak_intensity_max_all.max()
  intensity_ratios = intensity_ratios[ordered_transition_list]
  if use_weighted_pbar:
    weights = np_power(intensity_ratios, 2)
    weighted_pbar = (pbar_heavy_weight)*pbar_heavy + (1 - pbar_heavy_weight)*pbar_light
    pbar = np_average(weighted_pbar[ordered_transition_list], weights=weights)
  else:
    weighted_pbar = (pbar_heavy_weight)*pbar_heavy + (1 - pbar_heavy_weight)*pbar_light
    pbar = np_mean(weighted_pbar[ordered_transition_list])
  means = [0.3, 0.4, 0.5]
  sigmas = [0.05, 0.06, 0.07]
  pbar_factor = sum(scipy_norm.pdf(pbar, mu, sigmas[i]) for i, mu in enumerate(means))/10.33728076626613
  max_peak_intensity_ion_index = peak_intensity_max_all.argmax()
  intensity_heavy = np_hsplit(intensity, 2)[1]
  peak_mean = peak_intensity_heavy[:, max_peak_intensity_ion_index].mean()
  standard_intensity = intensity_heavy.mean(axis=0)
  sorted_mean = -np_sort(-standard_intensity)
  top_five_global_mean = sorted_mean[:3].mean()
  isSignal = 1 if peak_mean > 0.5*top_five_global_mean else 0
  return pbar, pbar_factor, isSignal, ordered_transition_list, intensity_ratios

def _get_pair_ratio_consistency_factor(ms_peak_qc, chrom_data, pair_ratio_factor):
  pair_ratio_consistency = ms_peak_qc['qualityEval'].calculateEachPairRatioConsistency(chrom_data[6])
  pair_ratio_consistency_median = np_median(pair_ratio_consistency)
  adjusted_pair_ratio_consistency = pair_ratio_consistency_median * pair_ratio_factor
  pair_ratio_consistency_factor = 0.0 if (adjusted_pair_ratio_consistency) >= 2 else ((2 - adjusted_pair_ratio_consistency)**2)/4
  return pair_ratio_consistency_median, pair_ratio_consistency_factor

def _each_transition_modality(signal_series):
    max_intensity = np_max(signal_series)
    if max_intensity == 0: # No intensity
      return 100.0
    flatness_factor = 0.0 if max_intensity <= 1000 else  0.05
    rise = []
    fall = []
    total = 0.0
    sign = np_sign(signal_series[0])
    rise_fall_list = []
    for _, intensity_diff in enumerate(signal_series):
      if np_sign(intensity_diff) == sign:
        total += intensity_diff
      else:
        if np_abs(total) > (flatness_factor * max_intensity):
          if sign > 0:
            rise.append(total)
            rise_fall_list.append(total)
          else:
            fall.append(total)
            rise_fall_list.append(total)
        total = intensity_diff
        sign = np_sign(intensity_diff)
    if np_sign(intensity_diff) == sign:
      if sign > 0:
        rise.append(total)
        rise_fall_list.append(total)
      else:
        fall.append(total)
        rise_fall_list.append(total)
    if len(rise) == 0 or len(fall) == 0: #no peaks in the region
      return 100.0
    sorted_rise_idx = np_argsort(rise)
    sorted_fall_idx = np_argsort(fall)
    rise_dip = rise[sorted_rise_idx[-2]] if len(rise) >= 2 else 0
    fall_dip = fall[sorted_fall_idx[1]] if len(fall) >= 2 else 0
    main_rise = rise[sorted_rise_idx[-1]]
    main_dip = fall[sorted_fall_idx[0]]
    main_rise_idx = rise_fall_list.index(main_rise)
    main_dip_idx = rise_fall_list.index(main_dip)
    if main_rise == 0 or main_dip == 0: # No main peaks in the region
      return 100.0
    elif main_rise_idx > main_dip_idx: # No main peaks in the region (continuous decreasing or increasing signals
      return 100.0
    max_dip = max(np_abs(fall_dip),  rise_dip)
    if max_dip == 0 and np_sign(signal_series[0]) < 0:  # (The first sign is negative and no dip => the velley signal pattern in the region
      return 100.0
    rise_fall_ratio = np_absolute(np_log(np_absolute(main_rise/main_dip)))
    modality = (max_dip / max_intensity) + (rise_fall_ratio/3)
    return 1.0 if modality > 1.0 else 0.0 if modality < 0.0 else modality

def _get_peak_modality_factor(chrom_data, ordered_transition_list, intensity_ratios, isHeavyFirst):
  peak_intensity = chrom_data[4]
  if peak_intensity.shape[0] < 2:
    return 100.0, 0.01
  (_, peak_intensity_heavy) = np_hsplit(peak_intensity, 2)
  peak_diff_mx = np_diff(peak_intensity_heavy, axis=0)
  modality_heavy = np_apply_along_axis(_each_transition_modality, 0, peak_diff_mx)
  modality_heavy = modality_heavy[ordered_transition_list]
  weights = np_power(intensity_ratios, 2)
  modality = np_average(modality_heavy, weights=weights)
  if modality >= 100: #Invalid peaks
    modality_factor = 0.01
  elif modality > 1:
    modality_factor = 0.1
  else:
    if modality <= 1:
      adjusted_modality = modality * 0.3
      modality_factor = (1 - adjusted_modality**0.4) if adjusted_modality < 1 else 0.0
    else:
      modality_factor = 0.1
    modality_factor = 0.1 if modality_factor < 0.1 else modality_factor
  if isHeavyFirst and modality != 100:
    modality_factor = 0.3 if modality_factor < 0.3 else modality_factor
  return modality, modality_factor

def _get_peak_boundary_consensus_factor(chrom_data, peak_kde):
  mol = chrom_data[0]
  start_kde_factor = -1
  end_kde_factor = -1
  kde_factor = 1
  isHeavyFirst = False
  if peak_kde and mol in peak_kde:
    if peak_kde[mol] == True:
      isHeavyFirst = True
    elif peak_kde[mol] == False:
      pass
    else:
      (peak_start, peak_end) = chrom_data[3]
      start_kde_func = peak_kde[mol]['start']
      end_kde_func = peak_kde[mol]['end']
      start_kde_factor = start_kde_func(peak_start)[0]/peak_kde[mol]['start_max']
      end_kde_factor = end_kde_func(peak_end)[0]/peak_kde[mol]['end_max']
      if start_kde_factor < 1e-5:
        start_kde_factor2 = 0.0
      else:
        start_kde_factor2 = start_kde_factor if start_kde_factor > 0.1 else 0.1
      if end_kde_factor < 1e-5:
        end_kde_factor2 = 0.0
      else:
        end_kde_factor2 = end_kde_factor if end_kde_factor > 0.1 else 0.1
      if start_kde_factor2 == 0 or end_kde_factor2 == 0:
        kde_factor = 0.01
      else:
        kde_factor = (start_kde_factor2 + end_kde_factor2)/2
  return kde_factor, start_kde_factor, end_kde_factor, isHeavyFirst

def _serialize_param(pair_ratio_factor, intensity_power, quality_power, top_n_ion):
  return f'{quality_power}_{top_n_ion}_{pair_ratio_factor}_{intensity_power}'

def _get_quality_reward(ms_peak_qc, chrom_data, isHeavyFirst):
  type_1_reward, type_2_reward = ms_peak_qc['qualityEval'].getOverallScore(chrom_data)
  raw_reward = type_1_reward + type_2_reward
  quality_reward = raw_reward / 17.605
  if quality_reward >= 0 and quality_reward < 0.1:
    quality_reward = 0.1
  elif quality_reward < 0:
    quality_reward = 0.1 if isHeavyFirst else 0.1 * ((1 + quality_reward)**2)
  quality_reward = 0 if quality_reward < 0 else quality_reward
  return quality_reward, type_1_reward, type_2_reward, raw_reward

def _calculate_chrom_score(ms_peak_qc, current_chrom, pair_ratio_factor=0.3, pbar_heavy_weight=2/3, intensity_power=1, quality_power=2.0, peak_kde=None, top_n_ion=1, use_weighted_pbar=True):
  kde_factor, start_kde_factor, end_kde_factor, isHeavyFirst = _get_peak_boundary_consensus_factor(current_chrom, peak_kde)
  if isHeavyFirst:          # Enter Heavy-First mode if no valid peaks were found in the target.
    intensity_power = 10    # We used a stronger factor to encourage selections of peak regions with strong heavy signals.
    pair_ratio_factor = 0   # In this case, we do not need to consider the pair ratio consistency.
  pbar, pbar_factor, isSignal, ordered_transition_list, intensity_ratios = _get_pbar_factor(current_chrom, pbar_heavy_weight=pbar_heavy_weight, use_weighted_pbar=use_weighted_pbar)
  pair_ratio_consistency_median, pair_ratio_consistency_factor = _get_pair_ratio_consistency_factor(ms_peak_qc, current_chrom, pair_ratio_factor)
  peak_modality, modality_factor = _get_peak_modality_factor(current_chrom, ordered_transition_list, intensity_ratios, isHeavyFirst)
  intensity_quantile, intensity_quantile_factor  = _get_quantile_factor(current_chrom, top_n=top_n_ion)
  quality_reward, type_1_reward, type_2_reward, raw_reward = _get_quality_reward(ms_peak_qc, current_chrom, isHeavyFirst)
  reward = (quality_reward**quality_power) * isSignal * pbar_factor * (intensity_quantile_factor**intensity_power) * pair_ratio_consistency_factor * modality_factor * kde_factor
  final_reward = -1.0 if reward <= 0.0 else (reward**1.5)*10
  param = _serialize_param(pair_ratio_factor, intensity_power, quality_power, top_n_ion)
  return dict(raw=(type_1_reward, type_2_reward),
              raw_reward=raw_reward,
              reward_score=quality_reward,
              isSignal=isSignal,
              PBAR=pbar,
              PBAR_factor=pbar_factor,
              pair_ratio_consistency_median=pair_ratio_consistency_median,
              pair_ratio_consistency_factor=pair_ratio_consistency_factor,
              peak_modality=peak_modality,
              modality_factor=modality_factor,
              intensity_quantile=intensity_quantile,
              intensity_quantile_factor=intensity_quantile_factor, 
              peak_start_factor=start_kde_factor,
              peak_end_factor=end_kde_factor,
              peak_location_factor=kde_factor,
              reward=reward,
              final_reward=final_reward,param=param)

def _process_peak_idx(s, e):
  if s > e:
    s, e = e, s
  elif s == e:
    s = s - 1
    e = e + 1
  s = max(0, min(1023, s))
  e = max(0, min(1023, e))
  if s >= e:
    if e < 1023:
      e += 1
    elif s > 0:
      s -= 1
  if s > e:
    s, e = e, s
  return s, e

def _step(current_obs, ms_peak_qc, current_chrom, current_step, action, history=None, max_step=5, pair_ratio_factor=0.5,
          pbar_heavy_weight=2/3, intensity_power=1, quality_power = 2.0,
          peak_kde=None, top_n_ion=1, use_weighted_pbar=True):
  index_offset_factor = 200
  rt_time = current_chrom[10]
  start_index_offset = int(np_around(action[0] * index_offset_factor))
  end_index_offset = int(np_around(action[1] * index_offset_factor))
  current_obs = current_obs.copy()
  try:
    peak_region = np_where(current_obs[1, :] >= 1)[0]
    if len(peak_region) == 0:
      (s, e) = ms_peak_qc['chromDB'].getRandomBoundary(current_chrom)
      current_chrom = ms_peak_qc['chromDB'].updateChromData(current_chrom, start=s, end=e)
      current_obs = _chrom_to_current_obs(current_chrom)
      return _step(current_obs, ms_peak_qc, current_chrom, current_step, action, history=history, max_step=max_step, 
                   pair_ratio_factor=pair_ratio_factor,
                   pbar_heavy_weight=pbar_heavy_weight, intensity_power=intensity_power, quality_power=quality_power,
                   peak_kde=peak_kde, top_n_ion=top_n_ion, use_weighted_pbar=use_weighted_pbar)
    elif len(peak_region) == 1:
      (current_start_index, current_end_index) = peak_region[0], peak_region[0]
    else:
      (current_start_index, current_end_index) = peak_region[[0, -1]]
  except Exception as e:
    print(np_where(current_obs[1, :] >= 1))
    raise e
  new_start_idx = current_start_index + start_index_offset
  new_end_idx = current_end_index + end_index_offset
  done = False
  (new_start_idx, new_end_idx) = _process_peak_idx(new_start_idx, new_end_idx)
  current_obs[0,:] = 1 - ((current_step + 1) * (1/max_step))
  current_obs[1,:] = 0
  current_obs[1,new_start_idx:new_end_idx+1] = 1
  (new_start_time, new_end_time)= rt_time[[new_start_idx, new_end_idx]]
  current_chrom = ms_peak_qc['chromDB'].updateChromData(current_chrom, start=new_start_time, end=new_end_time)
  current_step = current_step + 1
  is_stopping = current_step >= max_step
  score_obj = _calculate_chrom_score(ms_peak_qc, current_chrom, pair_ratio_factor=pair_ratio_factor, 
                                     pbar_heavy_weight=pbar_heavy_weight, 
                                     top_n_ion=top_n_ion,
                                     use_weighted_pbar=use_weighted_pbar,
                                     intensity_power=intensity_power, quality_power=quality_power, peak_kde=peak_kde)
  if history is not None:
    history.append(dict(
      sampleName = current_chrom[1],
      peptide=current_chrom[0],
      peak_start = current_chrom[3][0],
      peak_end = current_chrom[3][1],
      type_1_reward = score_obj['raw'][0],
      type_2_reward = score_obj['raw'][1],
      final_reward = score_obj['final_reward'],
      PBAR = score_obj['PBAR'],
      PBAR_factor = score_obj['PBAR_factor'],
      pair_ratio_consistency_median = score_obj['pair_ratio_consistency_median'],
      pair_ratio_consistency_factor = score_obj['pair_ratio_consistency_factor'],
      peak_modality = score_obj['peak_modality'],
      modality_factor = score_obj['modality_factor'],
      intensity_quantile = score_obj['intensity_quantile'],
      intensity_quantile_factor = score_obj['intensity_quantile_factor'],
      peak_start_factor = score_obj['peak_start_factor'],
      peak_end_factor = score_obj['peak_end_factor'],
      peak_location_factor = score_obj['peak_location_factor'],
      notes = score_obj['param']
    ))
  if is_stopping:
    done = True
    reward = score_obj['reward']
  current_obs = _shuffle_ions(current_obs)
  info = dict(chrom = current_chrom, score=score_obj)
  return current_obs, score_obj['reward'], done, info

class MSPeakFeatureExtractorNet(Module):
  def __init__(self, observation_space: gym_spaces_Box, features_dim=128, latent_dim=512):
    super().__init__()
    assert features_dim > 0
    self._observation_space = observation_space
    self._features_dim = features_dim
    self.convs = Sequential(
      Conv1d(observation_space.shape[0], 64, kernel_size = 3, stride=2),
      MaxPool1d(kernel_size=2),
      ReLU(),
      Conv1d(64, 32, kernel_size=3, stride=2),
      MaxPool1d(kernel_size=2),
      ReLU(),
      Flatten()
    )
    with th_no_grad():
      n_flatten = self.convs(th_as_tensor(observation_space.sample()[None]).float()).shape[1]
    self.fc = Sequential(
      Linear(n_flatten, latent_dim),
      Tanh(),
      Linear(latent_dim, features_dim),
      Tanh()
    )
  @property
  def features_dim(self) -> int:
    return self._features_dim
  def forward(self, observations: th_Tensor):
    return self.fc(self.convs(observations))

class MlpExtractor(Module):
  def __init__(self):
    super().__init__()
    self.policy_net = Sequential(
      Linear(128, 32, bias=True),
      Tanh(),
      Linear(32, 32, bias=True),
      Tanh()
    )
    self.value_net = Sequential(
      Linear(128, 32, bias=True),
      Tanh(),
      Linear(32, 32, bias=True),
      Tanh()
    )
  def forward(self, features: th_Tensor):
    latent_pi = self.policy_net(features)
    latent_vf = self.value_net(features)
    return latent_pi, latent_vf

class MsTargetPeakerPolicy(Module):
  def __init__(self, observation_space: gym_spaces_Box, features_dim=128, latent_dim=512):
    super().__init__()
    self.features_extractor = MSPeakFeatureExtractorNet(observation_space, features_dim, latent_dim)
    self.mlp_extractor = MlpExtractor()
    self.action_net = Linear(32, 2, bias=True)
    self.value_net = Linear(32, 1, bias=True)
  def forward(self,  obs: th_Tensor):
    features = self.features_extractor(obs)
    latent_pi, latent_vf = self.mlp_extractor(features)
    actions = self.action_net(latent_pi)
    values = self.value_net(latent_vf).detach()
    return actions, values

class MsTargetPeakerEnv():
  def __init__(self, chrom_tsv, policy_path=None, picked_peak_csv=None,
               max_step=5, device=None, 
               internal_standard_type='heavy',
               threshold_for_kde=2, 
               pair_ratio_factor=0.5,
               pbar_heavy_weight=2/3,
               intensity_power=1.0,
               quality_power=2.0,
               top_n_ion=1,
               use_weighted_pbar=True,
               use_kde=None,
               use_ref_if_no_kde=False):
    self.internal_standard_type = internal_standard_type
    self.max_time_point = 1024
    self.nFragment = 20
    self.max_step = max_step
    if device is None:
      self.device = 'cuda' if th_cuda_is_available() else 'cpu'
    else:
      self.device = device
    self.policy = MsTargetPeakerPolicy(gym_spaces_Box(low=0, high=1, shape=(42, 1024), dtype=np_float32), 128, 512)
    if policy_path is None:
      policy_path = os_path_join(os_path_split(__file__)[0], 'policy', 'MsTargetPeaker_env66_40-2.state_dict.pth')
    self.load_policy_state(policy_path)
    self.picked_peak_csv = picked_peak_csv
    self.peak_qc = dict(
      chromDB = ChromatogramDB(chrom_tsv, picked_peak_csv, internal_standard_type=internal_standard_type),
      qualityEval = PeakQualityEval(device=self.device)
    )
    self.pair_ratio_factor = pair_ratio_factor
    self.pbar_heavy_weight = pbar_heavy_weight
    self.intensity_power = intensity_power
    self.picked_peak_for_kde = use_kde or None
    self.peak_kde = None
    self.top_n_ion = top_n_ion
    self.quality_power = quality_power
    self.use_weighted_pbar = use_weighted_pbar
    print(f'Used picked peaks for kde: {use_kde}')
    if self.picked_peak_for_kde:
      kde_peak_df = read_csv(self.picked_peak_for_kde)
      kde_peak_df.columns = kde_peak_df.columns.str.replace(' ', '')
      self.peak_kde = {}
      print(f"Threshold for Peak Location KDE: {threshold_for_kde}.")
      print(f"Enter reference-first mode when no qualified peaks to form a KDE function: {use_ref_if_no_kde}.") 
      for pep, data in kde_peak_df.groupby('PeptideModifiedSequence'):
        if 'FinalReward' in data:
          valid_data = data[data['FinalReward'] >= threshold_for_kde]
        else:
          valid_data = data
        if len(valid_data) < 3:
          print(f'Cannot establish probability density functions of peak locations for moledcule {pep} because the number of peak groups are less than 3. Skipped.')
          if use_ref_if_no_kde:
            self.peak_kde[pep] = True # Reference-first mode to consider peaks mainly based on the reference peak height. (since no valid peaks can be used as references for finding peak locations in the last step of the rescue stage.)
          else:
            self.peak_kde[pep] = False # Use the provided search parameters to find peak locations.
          continue
        start_data = valid_data.sort_values(by='MinStartTime')
        x1 = start_data['MinStartTime']
        x1_weights= start_data['FinalReward'] if 'FinalReward' in start_data else None
        try:
          x1_kde = scipy_gaussian_kde(x1, weights=x1_weights)
        except:
          x1_kde = np_vectorize(lambda x: 1 if abs(x - np_mean(x1)) < 0.01 else 0)
        start_max = np_max(x1_kde(x1))
        end_data = valid_data.sort_values("MaxEndTime")
        x2 = end_data['MaxEndTime']
        x2_weights = end_data['FinalReward'] if 'FinalReward' in end_data else None
        try:
          x2_kde = scipy_gaussian_kde(x2, weights=x2_weights)
        except:
          x2_kde = np_vectorize(lambda x: 1 if abs(x - np_mean(x2)) < 0.01 else 0)
        end_max = np_max(x2_kde(x2))
        self.peak_kde[pep] = {
          'start': x1_kde,
          'end': x2_kde,
          'start_max': start_max,
          'end_max': end_max,
          'start_rt': x1.iloc[np_argmax(x1_kde(x1))],
          'end_rt': x2.iloc[np_argmax(x2_kde(x2))]
        }
      print(f'Got peak location distributions using kernel-density estimations from peaks in file {self.picked_peak_for_kde}.')
    print(f'Env option: pair_ratio_factor: {self.pair_ratio_factor}; pbar_heavy_weight: {self.pbar_heavy_weight}; intensity_power: {self.intensity_power}; quality_power: {self.quality_power}; use_kde: {"True" if self.peak_kde else "False"}')
    
  def load_policy_state(self, policy_path):
    self.policy.load_state_dict(th_load(policy_path, weights_only=True))
    self.policy.to(self.device)
    self.policy.eval()
    print(f'Successflly loaded the policy state dict from {policy_path}')

  def get_chrom_list(self):
    chrom_list = []
    for idx, row in self.peak_qc['chromDB'].chrom[['PeptideModifiedSequence', 'FileName']].drop_duplicates().iterrows():
      pep = row['PeptideModifiedSequence']
      file = row['FileName']
      chrom = self.peak_qc['chromDB'].getChromData(file, pep)
      if chrom is not None:
        chrom_list.append(chrom)
    return chrom_list

  def get_chrom_list_from_sample(self, sample_list, prescreen=0):
    chrom_list = []
    chromDB = self.peak_qc['chromDB']
    for fn, pep in sample_list:
      if prescreen > 0:
        chrom = self.init_chrom(fn, pep, prescreen=prescreen)
      else:
        chrom = chromDB.getChromData(fn, pep)
      if chrom is not None:
        chrom_list.append(chrom)
    return chrom_list

  def init_chrom(self, fn, pep, prescreen=300):
    chromDB = self.peak_qc['chromDB']
    chrom_candidate = chromDB.getChromData(fn, pep)
    if chrom_candidate is None:
      return None
    current_score = self.get_chrom_score(chrom_candidate)['final_reward']
    kde_obj = None
    if self.peak_kde and pep in self.peak_kde and type(self.peak_kde[pep]) != bool:
      kde_obj = self.peak_kde[pep]
    for _ in range(prescreen):
      if kde_obj is not None:
        chrom = chromDB.getChromData(fn, pep, start=kde_obj['start'].resample(1)[0][0], end=kde_obj['end'].resample(1)[0][0])
      else:
        chrom = chromDB.updateChromData(chrom_candidate)
      score = self.get_chrom_score(chrom)['final_reward']
      if score > current_score:
        chrom_candidate = chrom
        current_score = score
    return chrom_candidate

  def get_sample_lists(self, retry_threshold):
    need_repicking = None
    all_sample_list = []
    sample_list = []
    pre_picked = None
    if self.picked_peak_csv:
      need_repicking = dict()
      try:
        pre_picked = read_csv(self.picked_peak_csv)
        if 'FinalReward' in pre_picked:
          failed_records = pre_picked[pre_picked['FinalReward'] < retry_threshold]
          for idx, row in failed_records.iterrows():
            pep = row['Peptide Modified Sequence']
            file = row['File Name']
            if pep not in need_repicking:
                need_repicking[pep] = dict()
            need_repicking[pep][file] = row
      except:
        pre_picked = None
        print('Failed to read peak-picked CSV file')
    for _, row in self.peak_qc['chromDB'].chrom[['PeptideModifiedSequence', 'FileName']].drop_duplicates().iterrows():
      pep = row['PeptideModifiedSequence']
      file = row['FileName']
      all_sample_list.append((file, pep))
      if need_repicking is None or (pep in need_repicking and file in need_repicking[pep]):
        sample_list.append((file, pep))
    return all_sample_list, sample_list, need_repicking, pre_picked
  
  def random_peak(self, chrom_data):
    return self.peak_qc['chromDB'].updateChromData(chrom_data) # random peak start and end when not setting the start and end
  
  def batch_random_peak(self, batch_chrom, inplace=True):
    return [self.random_peak(chrom) for chrom in batch_chrom]
  
  def get_obs_from_chrom(self, chrom_data):
    return _chrom_to_current_obs(chrom_data)
  
  def batch_get_obs_from_chrom(self, batch_chrom):
    return [_chrom_to_current_obs(chrom) for chrom in batch_chrom]
  
  def reset_obs_df(self, obs, chrom):
    return _reset_obs(obs, chrom)
  
  def batch_reset_obs_df(self, batch_obs, batch_chrom):
    return [_reset_obs(obs, chrom) for (obs, chrom) in zip(batch_obs, batch_chrom)]

  def get_chrom_score(self, chrom_data):
    return _calculate_chrom_score(self.peak_qc, 
                                  chrom_data,
                                  pair_ratio_factor=self.pair_ratio_factor,
                                  pbar_heavy_weight=self.pbar_heavy_weight, 
                                  top_n_ion=self.top_n_ion, 
                                  use_weighted_pbar=self.use_weighted_pbar,
                                  intensity_power=self.intensity_power, 
                                  quality_power=self.quality_power,
                                  peak_kde=self.peak_kde)
  
  def step(self, obs_df, chrom_data, current_step, action, history=[], result_queue=None):
    results = _step(obs_df, self.peak_qc, chrom_data, current_step, action, history, self.max_step, 
                    pair_ratio_factor=self.pair_ratio_factor,
                    pbar_heavy_weight=self.pbar_heavy_weight, intensity_power=self.intensity_power, quality_power=self.quality_power,
                    top_n_ion=self.top_n_ion,
                    peak_kde=self.peak_kde, use_weighted_pbar=self.use_weighted_pbar)
    if result_queue:
      result_queue.put(results)
    return results

  def predict(self, obs_mx, returnValue = False):
    device = self.device
    obs = th_tensor(np_array([obs_mx]), dtype=th_float32, device=device)
    with th_no_grad():
      model_policy = self.policy
      features = model_policy.features_extractor(obs)
      latent_pi  = model_policy.mlp_extractor.policy_net(features)
      mean_actions = model_policy.action_net(latent_pi).detach()
      if returnValue:
        latent_vf = model_policy.mlp_extractor.value_net(features)
        value = model_policy.value_net(latent_vf).detach()
    actions = mean_actions.cpu().numpy()
    if returnValue:
      value = value[0][0].cpu().numpy()
      return actions[0], value
    else:
      return actions[0]
  
  def predict_val(self, obs_mx):
    obs = th_tensor(np_array([obs_mx]), dtype=th_float32, device=self.device)
    with th_no_grad():
      model_policy = self.policy
      features = model_policy.features_extractor(obs)
      latent_vf  = model_policy.mlp_extractor.value_net(features)
      value = model_policy.value_net(latent_vf).detach()
    return value[0][0].cpu().numpy()
  
  def run_mcts(self, chrom_init, option, label=None, initial_obs=None, result_queue=None, return_mcts_tree=False):
    cycle = option['cycle'] or 500
    alpha = option['alpha'] or 0.3
    beta = option['beta'] or 0.2
    K = option['K'] or 3**0.5
    selection_noise = option['selection_noise'] or 1.0
    if not label:
      print(f"\r\033[Kprocessing {chrom_init[0]} - {chrom_init[1]}", end='\r')
    else:
      print(f"\r\033[K{label} processing {chrom_init[0]} - {chrom_init[1]}", end='\r', flush=True)
    eval_mode = option['eval_mode'] or 'policy'
    if initial_obs is None:
      initial_obs = self.get_obs_from_chrom(chrom_init)
    mcts_model = MCTS_DPW(alpha=alpha, beta=beta, initial_obs = initial_obs, initial_chrom=chrom_init, env=self, K=K, eval_mode=eval_mode, selection_noise=selection_noise)
    observation = mcts_model.root.state
    chrom = chrom_init
    done = False
    step = 0
    fail_counter = 0
    while not done and (step < self.max_step):
      mcts_model.learn(cycle)
      try:
        action = mcts_model.best_action()
      except Exception as e:
        print(e)
        fail_counter += 1
        if fail_counter > 5:
          break
        else:
          continue
      observation, reward, done, info = self.step(observation, chrom, step, action, history=mcts_model.history)
      chrom = info['chrom']
      step += 1
      if not return_mcts_tree:
        mcts_model.forward(action, observation, chrom, step)
      if done:
        break
    best_peak = mcts_model.get_best_peak_from_history()
    if best_peak:
      best_peak['notes'] = f'{cycle}cycle_' + best_peak['notes']
    if result_queue:
      result_queue.put(best_peak)
    if not return_mcts_tree:
      del mcts_model
    gc_collect()
    if return_mcts_tree:
      return best_peak, mcts_model
    else:
      return best_peak
