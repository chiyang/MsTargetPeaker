from itertools import groupby
from .QualityEncoder import QualityEncoder
from numpy import (sort as np_sort, median as np_median, mean as np_mean, average as np_average, max as np_max, argmax as np_argmax, min as np_min, std as np_std, arange as np_arange, sqrt as np_sqrt, sum as np_sum, power as np_power,
                  nan as np_nan, isnan as np_isnan, nan_to_num as np_nan_to_num, abs as np_abs, zeros as np_zeros, fill_diagonal as np_fill_diagonal, floor as np_floor,
                  diff as np_diff, sign as np_sign, corrcoef as np_corrcoef, argsort as np_argsort, flip as np_flip,
                  hstack as np_hstack, vstack as np_vstack, concatenate as np_concatenate,
                  array as np_array, repeat as np_repeat, tile as np_tile, transpose as np_transpose, 
                  vsplit as np_vsplit, hsplit as np_hsplit, split as np_split,
                  errstate as np_errstate, where as np_where, apply_along_axis as np_apply_along_axis)

class PeakQualityEval():
  def __init__(self, device='auto'):
    self.flatness_factor = 0.05
    self.quality_encoder = QualityEncoder(device=device)

  def top_n_frag_idx_list(self, chrom_data, top_n=5):
    (_, heavy_peak_ints) = np_hsplit(chrom_data[4], 2)
    if heavy_peak_ints.shape[0] > 0:
      heavy_max_peak_ints = np_max(heavy_peak_ints, axis=0)
      transition_idx = np_argsort(heavy_max_peak_ints)[-top_n:]
      return np_flip(transition_idx)
    else:
      return np_arange(0, len(chrom_data[2][0]))[-top_n:]

  def getOverallScore(self, chrom_data, top_n=5):
    type1_quality_list = self.batchGetType1Quality([chrom_data])
    type1_scores =  self.quality_encoder(type1_quality_list[0])[0]
    (_, area2sum_ratio_heavy)  = np_hsplit(chrom_data[6], 2)
    q1_score_idx_list = np_flip(np_argsort(area2sum_ratio_heavy)[-top_n:])
    q1_score = np_median(type1_scores[q1_score_idx_list])
    if len(type1_scores) > top_n:
      (_, heavy_peak_ints) = np_hsplit(chrom_data[4], 2)
      if heavy_peak_ints.shape[0] > 0:
        heavy_max_peak_ints = np_max(heavy_peak_ints, axis=0)
        transition_idx = np_argsort(heavy_max_peak_ints)[-top_n:]
        transition_idx = np_concatenate((transition_idx, transition_idx + len(chrom_data[2][0])))
        type2_quality = self.makeType2Feature(chrom_data[4][:, transition_idx], chrom_data[5], chrom_data[6][transition_idx])
        q2_score = self.quality_encoder(np_array([type2_quality]))[0][0]
      else:
        type2_quality = self.batchGetType2Quality([chrom_data])
        q2_score = self.quality_encoder(np_array(type2_quality))[0][0]
    else:
      type2_quality = self.batchGetType2Quality([chrom_data])
      q2_score = self.quality_encoder(np_array(type2_quality))[0][0]
    return q1_score, q2_score
  
  def batchGetType1Score(self, chrom_data_list):
    type1_quality_list = self.batchGetType1Quality(chrom_data_list)
    return np_hstack([self.quality_encoder(q)[0] for q in type1_quality_list])
  def batchGetType2Score(self, chrom_data_list, top_n=None):
    type2_quality = self.batchGetType2Quality(chrom_data_list, top_n=top_n)
    return self.quality_encoder(np_array(type2_quality))[0]
  def batchGetType3Score(self, chrom_data_list):
    type3_quality_list = self.batchGetType3Quality(chrom_data_list)
    return np_hstack([self.quality_encoder(q)[0] for q in type3_quality_list])

  def batchGetType1Quality(self, chrom_data_list): #zip(peak_intensity_list, peak_time_list, area_sum_ratio_list)
    return [
      np_hstack(
        [np_hstack(np_vsplit(np_array([
          self.calculateTransitionJaggedness(x[4]),
          self.calculateTransitionSymmetry(x[4]),
          self.calculateTransitionModality(x[4]),
          self.calculateTransitionShift(x[4], x[5]),
          self.calculateTransitionFWHM(x[4], x[5]),
          self.calculateTransitionFWHM2base(x[4], x[5]),
          self.calculateTransitionMaxBoundaryIntensityNormalized(x[4])
        ]).T, 2)), 
        np_array([
          self.calculatePairSimilarity(x[4]),
          self.calculatePairShift(x[4], x[5]),
          self.calculatePairFWHMConsistency(x[4], x[5])
        ]).T])
      for x in chrom_data_list
    ]
  def processForType2(self, chrom_data, top_n):
      transition_idx = self.top_n_frag_idx_list(chrom_data, top_n)
      transition_idx = np_concatenate((transition_idx, transition_idx + len(chrom_data[2][0])))
      return self.makeType2Feature(chrom_data[4][:, transition_idx], chrom_data[5], chrom_data[6][transition_idx])
  def batchGetType2Quality(self, chrom_data_list, top_n = None):
    if top_n is None:
        return [self.makeType2Feature(x[4], x[5], x[6]) for x in chrom_data_list]
    else:
        return [self.processForType2(x, top_n) for x in chrom_data_list]
  def makeType2Feature(self, peak_ints, peak_time, area_ratio):
    return np_hstack([
      np_hstack(np_vsplit(np_array([
        self.calculateIsotopeJaggedness(peak_ints),
        self.calculateIsotopeSymmetry(peak_ints),
        self.calculateIsotopeSimilarity(peak_ints),
        self.calculateIsotopeModality(peak_ints),
        self.calculateIsotopeShift(peak_ints, peak_time),
        self.calculateIsotopeFWHM(peak_ints, peak_time),
        self.calculateIsotopeFWHM2base(peak_ints, peak_time)
      ]).T, 2))[0],
      np_array([
        self.calculatePeakGroupRatioCorr(area_ratio),
        self.calculatePeakGroupJaggedness(peak_ints),
        self.calculatePeakGroupSymmetry(peak_ints),
        self.calculatePeakGroupSimilarity(peak_ints),
        self.calculatePeakGroupModality(peak_ints),
        self.calculatePeakGroupShift(peak_ints, peak_time),
        self.calculatePeakGroupFWHM(peak_ints, peak_time),
        self.calculatePeakGroupFWHM2base(peak_ints, peak_time)
      ])
    ])
  def getType3Quality(self, grouped):
    file_number = len(grouped)
    result = np_hstack([
        np_transpose(np_hsplit(self.calculateMeanIsotopeRatioConsistency(grouped), 2), axes=(0, 2, 1)).reshape(2, 1, -1).squeeze(1).T,
        np_transpose(np_hsplit(self.calculateMeanIsotopeFWHMConsistency(grouped), 2), axes=(0, 2, 1)).reshape(2, 1, -1).squeeze(1).T,
        np_transpose(np_repeat(np_hsplit(self.calculateArea2SumRatioCV(grouped), 2), file_number).reshape(2, -1))
    ])
    result = np_hstack([
      result[:, ::2],  # Light
      result[:, 1::2], # Heavy
      np_array([
        np_tile(self.calculateMeanIsotopeRTConsistency(grouped), len(grouped[0][2][0])).T,
        self.calculatePairRatioConsistency(grouped).transpose().reshape(1, -1)[0]
      ]).T
    ])
    return np_vstack([result[i::file_number, :] for i in range(file_number)]) # ordering: 1) filename; 2) transition ions
  def batchGetType3Quality(self, chrom_data_list):
    hasVariedTransitions = False
    mol2transitions = {}
    sorted_chrom_data_list = sorted(chrom_data_list, key=lambda x: x[0])
    for mol, chrom_list in groupby(sorted_chrom_data_list, key=lambda x: x[0]):
      trans_set = None
      for chrom in chrom_list:
        if trans_set is None:
          trans_set = set(chrom[2][0])
          mol2transitions[mol] = trans_set
          continue
        chrom_trans = set(chrom[2][0])
        mol2transitions[mol] = mol2transitions[mol].union(chrom_trans)
        if len(trans_set) != len(chrom_trans) or trans_set != chrom_trans:
          hasVariedTransitions = True
          break
      if hasVariedTransitions:
        break
    if not hasVariedTransitions:
      return [self.getType3Quality(tuple(x[1])) for x in groupby(sorted_chrom_data_list, key=lambda x: x[0])] # chrom_data_list should be sorted by molecules
    else:
      print('Varied transition numbers are currently unsupported')

# TargetedMSQC metric calculations
# Intensity(1)
  def calculateTransitionMaxBoundaryIntensityNormalized(self, peak_intensity_matrix): #Level 1
    # Set the upperbound value to 5  (the worst case value is 5)
    if peak_intensity_matrix.shape[0] > 0:
      transition_max_boundary_intensity = peak_intensity_matrix[[0, -1], :].max(axis=0)  
      transition_max_intensity = np_max(peak_intensity_matrix, axis=0)
      with np_errstate(divide='ignore', invalid='ignore'):
        TransitionMaxBoundaryIntensityNormalized = transition_max_boundary_intensity / transition_max_intensity
    else:
      TransitionMaxBoundaryIntensityNormalized = np_array([np_nan] * peak_intensity_matrix.shape[1])
    np_nan_to_num(TransitionMaxBoundaryIntensityNormalized, copy=False, nan=5.0)
    TransitionMaxBoundaryIntensityNormalized = np_where(TransitionMaxBoundaryIntensityNormalized < 5, TransitionMaxBoundaryIntensityNormalized, 5)
    return TransitionMaxBoundaryIntensityNormalized

# Shift(4)
  def calculatePairShift(self, peak_intensity_matrix, peak_time): #4 Level 1
    if len(peak_intensity_matrix) == 0 or len(peak_time) <= 1:
      return [1] * int(peak_intensity_matrix.shape[1]/2)
    paired_peak_intensity = np_hsplit(peak_intensity_matrix, 2)
    peak_height_indices = np_argmax(paired_peak_intensity, axis=1)
    PairShift = np_abs(peak_time[peak_height_indices][0] - peak_time[peak_height_indices][1]) / (peak_time[-1]-peak_time[0])
    return PairShift
  def calculateTransitionShift(self, peak_intensity_matrix, peak_time): #8 Level 1
    transition_ion_number = peak_intensity_matrix.shape[1]
    if len(peak_intensity_matrix) < 2:
      return np_array([1] * transition_ion_number)
    peak_max_time = peak_time[peak_intensity_matrix.argmax(axis=0)]
    peak_max_median = np_median(peak_max_time)
    return np_abs(peak_max_time - peak_max_median) / (peak_time[-1]-peak_time[0])
  def calculateIsotopeShift(self, peak_intensity_matrix, peak_time): #2 Level 2
    transition_shift = self.calculateTransitionShift(peak_intensity_matrix, peak_time)
    return np_mean(np_hsplit(transition_shift, 2), axis=1)
  def calculatePeakGroupShift(self,peak_intensity_matrix, peak_time): #1 #Level 2
    transition_shift = self.calculateTransitionShift(peak_intensity_matrix, peak_time)
    return np_mean(transition_shift)

# Jaggedness(3)
  def calculateTransitionJaggedness(self, peak_intensity_matrix=np_array([[]])):
    if peak_intensity_matrix.shape[0] <= 2:
      return np_array([1]*peak_intensity_matrix.shape[1])
    peak_diff_arr = np_diff(peak_intensity_matrix, axis=0)
    peak_diff_arr[np_abs(peak_diff_arr) < self.flatness_factor * np_max(peak_intensity_matrix, axis=0)] = 0
    with np_errstate(divide='ignore', invalid='ignore'):
      jaggedness = ((np_abs(np_diff(np_sign(peak_diff_arr), axis=0)) > 1).sum(axis=0) - 1) / (len(peak_diff_arr)-1)
    np_nan_to_num(jaggedness, copy=False, nan=1)
    jaggedness = np_where(jaggedness > 0, jaggedness, 0)
    return jaggedness
  def calculateIsotopeJaggedness(self, peak_intensity_matrix):
    jaggedness = self.calculateTransitionJaggedness(peak_intensity_matrix)
    isotope_jaggedness = np_mean(np_split(jaggedness, 2), axis=1)
    return isotope_jaggedness
  def calculatePeakGroupJaggedness(self, peak_intensity_matrix): #1 #Level 2
    jaggedness = self.calculateTransitionJaggedness(peak_intensity_matrix)
    return np_mean(jaggedness)
  # Similarity(3)
  def calculatePairSimilarityMatrix(self, peak_intensity_matrix): #matrix
    transition_ion_number = peak_intensity_matrix.shape[1]
    if peak_intensity_matrix.shape[0] < 2:
      PairSimilarityMx = np_zeros((transition_ion_number, transition_ion_number))
      np_fill_diagonal(PairSimilarityMx, 1)
    else:
      with np_errstate(divide='ignore', invalid='ignore'):
        PairSimilarityMx = np_corrcoef(peak_intensity_matrix,  rowvar=False)
    np_nan_to_num(PairSimilarityMx, copy=False, nan=0)
    return PairSimilarityMx
  def calculatePairSimilarity(self, peak_intensity_matrix): #4 Level 1
    similarity_mx = self.calculatePairSimilarityMatrix(peak_intensity_matrix)
    frag_ion_number = int(peak_intensity_matrix.shape[1]/2)
    return similarity_mx[frag_ion_number:, :frag_ion_number].diagonal()
  def calculateIsotopeSimilarity(self, peak_intensity_mx): #2 #Level 2
    similarity_mx = self.calculatePairSimilarityMatrix(peak_intensity_mx)
    frag_num = int(peak_intensity_mx.shape[1]/2)
    light_mean = similarity_mx[:frag_num, :frag_num].mean()
    heavy_mean = similarity_mx[frag_num:, frag_num:].mean()
    return np_array([light_mean, heavy_mean])
  def calculatePeakGroupSimilarity(self, peak_intensity_mx): #1 #Level 2
    similarity_mx = self.calculatePairSimilarityMatrix(peak_intensity_mx)
    return np_mean(similarity_mx)
  
# Symmetry(3)
  def calculateTransitionSymmetry(self, peak_intensity_matrix): #8 #Level 1
    timepoint_number = np_floor(peak_intensity_matrix.shape[0]/2).astype(int)
    if timepoint_number <= 2:
      return np_array([0] * peak_intensity_matrix.shape[1])
    left = peak_intensity_matrix[:(timepoint_number)]
    right = peak_intensity_matrix[:(timepoint_number):-1]
    min_len = np_min((left.shape[0], right.shape[0])) # in case that the length is an odd number
    with np_errstate(divide='ignore', invalid='ignore'):
      corr_mx = np_corrcoef(left[:min_len,:], right[:min_len,:], rowvar=False)
    np_nan_to_num(corr_mx, copy=False, nan=0)
    return np_vsplit(corr_mx, 2)[1].diagonal()
  def calculateIsotopeSymmetry(self, peak_intensity_matrix): #2 #Level 2
    symmetry_arr = self.calculateTransitionSymmetry(peak_intensity_matrix)
    return np_mean(np_hsplit(symmetry_arr, 2), axis=1)
  def calculatePeakGroupSymmetry(self, peak_intensity_matrix): #1 #Level 2
    symmetry_arr = self.calculateTransitionSymmetry(peak_intensity_matrix)
    return np_mean(symmetry_arr)

# FWHM(8)
  def calc_fwhm(self, sig, time):
    peakmax = np_max(sig)
    higher_than_halfpeak = np_where((sig - peakmax/2) > 0)[0]
    try:
      left_index = np_array([higher_than_halfpeak[0] - 1, higher_than_halfpeak[0]])
    except:
      left_index = np_array([np_nan, np_nan])
    try:
      right_index = np_array([higher_than_halfpeak[-1], higher_than_halfpeak[-1] + 1])
    except:
      right_index = np_array([np_nan, np_nan])
    if (left_index[0] == -1) or (np_isnan(left_index[0])):
      t_left = time[0]
    else:
      t_left = (time[left_index[1]] - time[left_index[0]])/(sig[left_index[1]] - sig[left_index[0]])*(peakmax/2 - sig[left_index[0]]) + time[left_index[0]]
    if (right_index[1] > (len(time)-1)) or (np_isnan(right_index[1])):
      t_right = time[-1]
    else:
      t_right = (time[right_index[1]] - time[right_index[0]])/(sig[right_index[1]] - sig[right_index[0]])*(peakmax/2 - sig[right_index[0]]) + time[right_index[0]]
    fwhm = t_right - t_left
    return fwhm
  def calculateTransitionFWHM(self, peak_intensity_matrix, peak_time): #8
    if len(peak_time) < 1:
        return np_array([5] * peak_intensity_matrix.shape[1])
    TransitionFWHM = np_apply_along_axis(self.calc_fwhm, 0, peak_intensity_matrix, peak_time)
    TransitionFWHM = np_where(TransitionFWHM < 5, TransitionFWHM, 5) # Set the upperbound value to 5
    return TransitionFWHM
  def calculateTransitionFWHM2base(self, peak_intensity_matrix, peak_time): #8 #Level 1
    TransitionFWHM = self.calculateTransitionFWHM(peak_intensity_matrix, peak_time)
    TransitionFWHM2Base = TransitionFWHM/(peak_time[-1] - peak_time[0]) if len(peak_time) >= 2 else TransitionFWHM
    return np_where(TransitionFWHM2Base < 5, TransitionFWHM2Base, 5) # Set the upperbound value to 5
  def calculateIsotopeFWHM(self, peak_intensity_matrix, peak_time): #2 #Level 2
    TransitionFWHM = self.calculateTransitionFWHM( peak_intensity_matrix, peak_time)
    IsotopeFWHM = np_mean(np_hsplit(TransitionFWHM, 2), axis=1)
    return np_where(IsotopeFWHM < 5, IsotopeFWHM, 5) # Set the upperbound value to 5
  def calculateIsotopeFWHM2base(self, peak_intensity_matrix, peak_time): #2 Level 2
    TransitionFWHM2Base = self.calculateTransitionFWHM2base(peak_intensity_matrix, peak_time)
    IsotopeFWHM2Base = np_mean(np_hsplit(TransitionFWHM2Base, 2), axis=1)
    return np_where(IsotopeFWHM2Base < 5, IsotopeFWHM2Base, 5) # Set the upperbound value to 5
  def calculatePeakGroupFWHM(self, peak_intensity_matrix, peak_time): #1 #Level 2
    TransitionFWHM = self.calculateTransitionFWHM(peak_intensity_matrix, peak_time)
    PeakGroupFWHM = np_mean(TransitionFWHM)
    return PeakGroupFWHM if PeakGroupFWHM < 5 else 5
  def calculatePeakGroupFWHM2base(self, peak_intensity_matrix, peak_time): #1 #Level 2
    TransitionFWHM2Base = self.calculateTransitionFWHM2base(peak_intensity_matrix, peak_time)
    PeakGroupFWHM2base = np_mean(TransitionFWHM2Base)
    return PeakGroupFWHM2base if PeakGroupFWHM2base < 5 else 5
  def calculatePairFWHMConsistency(self, peak_intensity_matrix, peak_time): #4 Level 1
    TransitionFWHM = self.calculateTransitionFWHM(peak_intensity_matrix, peak_time)
    [lightFWHM, heavyFWHM] = np_hsplit(TransitionFWHM, 2)
    with np_errstate(divide='ignore', invalid='ignore'):
      PairFWHMConsistency = np_abs((lightFWHM - heavyFWHM)/heavyFWHM)
    np_nan_to_num(PairFWHMConsistency, copy=False, nan=5)
    return np_where(PairFWHMConsistency < 5, PairFWHMConsistency, 5) # set the upperbound value to 5
  def calculateMeanIsotopeFWHMConsistency(self, chrom_data_list): #8 # Level 3
    TransitionFWHM_crossAll = np_array([self.calculateTransitionFWHM(x[4], x[5]) for x in chrom_data_list])
    mean = np_mean(TransitionFWHM_crossAll, axis=0)
    with np_errstate(divide='ignore', invalid='ignore'):
      MeanIsotopeFWHMConsistency = np_abs(TransitionFWHM_crossAll - mean)/mean
    np_nan_to_num(MeanIsotopeFWHMConsistency, copy=False, nan=5)
    return np_where(MeanIsotopeFWHMConsistency < 5, MeanIsotopeFWHMConsistency, 5)

# Modality(3)
  def calculateTransitionModality(self, peak_intensity_matrix): #8 #Level 1
    if peak_intensity_matrix.shape[0] < 2:
      return np_array([1]*peak_intensity_matrix.shape[1])
    peak_diff_arr = np_diff(peak_intensity_matrix, axis=0)
    peak_diff_arr[np_abs(peak_diff_arr) < self.flatness_factor * np_max(peak_intensity_matrix, axis=0)] = 0
    def first_fall(series):
      fall_list = np_where(series < 0)[0]
      return fall_list[0] if len(fall_list) > 0 else len(series)
    def last_rise(series):
      rise_list = np_where(series > 0)[0]
      return rise_list[-1] if len(rise_list) > 0 else -1
    first_falls = np_apply_along_axis(first_fall, 0, peak_diff_arr)
    last_rises = np_apply_along_axis(last_rise, 0, peak_diff_arr)
    TransitionModality = []
    for idx, (data_series, f, l) in enumerate(zip(peak_diff_arr.T, first_falls, last_rises)):
      max_dip = 0
      if f < l:
        max_dip = np_max(np_abs(data_series[f:l+1]))
      if len(data_series) == 0:
        modality = 1
      elif np_max(data_series) == 0:
        modality = 1
      else:
        modality = max_dip/np_max(peak_intensity_matrix[:, idx])
      TransitionModality.append(modality)
    return np_array(TransitionModality)
  def calculateIsotopeModality(self, peak_intensity_matrix): #2 #Level 2
    TransitionModality = self.calculateTransitionModality(peak_intensity_matrix)
    return np_mean(np_hsplit(TransitionModality, 2), axis=1)
  def calculatePeakGroupModality(self, peak_intensity_matrix): #1 #Level 2
    TransitionModality = self.calculateTransitionModality(peak_intensity_matrix)
    return np_mean(TransitionModality)

# Area Ratio(4)
  def calculateArea2SumRatioCV(self, chrom_data_list): #8(sum補0) # Level 3
    Area2SumRatios = np_array([x[6] for x in chrom_data_list])
    mean_ratio = np_mean(Area2SumRatios, axis=0)
    with np_errstate(divide='ignore', invalid='ignore'):
      Area2SumRatioCV = np_std(Area2SumRatios, axis=0, ddof=1) / mean_ratio
    np_nan_to_num(Area2SumRatioCV, copy=False, nan=5)
    return np_where(Area2SumRatioCV < 5, Area2SumRatioCV, 5) # Set the upperbound value to 5
  def calculatePeakGroupRatioCorr(self, area2sum_ratios):
    (light_ratios, heavy_ratios) = np_hsplit(area2sum_ratios, 2)
    if len(light_ratios) <= 2 or len(heavy_ratios) <= 2:
      corr = 0
    else:
      with np_errstate(divide='ignore', invalid='ignore'):
        corr = np_corrcoef(light_ratios, heavy_ratios)[0][1]
    np_nan_to_num(corr, copy=False, nan=0)
    return 0 if np_isnan(corr) else corr
  def calculateEachPairRatioConsistency(self, area2sum_ratios):
    (light_ratios, heavy_ratios) = np_hsplit(area2sum_ratios, 2)
    with np_errstate(divide='ignore', invalid='ignore'):
      PairRatioConsistency = np_abs((light_ratios - heavy_ratios)/heavy_ratios)
      np_nan_to_num(PairRatioConsistency, copy=False, nan=5)
      PairRatioConsistency = np_where(PairRatioConsistency < 5, PairRatioConsistency, 5)
    return PairRatioConsistency
  def calculatePairRatioConsistency(self, chrom_data_list):
    PairRatioConsistency = np_array([self.calculateEachPairRatioConsistency(x[6]) for x in chrom_data_list])
    return PairRatioConsistency
  def calculateMeanIsotopeRatioConsistency(self, chrom_data_list): #8(sum補0) #Level 3, each File has its own value
    Area2SumRatio = np_array([x[6] for x in chrom_data_list])
    mean = np_mean(Area2SumRatio, axis=0)
    with np_errstate(divide='ignore', invalid='ignore'):
      MeanIsotopeRatioConsistency = np_abs((Area2SumRatio - mean)/mean)
    np_nan_to_num(MeanIsotopeRatioConsistency, copy=False, nan=5)
    return np_where(MeanIsotopeRatioConsistency < 5, MeanIsotopeRatioConsistency, 5)  

# RT(1)
  def calculateMeanIsotopeRTConsistency(self, chrom_data_list): #1(sum補相同數值) # Level 3
    peakRT_crossAll = np_array([ (x[3][0] + x[3][1])/2 for x in chrom_data_list])
    mean = np_mean(peakRT_crossAll)
    with np_errstate(divide='ignore', invalid='ignore'):
      meanIsotopeRTconsistency = np_abs(peakRT_crossAll - mean) / mean
    np_nan_to_num(meanIsotopeRTconsistency, copy=False, nan=5)
    return np_where(meanIsotopeRTconsistency < 5, meanIsotopeRTconsistency, 5)

# TMSQE quality feature names
  def type1QualityTitle(self):
    return [
      'TransitionJaggedness_light',
      'TransitionSymmetry_light',
      'TransitionModality_light',
      'TransitionShift_light',
      'TransitionFWHM_light',
      'TransitionFWHM2base_light',
      'TransitionMaxBoundaryIntensityNormalized_light',
      'TransitionJaggedness_heavy',
      'TransitionSymmetry_heavy',
      'TransitionModality_heavy',
      'TransitionShift_heavy',
      'TransitionFWHM_heavy',
      'TransitionFWHM2base_heavy',
      'TransitionMaxBoundaryIntensityNormalized_heavy',
      'PairSimilarity',
      'PairShift',
      'PairFWHMConsistency'
    ]
  def type2QualityTitle(self):
    return [
      'IsotopeJaggedness_light',
      'IsotopeSymmetry_light',
      'IsotopeSimilarity_light',
      'IsotopeModality_light',
      'IsotopeShift_light',
      'IsotopeFWHM_light',
      'IsotopeFWHM2base_light',
      'IsotopeJaggedness_heavy',
      'IsotopeSymmetry_heavy',
      'IsotopeSimilarity_heavy',
      'IsotopeModality_heavy',
      'IsotopeShift_heavy',
      'IsotopeFWHM_heavy',
      'IsotopeFWHM2base_heavy',
      'PeakGroupRatioCorr',
      'PeakGroupJaggedness',
      'PeakGroupSymmetry',
      'PeakGroupSimilarity',
      'PeakGroupModality',
      'PeakGroupShift',
      'PeakGroupFWHM',
      'PeakGroupFWHM2base'
    ]
  def type3QualityTitle(self):
    return [
      "MeanIsotopeRatioConsistency_light",
      "MeanIsotopeFWHMConsistency_light",
      "Area2SumRatioCV_light",
      "MeanIsotopeRatioConsistency_heavy",
      "MeanIsotopeFWHMConsistency_heavy",
      "Area2SumRatioCV_heavy",
      "MeanIsotopeRTConsistency",
      "PairRatioConsistency"
    ]
