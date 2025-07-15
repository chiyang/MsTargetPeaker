from .PeakQualityEval import PeakQualityEval
from .ChromatogramDB import ChromatogramDB
import numpy as np
import pandas as pd
import time
import os
import re
import math
from scipy.stats import gaussian_kde
from itertools import groupby
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class PeakQualityReport():
  def __init__(self, chromatogram_tsv, peak_csv, output_folder, internal_standard_type='heavy', top_n_transitions=5, group_csv=None, quantifier_csv=None):
    self.internal_standard_type = internal_standard_type if internal_standard_type == 'heavy' else 'light'
    self.chrom_db = ChromatogramDB(chromatogram_tsv, peak_csv, internal_standard_type=self.internal_standard_type, groupFile=group_csv)
    self.quantifier_df = pd.read_csv(quantifier_csv, header=None, index_col=0).T if quantifier_csv is not None else None
    self.mol2quantifier = self.quantifier_df.to_dict('records')[0] if self.quantifier_df else None
    self.peak_qc = PeakQualityEval()
    self.transition_quality = {}
    self.top_n_transitions = top_n_transitions
    self.peak_qc = PeakQualityEval()
    self.output_folder = output_folder
    self.peak_location_kde = {}
    self._hasGroup = self.chrom_db.hasGroup()
    self.quality_warn_range_config = dict(
      # Type 1 quality
      TransitionJaggedness_light = dict(min=0, max=0.3, direction=-1),
      TransitionSymmetry_light=dict(min=0, max=1, direction=1),
      TransitionModality_light=dict(min=0, max=0.3, direction=-1),
      TransitionShift_light=dict(min=0, max=0.3, direction=-1),
      TransitionFWHM_light=dict(min=0.05, max=1, direction=-1),
      TransitionFWHM2base_light=dict(min=0.25, max=1, direction=-1),
      TransitionMaxBoundaryIntensityNormalized_light=dict(min=0, max=1, direction=-1),
      TransitionJaggedness_heavy = dict(min=0, max=0.3, direction=-1),
      TransitionSymmetry_heavy=dict(min=0, max=1, direction=1),
      TransitionModality_heavy=dict(min=0, max=0.3, direction=-1),
      TransitionShift_heavy=dict(min=0, max=0.3, direction=-1),
      TransitionFWHM_heavy=dict(min=0.05, max=1, direction=-1),
      TransitionFWHM2base_heavy=dict(min=0.25, max=1, direction=-1),
      TransitionMaxBoundaryIntensityNormalized_heavy=dict(min=0, max=1, direction=-1),
      PairSimilarity=dict(min=0.7, max=1, direction=1),
      PairShift=dict(min=0, max=0.3, direction=-1),
      PairFWHMConsistency=dict(min=0, max=1, direction=-1),
      # Type 2 quality
      IsotopeJaggedness_light=dict(min=0, max=0.3, direction=-1),
      IsotopeSymmetry_light=dict(min=0, max=1, direction=1),
      IsotopeSimilarity_light=dict(min=0.7, max=1, direction=1),
      IsotopeModality_light=dict(min=0, max=0.3, direction=-1),
      IsotopeShift_light=dict(min=0, max=0.3, direction=-1),
      IsotopeFWHM_light=dict(min=0.2, max=1, direction=-1),
      IsotopeFWHM2base_light=dict(min=0.2, max=1, direction=-1),
      IsotopeJaggedness_heavy=dict(min=0, max=0.3, direction=-1),
      IsotopeSymmetry_heavy=dict(min=0, max=1, direction=1),
      IsotopeSimilarity_heavy=dict(min=0.7, max=1, direction=1),
      IsotopeModality_heavy=dict(min=0, max=0.3, direction=-1),
      IsotopeShift_heavy=dict(min=0, max=0.3, direction=-1),
      IsotopeFWHM_heavy=dict(min=0.2, max=1, direction=-1),
      IsotopeFWHM2base_heavy=dict(min=0.2, max=1, direction=-1),
      PeakGroupRatioCorr=dict(min=0.7, max=1, direction=1),
      PeakGroupJaggedness=dict(min=0, max=0.3, direction=-1),
      PeakGroupSymmetry=dict(min=0, max=1, direction=1),
      PeakGroupSimilarity=dict(min=0.7, max=1, direction=1),
      PeakGroupModality=dict(min=0, max=0.3, direction=-1),
      PeakGroupShift=dict(min=0, max=0.3, direction=-1),
      PeakGroupFWHM=dict(min=0.2, max=1, direction=-1),
      PeakGroupFWHM2base=dict(min=0.2, max=1, direction=-1),
      #Type 3 quality
      MeanIsotopeRatioConsistency_light=dict(min=0, max=0.5, direction=-1),
      MeanIsotopeFWHMConsistency_light=dict(min=0, max=1, direction=-1),
      Area2SumRatioCV_light=dict(min=0, max=0.2, direction=-1),
      MeanIsotopeRatioConsistency_heavy=dict(min=0, max=0.5, direction=-1),
      MeanIsotopeFWHMConsistency_heavy=dict(min=0, max=1, direction=-1),
      Area2SumRatioCV_heavy=dict(min=0, max=0.2, direction=-1),
      MeanIsotopeRTConsistency=dict(min=0, max=0.5, direction=-1),
      PairRatioConsistency=dict(min=0, max=0.5, direction=-1)
    )
  def run(self):
    start = time.time()
    self.load_chrom()
    print(f"{len(self.chrom_list)} chromatograms loaded.")
    print(f"Summarizing peak quality...")
    self.summarize_dataset()
    print(f"Output transition quality to {os.path.join(self.output_folder, 'transition_quality')}")
    self.output_transition_quality()
    print(f"Output sample quality to {os.path.join(self.output_folder, 'sample_quality.xlsx')}.")
    self.output_sample_quality()
    self.output_peak_location_distribution()
    end = time.time()
    print(f'Finished quality reporting in {round((end - start)/60, 2)} minutes.')
    
  def load_chrom(self):
    self.sample_list = self.chrom_db.getSampleList()
    self.chrom_list = []
    for fn, ps in tqdm(self.sample_list):
      self.chrom_list.append(self.chrom_db.getChromData(fn, ps, chrom_only=True, missing_transitions=True))
    self.chrom_list = list(filter(lambda x: x is not None, self.chrom_list))
    self.chrom_list = list(sorted(self.chrom_list, key= lambda x: x[0]))
    self.chrom_dict = {}
    for chrom_data in self.chrom_list:
      (mol, fn) = (chrom_data[0], chrom_data[1])
      if mol not in self.chrom_dict:
        self.chrom_dict[mol] = {}
      self.chrom_dict[mol][fn] = chrom_data

  def calculate_peak_loc_kde(self):
    peak_df = self.chrom_db.peak_boundary
    self.peak_location_kde = {}
    for mol, data in peak_df.groupby('PeptideModifiedSequence'):
      data = data.reset_index()
      data = data[(data['MinStartTime'].notna()) & (data['MaxEndTime'].notna())]
      x1 = data['MinStartTime']
      x2 = data['MaxEndTime']
      peak_width = (x2 - x1).to_numpy()
      try:
        x1_kde = gaussian_kde(x1)
      except Exception as e:
        print(e)
        x1_kde = lambda x: 1 if abs(x - np.mean(x1)) < 0.01 else 0
      try:
        x2_kde = gaussian_kde(x2)
      except Exception as e:
        print(e)
        x2_kde = lambda x: 1 if abs(x - np.mean(x2)) < 0.01 else 0
      try:
        width_kde = gaussian_kde(peak_width)
      except Exception as e:
        print(e)
        def width_kde_func(x):
            print(type(x))
            if type(x) is float:
                return 1 if abs(x - np.mean(peak_width)) < 0.01 else 0
            else:
                return np.array([1 if np.abs(y - np.mean(peak_width)) < 0.01 else 0 for y in x])
        width_kde = width_kde_func
      self.peak_location_kde[mol] = {
        'start': x1_kde,
        'end': x2_kde,
        'start_max': np.max(x1_kde(x1)),
        'end_max': np.max(x2_kde(x2)),
        'width': width_kde,
        'width_max': np.max(width_kde(peak_width)),
        'start_rt': x1.iloc[np.argmax(x1_kde(x1))],
        'end_rt': x2.iloc[np.argmax(x2_kde(x2))],
      }

  def _rankFragments(self, store_dict, ordered_fragments):
    for idx, frag in enumerate(ordered_fragments):
      if frag not in store_dict:
        store_dict[frag] = 0
      store_dict[frag] += (len(ordered_fragments) - idx)

  def rankFragments(self, store_dict, original_transitions, fragment_values, top_n=0):
    if top_n >= 1:
      frag_order = np.flip(np.argsort(fragment_values))[:top_n]
    else:
      frag_order = np.flip(np.argsort(fragment_values))
    ordered_transitions = original_transitions[frag_order]
    self._rankFragments(store_dict, ordered_transitions)
    return ordered_transitions, frag_order

  def summarize_dataset(self):
    self.transition_quality = {}
    self.sample_quality = {}
    self.group_quality = {}
    self.calculate_peak_loc_kde()
    sorted_chrom_list = sorted(self.chrom_list, key=lambda x: x[0])
    for mol, target_chrom_list in groupby(sorted_chrom_list, key=lambda x: x[0]):
      quantifier = self.mol2quantifier[mol] if self.mol2quantifier and mol in self.mol2quantifier else None
      if mol not in self.transition_quality:
        self.transition_quality[mol] = []
        self.sample_quality[mol] = []
        self.group_quality[mol] = []
      if self._hasGroup:
        sorted_by_group = sorted(target_chrom_list, key=lambda x: (x[12] is None, x[12]))
      else:
        sorted_by_group = sorted(target_chrom_list, key=lambda x: x[0])
      group_iter = groupby(sorted_by_group, key=lambda x: (x[12] is None, x[12])) if self._hasGroup else groupby(sorted_by_group, key=lambda x: x[0])
      for group_key, each_group_chrom_list in group_iter:
        group = group_key[1] if self._hasGroup else group_key
        if group is None:
          continue
        each_group_chrom_list = tuple(each_group_chrom_list)
        grouped_chroms = tuple(filter(lambda x: x[3] is not None, each_group_chrom_list))
        grouped_chrom_dict = {}
        for chrom in grouped_chroms:
          if chrom[0] not in grouped_chrom_dict:
            grouped_chrom_dict[chrom[0]] = {}
          grouped_chrom_dict[chrom[0]][chrom[1]] = chrom
        type1_feature = self.peak_qc.batchGetType1Quality(grouped_chroms)
        type1_quality = [self.peak_qc.quality_encoder(q)[0] for q in type1_feature]
        type2_feature = self.peak_qc.batchGetType2Quality(grouped_chroms, top_n=self.top_n_transitions)
        type2_quality = self.peak_qc.quality_encoder(np.array(type2_feature))[0]
        type3_feature = self.peak_qc.batchGetType3Quality(grouped_chroms)[0]
        pair_ratio_consistency = type3_feature[:, 7]
        pair_ratio_consistency = np.where(pair_ratio_consistency < 1, pair_ratio_consistency, 1)
        mean_isotope_rt_consistency = type3_feature[:, 6]
        mean_isotope_rt_consistency = np.where(mean_isotope_rt_consistency < 1, mean_isotope_rt_consistency, 1)
        type3_quality = self.peak_qc.quality_encoder(type3_feature)[0]
        target_trans_idx = 0
        light_peaktop_ranks = {}
        heavy_peaktop_ranks = {}
        light_peakarea_ranks = {}
        heavy_peakarea_ranks = {}
        q3_frag_ranks = {}
        q1_frag_ranks = {}
        peakstart_list = []
        peakend_list = []
        peakwidth_list = []
        peakstart_density_list = []
        peakend_density_list =   []
        peakwidth_density_list = []
        peakloc_list = []
        light_peaktop_list =  []
        light_peakarea_list = []
        heavy_peaktop_list =  []
        heavy_peakarea_list = []
        peaktop_ratio_list =  []
        peakarea_ratio_list = []
        q1_scores = []
        q2_scores = []
        q3_scores = []
        chrom_idx = -1
        for chrom in each_group_chrom_list:
          fn = chrom[1]
          transitions = np.array(chrom[2][0])
          quantifier_idx = chrom[2][0].index(quantifier) if quantifier and quantifier in chrom[2][0] else -1
          (transtion_area_dotp, contrast_angle) = self.chrom_db.transitionOrderConsistency(chrom)
          if fn not in grouped_chrom_dict[mol]:
            (_, heavy_ints) = np.hsplit(chrom[9], 2)
            heavy_max_ints = np.max(heavy_ints, axis=0)
            transition_idx = np.flip(np.argsort(heavy_max_ints)[-self.top_n_transitions:])
            for trans_idx in transition_idx:
              self.transition_quality[mol].append([mol, fn] + ([group] if self._hasGroup else [])  +  [transitions[trans_idx]] +  ([None]*54))
            self.sample_quality[mol].append([mol, fn] + ([group] if self._hasGroup else []) + [None]*19)
            continue
          chrom_idx += 1
          target_trans_end_idx = target_trans_idx + len(transitions)
          (peak_start, peak_end) = chrom[3]
          peak_width = (peak_end - peak_start)
          peakstart_list.append(peak_start)
          peakend_list.append(peak_end)
          peakwidth_list.append(peak_width)
          peak_start_density = self.peak_location_kde[mol]['start'](peak_start)[0]/self.peak_location_kde[mol]['start_max'] if mol in self.peak_location_kde else None
          peak_end_density = self.peak_location_kde[mol]['end'](peak_end)[0]/self.peak_location_kde[mol]['end_max'] if mol in self.peak_location_kde else None
          try:
            peak_width_density = self.peak_location_kde[mol]['width'](peak_width)[0]/self.peak_location_kde[mol]['width_max'] if mol in self.peak_location_kde else None
          except:
            peak_width_density = self.peak_location_kde[mol]['width'](peak_width)[0]/self.peak_location_kde[mol]['width_max'] if mol in self.peak_location_kde else None
          peak_loc_mean = np.sqrt(peak_start_density * peak_end_density) if peak_start_density is not None and peak_end_density is not None else -1 #Geomatrix mean
          peak_start_density = peak_start_density if peak_start_density is not None else -1
          peak_end_density = peak_end_density if peak_end_density is not None else -1
          peak_width_density = peak_width_density if peak_width_density is not None else -1
          peakstart_density_list.append(peak_start_density)
          peakend_density_list.append(peak_end_density)
          peakwidth_density_list.append(peak_width_density)
          peakloc_list.append(peak_loc_mean)
          q1_score_idx_list = np.flip(np.argsort(type1_quality[chrom_idx]))
          q1_score_top_n_list = q1_score_idx_list[:self.top_n_transitions]
          q1_top_n_transitions = transitions[q1_score_top_n_list]
          self._rankFragments(q1_frag_ranks, q1_top_n_transitions)
          q1_score_median = np.median(type1_quality[chrom_idx][q1_score_top_n_list])
          q3_score_idx_list = np.flip(np.argsort(type3_quality[target_trans_idx:target_trans_end_idx]))
          q3_score_top_n_list = q3_score_idx_list[:self.top_n_transitions]
          q3_top_n_transitions = transitions[q3_score_top_n_list]
          self._rankFragments(q3_frag_ranks, q3_top_n_transitions)
          q3_score_median = np.median(type3_quality[target_trans_idx:target_trans_end_idx][q3_score_top_n_list])
          q1_scores.append(q1_score_median)
          q2_scores.append(type2_quality[chrom_idx])
          q3_scores.append(q3_score_median)
          with np.errstate(divide='ignore', invalid='ignore'):
            # top_n_frag_idx_list = self.peak_qc.top_n_frag_idx_list(chrom)
            (light_peak_area, heavy_peak_area) = np.hsplit(chrom[7], 2)
            light_peak_area = np.where(light_peak_area > 0, np.nan_to_num(np.log10(light_peak_area)), 0)
            light_peakarea_list.append(np.max(light_peak_area))
            # (light_peakarea_transitions, light_peakarea_frag_order) = self.rankFragments(light_peakarea_ranks, transitions, light_peak_area, self.top_n_transitions)
            heavy_peak_area =  np.where(heavy_peak_area > 0, np.nan_to_num(np.log10(heavy_peak_area)), 0)
            heavy_peakarea_list.append(np.max(heavy_peak_area))
            (heavy_peakarea_transitions, heavy_peakarea_frag_order) = self.rankFragments(heavy_peakarea_ranks, transitions, heavy_peak_area, self.top_n_transitions)
            peakarea_log10_ratio = np.where(light_peak_area > 0, light_peak_area - heavy_peak_area, 0)
            (light_peak_ints, heavy_peak_ints) = np.hsplit(chrom[4], 2)
            if len(light_peak_ints) == 0:
              light_peak_ints = np.array(np.zeros(len(transitions)))
              light_peak_top = np.array(np.zeros(len(transitions)))
            else:
              light_peak_top = np.max(light_peak_ints, axis=0) if len(light_peak_ints) > 0 else 0
              light_peak_top = np.where(light_peak_top > 0, np.nan_to_num(np.log10(light_peak_top)), 0)
            light_peaktop_list.append(np.max(light_peak_top))
            # (light_peaktop_transitions, light_peaktop_frag_order) = self.rankFragments(light_peaktop_ranks, transitions, light_peak_top, self.top_n_transitions)
            if len(heavy_peak_ints) == 0:
                heavy_peak_ints = np.array(np.zeros(len(transitions)))
                heavy_peak_top = np.array(np.zeros(len(transitions)))
            else:
                heavy_peak_top = np.max(heavy_peak_ints, axis=0) if len(heavy_peak_ints) > 0 else 0
                heavy_peak_top = np.where(heavy_peak_top > 0, np.nan_to_num(np.log10(heavy_peak_top)), 0)
            heavy_peaktop_list.append(np.max(heavy_peak_top))
            (heavy_peaktop_transitions, heavy_peaktop_frag_order) = self.rankFragments(heavy_peaktop_ranks, transitions, heavy_peak_top, self.top_n_transitions)
            peaktop_log10_ratio = np.where(light_peak_top > 0, light_peak_top - heavy_peak_top, 0)
            if quantifier and quantifier_idx >= 0:
              chrom_peaktop_ratio = np.power(10, peaktop_log10_ratio[quantifier_idx]) if peaktop_log10_ratio[quantifier_idx] != 0 else 0
              chrom_peakarea_ratio = np.power(10, peakarea_log10_ratio[quantifier_idx]) if peakarea_log10_ratio[quantifier_idx] != 0 else 0
            else:
              if quantifier and quantifier_idx < 0:
                print(f"{mol}'s quantifier ion is set to {quantifier} but cannot be found in transition list: {transitions}. Using the transition ions ({heavy_peaktop_frag_order[0]},{heavy_peakarea_frag_order[0]}) with the highest peak height/peakarea, respectively.")
              try:
                chrom_peaktop_ratio = np.power(10, peaktop_log10_ratio[heavy_peaktop_frag_order[0]]) if heavy_peaktop_frag_order[0] >= 0 and peaktop_log10_ratio[heavy_peaktop_frag_order[0]] != 0 else 0
                chrom_peakarea_ratio = np.power(10, peakarea_log10_ratio[heavy_peakarea_frag_order[0]]) if heavy_peakarea_frag_order[0] >= 0 and peakarea_log10_ratio[heavy_peakarea_frag_order[0]] != 0 else 0
              except Exception as e:
                print(chrom)
                raise e
            peaktop_ratio_list.append(chrom_peaktop_ratio)
            peakarea_ratio_list.append(chrom_peakarea_ratio)
          for trans_idx in q1_score_idx_list:
            self.transition_quality[mol].append([mol, fn] + ([group] if self._hasGroup else [])  + 
              [transitions[trans_idx], type1_quality[chrom_idx][trans_idx], type2_quality[chrom_idx], type3_quality[target_trans_idx:target_trans_end_idx][trans_idx]] + 
              [light_peak_top[trans_idx], heavy_peak_top[trans_idx], peaktop_log10_ratio[trans_idx], light_peak_area[trans_idx], heavy_peak_area[trans_idx], peakarea_log10_ratio[trans_idx]] +
              type1_feature[chrom_idx][trans_idx].tolist() + type2_feature[chrom_idx].tolist() + type3_feature[target_trans_idx:target_trans_end_idx][trans_idx].tolist())
          self.sample_quality[mol].append([mol, fn] + ([group] if self._hasGroup else []) + [ q1_score_median, type2_quality[chrom_idx], q3_score_median] +
              [(light_peak_top[quantifier_idx] if quantifier_idx is not None else light_peak_top[heavy_peaktop_frag_order[0]]),
               (heavy_peak_top[quantifier_idx] if quantifier_idx is not None else heavy_peak_top[heavy_peaktop_frag_order[0]]),
               chrom_peaktop_ratio,
               (light_peak_area[quantifier_idx] if quantifier_idx is not None else light_peak_area[heavy_peakarea_frag_order[0]]),
               (heavy_peak_area[quantifier_idx] if quantifier_idx is not None else heavy_peak_area[heavy_peakarea_frag_order[0]]),
               chrom_peakarea_ratio] + 
              [';'.join(q1_top_n_transitions), ';'.join(q3_top_n_transitions), 
               ';'.join(heavy_peaktop_transitions), ';'.join(heavy_peakarea_transitions)] + 
              [peak_start, peak_end, peak_start_density, peak_end_density, peak_loc_mean, peak_width_density] + 
              [transtion_area_dotp, contrast_angle, np.mean(pair_ratio_consistency[target_trans_idx:target_trans_end_idx])])
          target_trans_idx = target_trans_end_idx

        #GroupQuality
        ordered_q1_transitions = sorted(q1_frag_ranks, key=lambda x: q1_frag_ranks[x], reverse=True)
        ordered_q3_transitions = sorted(q3_frag_ranks, key=lambda x: q3_frag_ranks[x], reverse=True)
        # ordered_light_peaktop_trans = sorted(light_peaktop_ranks, key=lambda x: light_peaktop_ranks[x], reverse=True)
        ordered_heavy_peaktop_trans = sorted(heavy_peaktop_ranks, key=lambda x: heavy_peaktop_ranks[x], reverse=True)
        # ordered_light_peakarea_trans = sorted(light_peakarea_ranks, key=lambda x: light_peakarea_ranks[x], reverse=True)
        ordered_heavy_peakarea_trans = sorted(heavy_peakarea_ranks, key=lambda x: heavy_peakarea_ranks[x], reverse=True)
        light_peaktop_cv = np.nanstd(light_peaktop_list, ddof=1)/np.nanmean(light_peaktop_list)
        heavy_peaktop_cv = np.nanstd(heavy_peaktop_list, ddof=1)/np.nanmean(heavy_peaktop_list)
        light_peakarea_cv = np.nanstd(light_peakarea_list, ddof=1)/np.nanmean(light_peakarea_list)
        heavy_peakarea_cv = np.nanstd(heavy_peakarea_list, ddof=1)/np.nanmean(heavy_peakarea_list)
        with np.errstate(divide='ignore', invalid='ignore'):
          peaktop_ratio_cv = np.abs(np.nanstd(peaktop_ratio_list, ddof=1)/np.nanmean(peaktop_ratio_list))
          peakarea_ratio_cv = np.abs(np.nanstd(peakarea_ratio_list, ddof=1)/np.nanmean(peakarea_ratio_list))
          peakstart_cv = np.nanstd(peakstart_list, ddof=1)/np.nanmean(peakstart_list)
          peakend_cv = np.nanstd(peakend_list, ddof=1)/np.nanmean(peakend_list)
          peakwidth_cv = np.nanstd(peakwidth_list, ddof=1)/np.nanmean(peakwidth_list)
        self.group_quality[mol].append([mol, group, np.nanmean(q1_scores), np.nanmean(q2_scores), np.nanmean(q3_scores)] +  # TMSQE scoring
                                       [np.nanmean(light_peaktop_list), np.nanmean(light_peakarea_list), light_peaktop_cv, light_peakarea_cv ] +
                                       [np.nanmean(heavy_peaktop_list), np.nanmean(heavy_peakarea_list), heavy_peaktop_cv, heavy_peakarea_cv] +
                                       [np.nanmean(peaktop_ratio_list), np.nanmean(peakarea_ratio_list), peaktop_ratio_cv, peakarea_ratio_cv] +
                                       [peakstart_cv, peakend_cv, peakwidth_cv] + #Peak Location CV
                                       [np.nanmean(peakstart_density_list), np.nanmean(peakend_density_list), np.nanmean(peakloc_list), np.nanmean(peakwidth_density_list)] + #Peak Location
                                       [';'.join(ordered_q1_transitions), ';'.join(ordered_q3_transitions), 
                                        ';'.join(ordered_heavy_peaktop_trans), ';'.join(ordered_heavy_peakarea_trans)]
                                      )
    #PeakAreaCV, PeakTopCV, PeakStartCV, PeakEndCV, PeakWidthCV, PeakLocationMean
    self.sample_df = pd.concat([pd.DataFrame(self.sample_quality[mol]) for mol in self.sample_quality.keys()]).reset_index(drop=True)
    self.sample_df.columns = (['molecule', 'sample'] + (['group'] if self._hasGroup else []) + 
                    ['type1_score_median', 'type2_score', 'type3_score_median','light_peakheight', 'heavy_peakheight', 'peakheight_ratio', 'light_peakarea', 'heavy_peakarea', 'peakarea_ratio'] + 
                    ['type1_score_ions', 'type3_score_ions', 'heavy_peakheight_ions', 'heavy_peakarea_ions'] + 
                    ['peak_start', 'peak_end', 'peak_start_density_ratio', 'peak_end_density_ratio', 'peak_loc_geomean', 'peak_width_density_ratio', 'transition_area_dotp', 'normalized_contrast_angle', 'pair_ratio_consistency_mean'])
    self.group_df =  pd.concat([pd.DataFrame( self.group_quality[mol]) for mol in self.group_quality.keys()]).reset_index(drop=True)
    self.group_df.columns = (['molecule', 'group', 'type1_mean', 'type2_mean', 'type3_mean'] +
                             ['light_peak_height_mean', 'light_peakarea_mean', 'light_peakheight_CV', 'light_peakarea_CV'] +
                             ['heavy_peak_height_mean', 'heavy_peakarea_mean', 'heavy_peakheight_CV', 'heavy_peakarea_CV'] +
                             ['peak_height_ratio_mean', 'peakarea_ratio_mean', 'peak_height_ratio_CV', 'peakrea_ratio_CV'] + 
                             ['peakstart_CV', 'peakend_CV', 'peakwidth_CV'] +
                             ['peakstart_density', 'peakend_density', 'peaklocation_index', 'peakwidth_density'] +
                             ['type1_ordered_transitions', 'type3_ordered_transitions', 
                              'heavy_peaktop_orderd_transitions', 'heavy_peakarea_orderd_transitions']
                            )

  def _get_workbook_formats(self, workbook, options=None):
    return dict(
      type1_color_format = {
        'type': '2_color_scale',
        'min_value': 6.7 if options is None or 'type1_min' not in options else options['type1_min'],
        'max_value': 7.7 if options is None or 'type1_max' not in options else options['type1_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'red',
        'max_color': '#00FF00'
      },
      type2_color_format = {
        'type': '2_color_scale',
        'min_value': 5.0 if options is None or 'type2_min' not in options else options['type2_min'],
        'max_value': 7.2 if options is None or 'type2_max' not in options else options['type2_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'red',
        'max_color': '#00FF00'
      },
      type3_color_format = {
        'type': '2_color_scale',
        'min_value': 7.901 if options is None or 'type3_min' not in options else options['type3_min'],
        'max_value': 8.665 if options is None or 'type3_max' not in options else options['type3_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'red',
        'max_color': '#00FF00'
      },
      peaktop_color_format = {
        'type': '2_color_scale',
        'min_value': 2 if options is None or 'peaktop_min' not in options else options['peaktop_min'],
        'max_value': 4 if options is None or 'peaktop_max' not in options else options['peaktop_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'red',
        'max_color': 'white'
      },
      peakarea_color_format = {
        'type': '2_color_scale',
        'min_value': 2 if options is None or 'peakarea_min' not in options else options['peakarea_min'],
        'max_value': 4 if options is None or 'peakarea_max' not in options else options['peakarea_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'red',
        'max_color': 'white'
      },
      peakloc_color_format = {
        'type': '2_color_scale',
        'min_value': 0.2 if options is None or 'peakloc_min' not in options else options['peakloc_min'],
        'max_value': 0.5 if options is None or 'peakloc_max' not in options else options['peakloc_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'red',
        'max_color': 'white'
      },
      ratio_consistency_color_format = {
        'type': '2_color_scale',
        'min_value': 0.4 if options is None or 'ratio_consistency_min' not in options else options['ratio_consistency_min'],
        'max_value': 0.7 if options is None or 'ratio_consistency_max' not in options else options['ratio_consistency_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'white',
        'max_color': 'red'
      },
      cv_color_format = {
        'type': '2_color_scale',
        'min_value': 0.05 if options is None or 'peakloc_min' not in options else options['peakloc_min'],
        'max_value': 0.2 if options is None or 'peakloc_max' not in options else options['peakloc_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'white',
        'max_color': 'red'
      },
      density_color_format = {
        'type': '2_color_scale',
        'min_value': 0.2 if options is None or 'peakloc_min' not in options else options['peakloc_min'],
        'max_value': 0.5 if options is None or 'peakloc_max' not in options else options['peakloc_max'],
        'min_type': 'num',
        'max_type': 'num',
        'min_color': 'red',
        'max_color': 'white'
      },
      title_format = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': 'black'}),
      digit_format = workbook.add_format({'num_format': '0.000'}),
      digit2_format = workbook.add_format({'num_format': '0.00'}),
      text_center = workbook.add_format({'align': 'center'}),
      top_border = workbook.add_format({'top': 2, 'border_color': 'black'}),
      top_border_with_percent = workbook.add_format({'top': 2, 'border_color': 'black', 'num_format': '0.00%'}),
      left_thin_border = workbook.add_format({'left': 1, 'border_color': 'black'}),
      left_thin_border_with_percent = workbook.add_format({'left': 1, 'border_color': 'black', 'num_format': '0.00%'}),
      left_thick_border = workbook.add_format({'left': 5, 'border_color': 'black'}),
      left_thick_border_with_percent = workbook.add_format({'left': 5, 'border_color': 'black', 'num_format': '0.00%'}),
      thick_left_top_border = workbook.add_format({'left': 5, 'top': 2, 'border_color': 'black'}),
      thick_left_top_border_with_percent = workbook.add_format({'left': 5, 'top': 2, 'border_color': 'black', 'num_format': '0.00%'}),
      thin_left_top_border = workbook.add_format({'left': 1, 'top': 2, 'border_color': 'black'}),
      thin_left_top_border_with_percent = workbook.add_format({'left': 1, 'top': 2, 'border_color': 'black', 'num_format': '0.00%'}),
      good_format = workbook.add_format({'bg_color': '#00FF00'}),
      warn_format = workbook.add_format({'bg_color': '#ffc107'}),
      bad_format = workbook.add_format({'bg_color': 'red'}),
      cv_format = workbook.add_format({'num_format': '0.00%'})
    )
    
  def output_transition_quality(self):
    transition_output_dir = os.path.join(self.output_folder, 'transition_quality')
    os.makedirs(transition_output_dir, exist_ok =True)
    for mol in self.transition_quality.keys():
      prcessed_mol = re.sub(r'\/', '_', mol)
      transition_output = os.path.join(transition_output_dir, prcessed_mol + '.xlsx')
      with pd.ExcelWriter(transition_output, engine='xlsxwriter') as writer:
        transition_df = pd.DataFrame(self.transition_quality[mol])
        qv_titles = self.peak_qc.type1QualityTitle() + self.peak_qc.type2QualityTitle() + self.peak_qc.type3QualityTitle()
        transition_df.columns = (['molecule', 'sample'] + (['group'] if self._hasGroup else []) + 
                                 ['transition', 'type1_score', 'type2_score', 'type3_score', 'light_peak_height', 'heavy_peak_height', 'peaktop_log10ratio', 'light_peak_area', 'heavy_peak_area', 'peakarea_log10ratio'] +
                                 #['skyline_light_peak_height', 'skyline_heavy_peak_height', 'skyline_light_peak_area', 'skyline_heavy_peak_area'] +
                                 #['skyline_light_bg_height', 'skyline_heavy_bg_height', 'skyline_light_bg_area', 'skyline_heavy_bg_area'] +
                                 qv_titles)
        transition_df.to_excel(writer, sheet_name='Transition Quality', index=False)
        transition_df_row_num = transition_df.shape[0]
        workbook = writer.book
        workbook_formats = self._get_workbook_formats(workbook, options=None)
        type1_color_format = workbook_formats['type1_color_format']
        type2_color_format = workbook_formats['type2_color_format']
        type3_color_format = workbook_formats['type3_color_format']
        peaktop_color_format = workbook_formats['peaktop_color_format']
        peakarea_color_format = workbook_formats['peakarea_color_format']
        title_format = workbook_formats['title_format']
        digit_format = workbook_formats['digit_format']
        digit2_format = workbook_formats['digit2_format']
        text_center = workbook_formats['text_center']
        top_border = workbook_formats['top_border']
        left_thin_border = workbook_formats['left_thin_border']
        left_thick_border = workbook_formats['left_thick_border']
        thick_left_top_border = workbook_formats['thick_left_top_border']
        thin_left_top_border = workbook_formats['thin_left_top_border']
        good_format = workbook_formats['good_format']
        warn_format = workbook_formats['warn_format']
        bad_format = workbook_formats['bad_format']
        writer.sheets['Transition Quality'].set_column(1, 20, None, text_center)
        col_idx_shift = 1 if 'group' in transition_df else 0
        for idx, column in enumerate(transition_df):
          if idx >= 12 + col_idx_shift:
            writer.sheets['Transition Quality'].write(0, idx, column, title_format)
            writer.sheets['Transition Quality'].set_column(idx, idx, 3)
          elif idx >= 3 + col_idx_shift:
            writer.sheets['Transition Quality'].write(0, idx, column, title_format)
            writer.sheets['Transition Quality'].set_column(idx, idx, 8)
          else:
            writer.sheets['Transition Quality'].write(0, idx, column, title_format)
            column_len = max(transition_df[column].astype(str).map(len).max(), len(column))
            writer.sheets['Transition Quality'].set_column(idx, idx, column_len + 4)
        writer.sheets['Transition Quality'].set_column(3 + col_idx_shift, 3 + col_idx_shift, None, left_thick_border)
        writer.sheets['Transition Quality'].set_column(6 + col_idx_shift, 6 + col_idx_shift, None, left_thick_border)
        writer.sheets['Transition Quality'].set_column(8 + col_idx_shift, 8 + col_idx_shift, None, left_thin_border)
        writer.sheets['Transition Quality'].set_column(10 + col_idx_shift, 10 + col_idx_shift, None, left_thick_border)
        writer.sheets['Transition Quality'].set_column(12 + col_idx_shift, 12 + col_idx_shift, None, left_thick_border)
        
        writer.sheets['Transition Quality'].set_column(19 + col_idx_shift, 19 + col_idx_shift, 3, left_thin_border)
        writer.sheets['Transition Quality'].set_column(26 + col_idx_shift, 26 + col_idx_shift, 3, left_thin_border)
        writer.sheets['Transition Quality'].set_column(29 + col_idx_shift, 29 + col_idx_shift, 3, left_thick_border)
        writer.sheets['Transition Quality'].set_column(36 + col_idx_shift, 36 + col_idx_shift, 3, left_thin_border)
        writer.sheets['Transition Quality'].set_column(43 + col_idx_shift, 43 + col_idx_shift, 3, left_thin_border)
        writer.sheets['Transition Quality'].set_column(51 + col_idx_shift, 51 + col_idx_shift, 3, left_thick_border)
        writer.sheets['Transition Quality'].set_column(59 + col_idx_shift, 59 + col_idx_shift, 3, left_thick_border)
        temp_id = None
        for index, row in transition_df.iterrows():
          id = row['molecule'] + row['sample']
          if temp_id is None:
            temp_id = id
          elif temp_id != id:
            writer.sheets['Transition Quality'].set_row(index + 1, None, top_border)
            writer.sheets['Transition Quality'].conditional_format(index + 1, 3 + col_idx_shift, index+1, 3 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
            writer.sheets['Transition Quality'].conditional_format(index + 1, 6 + col_idx_shift, index+1, 6 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
            writer.sheets['Transition Quality'].conditional_format(index + 1, 8 + col_idx_shift, index+1, 8 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
            writer.sheets['Transition Quality'].conditional_format(index + 1, 10 + col_idx_shift, index+1, 10 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
            writer.sheets['Transition Quality'].conditional_format(index + 1, 12 + col_idx_shift, index+1, 12 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
            
            writer.sheets['Transition Quality'].conditional_format(index + 1, 19 + col_idx_shift, index+1, 19 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
            writer.sheets['Transition Quality'].conditional_format(index + 1, 26 + col_idx_shift, index+1, 26 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
            writer.sheets['Transition Quality'].conditional_format(index + 1, 29 + col_idx_shift, index+1, 29 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
            writer.sheets['Transition Quality'].conditional_format(index + 1, 36 + col_idx_shift, index+1, 36 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
            writer.sheets['Transition Quality'].conditional_format(index + 1, 43 + col_idx_shift, index+1, 43 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
            writer.sheets['Transition Quality'].conditional_format(index + 1, 51 + col_idx_shift, index+1, 51 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
            writer.sheets['Transition Quality'].conditional_format(index + 1, 59 + col_idx_shift, index+1, 59 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
            temp_id = id
            # writer.sheets['Transition Quality'].set_row(index + 1, None, top_border)
        writer.sheets['Transition Quality'].set_row(index + 2, None, top_border)
        writer.sheets['Transition Quality'].conditional_format(index + 2, 3 + col_idx_shift, index+ 2, 3 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
        writer.sheets['Transition Quality'].conditional_format(index + 2, 6 + col_idx_shift, index+ 2, 6 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
        writer.sheets['Transition Quality'].conditional_format(index + 2, 8 + col_idx_shift, index+ 2, 8 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
        writer.sheets['Transition Quality'].conditional_format(index + 2, 10 + col_idx_shift, index+2, 10 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
        writer.sheets['Transition Quality'].conditional_format(index + 2, 12 + col_idx_shift, index+2, 12 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
        
        writer.sheets['Transition Quality'].conditional_format(index + 2, 17 + col_idx_shift, index+2, 17 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
        writer.sheets['Transition Quality'].conditional_format(index + 2, 24 + col_idx_shift, index+2, 24 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
        writer.sheets['Transition Quality'].conditional_format(index + 2, 27 + col_idx_shift, index+2, 27 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
        writer.sheets['Transition Quality'].conditional_format(index + 2, 34 + col_idx_shift, index+2, 34 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
        writer.sheets['Transition Quality'].conditional_format(index + 2, 41 + col_idx_shift, index+2, 41 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
        writer.sheets['Transition Quality'].conditional_format(index + 2, 49 + col_idx_shift, index+2, 49 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
        writer.sheets['Transition Quality'].conditional_format(index + 2, 57 + col_idx_shift, index+2, 57 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
        # writer.sheets['Transition Quality'].set_row(index + 2, None, top_border)
        writer.sheets['Transition Quality'].freeze_panes(1, 0)
        if self._hasGroup:
          type1_col_t = 'E'
          type2_col_t = 'F'
          type3_col_t = 'G'
          qv_start_col = 13
        else:
          type1_col_t = 'D'
          type2_col_t = 'E'
          type3_col_t = 'F'
          qv_start_col = 12
        writer.sheets['Transition Quality'].conditional_format(type1_col_t + '2:' + type1_col_t + str(transition_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
        # writer.sheets['Transition Quality'].conditional_format(type1_col_t + '2:' + type1_col_t + str(transition_df_row_num + 1), type1_color_format)
        writer.sheets['Transition Quality'].conditional_format(type1_col_t + '2:' + type1_col_t + str(transition_df_row_num + 1), 
                                                               {'type': 'cell', 'criteria': 'between', 'minimum': type1_color_format['min_value'], 'maximum': type1_color_format['max_value'], 'format': warn_format})
        writer.sheets['Transition Quality'].conditional_format(type1_col_t + '2:' + type1_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type1_color_format['max_value'], 'format': good_format})
        writer.sheets['Transition Quality'].conditional_format(type1_col_t + '2:' + type1_col_t + str(transition_df_row_num + 1), 
                                                               {'type': 'cell', 'criteria': 'less than', 'value': type1_color_format['min_value'], 'format': bad_format})
        writer.sheets['Transition Quality'].conditional_format(type2_col_t + '2:' + type2_col_t + str(transition_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
        # writer.sheets['Transition Quality'].conditional_format(type2_col_t + '2:' + type2_col_t + str(transition_df_row_num + 1), type2_color_format)
        writer.sheets['Transition Quality'].conditional_format(type2_col_t + '2:' + type2_col_t + str(transition_df_row_num + 1), 
                                                               {'type': 'cell', 'criteria': 'between', 'minimum': type2_color_format['min_value'], 'maximum': type2_color_format['max_value'], 'format': warn_format})
        writer.sheets['Transition Quality'].conditional_format(type2_col_t + '2:' + type2_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type2_color_format['max_value'], 'format': good_format})
        writer.sheets['Transition Quality'].conditional_format(type2_col_t + '2:' + type2_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type2_color_format['min_value'], 'format': bad_format})
        writer.sheets['Transition Quality'].conditional_format(type3_col_t + '2:' + type3_col_t + str(transition_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
        #writer.sheets['Transition Quality'].conditional_format(type3_col_t + '2:' + type3_col_t + str(transition_df_row_num + 1), type3_color_format)
        writer.sheets['Transition Quality'].conditional_format(type3_col_t + '2:' + type3_col_t + str(transition_df_row_num + 1), 
                                                               {'type': 'cell', 'criteria': 'between', 'minimum': type3_color_format['min_value'], 'maximum': type3_color_format['max_value'], 'format': warn_format})
        writer.sheets['Transition Quality'].conditional_format(type3_col_t + '2:' + type3_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type3_color_format['max_value'], 'format': good_format})
        writer.sheets['Transition Quality'].conditional_format(type3_col_t + '2:' + type3_col_t + str(transition_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type3_color_format['min_value'], 'format': bad_format})
        writer.sheets['Transition Quality'].conditional_format(1, qv_start_col - 6, transition_df_row_num + 1, qv_start_col - 6, peaktop_color_format)
        writer.sheets['Transition Quality'].conditional_format(1, qv_start_col - 5, transition_df_row_num + 1, qv_start_col - 5, peaktop_color_format)
        writer.sheets['Transition Quality'].conditional_format(1, qv_start_col - 4, transition_df_row_num + 1, qv_start_col - 4, peakarea_color_format)
        writer.sheets['Transition Quality'].conditional_format(1, qv_start_col - 3, transition_df_row_num + 1, qv_start_col - 3, peakarea_color_format)
        for i in range(len(qv_titles)):
          col_idx = qv_start_col + i
          qv_title = qv_titles[i]
          quality_range = self.quality_warn_range_config.get(qv_title)
          writer.sheets['Transition Quality'].conditional_format(1, col_idx, transition_df_row_num + 1, col_idx, {
            'type': '2_color_scale',
            'min_value': quality_range.get('min'),
            'max_value': quality_range.get('max'),
            'min_type': 'num',
            'max_type': 'num',
            'min_color': '#FF0000' if quality_range.get('direction') > 0 else '#00FF00',
            'max_color': '#00FF00' if quality_range.get('direction') > 0 else '#FF0000'
          })
    
  def output_sample_quality(self):
    output_path = os.path.join(self.output_folder, 'sample_quality.xlsx')
    sample_df = self.sample_df
    group_df = self.group_df
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
      sample_df.to_excel(writer, sheet_name='Sample Quality', index=False)
      sample_df_row_num = sample_df.shape[0]
      # if self._hasGroup:
      group_df.to_excel(writer, sheet_name="Group Quality", index=False)
      workbook = writer.book
      workbook_formats = self._get_workbook_formats(workbook, options=None)
      type1_color_format = workbook_formats['type1_color_format']
      type2_color_format = workbook_formats['type2_color_format']
      type3_color_format = workbook_formats['type3_color_format']
      peaktop_color_format = workbook_formats['peaktop_color_format']
      peakarea_color_format = workbook_formats['peakarea_color_format']
      peakloc_color_format = workbook_formats['peakloc_color_format']
      ratio_consistency_color_format = workbook_formats['ratio_consistency_color_format']
      cv_color_format = workbook_formats['cv_color_format']
      density_color_format = workbook_formats['density_color_format']
      title_format = workbook_formats['title_format']
      digit_format = workbook_formats['digit_format']
      digit2_format = workbook_formats['digit2_format']
      text_center = workbook_formats['text_center']
      top_border = workbook_formats['top_border']
      top_border_with_percent = workbook_formats['top_border_with_percent']
      left_thin_border = workbook_formats['left_thin_border']
      left_thin_border_with_percent = workbook_formats['left_thin_border_with_percent']
      left_thick_border = workbook_formats['left_thick_border']
      left_thick_border_with_percent = workbook_formats['left_thick_border_with_percent']
      thick_left_top_border = workbook_formats['thick_left_top_border']
      thick_left_top_border_with_percent = workbook_formats['thick_left_top_border_with_percent']
      thin_left_top_border = workbook_formats['thin_left_top_border']
      thin_left_top_border_with_percent = workbook_formats['thin_left_top_border_with_percent']
      good_format = workbook_formats['good_format']
      warn_format = workbook_formats['warn_format']
      bad_format = workbook_formats['bad_format']
      cv_format = workbook_formats['cv_format']
      writer.sheets['Sample Quality'].set_column(1, 20, None, text_center)
      col_idx_shift = 1 if self._hasGroup else 0
      for idx, column in enumerate(sample_df):
        writer.sheets['Sample Quality'].write(0, idx, column, title_format)
        if idx <= 2 + col_idx_shift:
          column_len = max(sample_df[column].astype(str).map(len).max(), len(column))
          writer.sheets['Sample Quality'].set_column(idx, idx, column_len + 8)
        else:
          writer.sheets['Sample Quality'].set_column(idx, idx, 10)
      writer.sheets['Sample Quality'].set_column(2 + col_idx_shift, 2 + col_idx_shift, None, left_thick_border)
      writer.sheets['Sample Quality'].set_column(5 + col_idx_shift, 5 + col_idx_shift, None, left_thick_border)
      writer.sheets['Sample Quality'].set_column(8 + col_idx_shift, 8 + col_idx_shift, None, left_thin_border)
      
      writer.sheets['Sample Quality'].set_column(11 + col_idx_shift, 11 + col_idx_shift, None, left_thick_border)
      writer.sheets['Sample Quality'].set_column(13 + col_idx_shift, 13 + col_idx_shift, None, left_thin_border)
      writer.sheets['Sample Quality'].set_column(15 + col_idx_shift, 15 + col_idx_shift, None, left_thick_border)
      writer.sheets['Sample Quality'].set_column(17 + col_idx_shift, 17 + col_idx_shift, None, left_thick_border)
      writer.sheets['Sample Quality'].set_column(19 + col_idx_shift, 19 + col_idx_shift, None, left_thin_border)
      writer.sheets['Sample Quality'].set_column(21 + col_idx_shift, 21 + col_idx_shift, None, left_thick_border)
      writer.sheets['Sample Quality'].set_column(23 + col_idx_shift, 23 + col_idx_shift, None, left_thin_border)
      writer.sheets['Sample Quality'].set_column(24 + col_idx_shift, 24 + col_idx_shift, None, left_thick_border)
      temp_id = None
      for index, row in sample_df.iterrows():
        if 'group' in row:
          id = f"{row['molecule']}_{row['group']}"
        else:
          id = row['molecule']
        if temp_id is None:
          temp_id = id
        elif temp_id != id:
          writer.sheets['Sample Quality'].set_row(index + 1, None, top_border)
          writer.sheets['Sample Quality'].conditional_format(index + 1, 2 + col_idx_shift, index+1, 2 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 5 + col_idx_shift, index+1, 5 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 8 + col_idx_shift, index+1, 8 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 11 + col_idx_shift, index+1, 11 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 13 + col_idx_shift, index+1, 13 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 15 + col_idx_shift, index+1, 15 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 17 + col_idx_shift, index+1, 17 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 19 + col_idx_shift, index+1, 19 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 21 + col_idx_shift, index+1, 21 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 23 + col_idx_shift, index+1, 23 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
          writer.sheets['Sample Quality'].conditional_format(index + 1, 24 + col_idx_shift, index+1, 24 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
          temp_id = id
      writer.sheets['Sample Quality'].set_row(index + 2, None, top_border)
      writer.sheets['Sample Quality'].conditional_format(index + 2,  2 + col_idx_shift, index+2,  2 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2,  5 + col_idx_shift, index+2,  5 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2,  8 + col_idx_shift, index+2,  8 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2, 11 + col_idx_shift, index+2, 11 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2, 13 + col_idx_shift, index+2, 13 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2, 15 + col_idx_shift, index+2, 15 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2, 17 + col_idx_shift, index+2, 17 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2, 19 + col_idx_shift, index+2, 19 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2, 21 + col_idx_shift, index+2, 21 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2, 23 + col_idx_shift, index+2, 23 + col_idx_shift, {'type': 'no_errors', 'format': thin_left_top_border})
      writer.sheets['Sample Quality'].conditional_format(index + 2, 24 + col_idx_shift, index+2, 24 + col_idx_shift, {'type': 'no_errors', 'format': thick_left_top_border})
      writer.sheets['Sample Quality'].freeze_panes(1, 0)
      if True:
        # writer.sheets['Group Quality'].set_column(1, 20, None, text_center)
        # writer.sheets['Group Quality'].write(0, 0, 'Molecule', title_format)
        # writer.sheets['Group Quality'].write(0, 1, 'Group', title_format)
        for idx, column in enumerate(group_df):
          writer.sheets['Group Quality'].write(0, idx, column, title_format)
          if idx < 2:
            column_len = max(group_df[column].astype(str).map(len).max(), len(column))
            writer.sheets['Group Quality'].set_column(idx, idx, column_len + 8)
          elif idx <= 23:
            writer.sheets['Group Quality'].set_column(idx, idx, 10)
          else:
            writer.sheets['Group Quality'].set_column(idx, idx, 28)
        #writer.sheets['Group Quality'].set_column(0, 0, max(list(map(lambda x: len(x[0]), group_df.index)))) 這行應該是針對原本DataFrame.groupby做的處理
        writer.sheets['Group Quality'].set_column(2,  2, None, left_thick_border)
        writer.sheets['Group Quality'].set_column(5,  5, None, left_thick_border)
        writer.sheets['Group Quality'].set_column(7,  7, None, left_thin_border_with_percent)
        writer.sheets['Group Quality'].set_column(9,  9, None, left_thick_border)
        writer.sheets['Group Quality'].set_column(11,  11, None, left_thin_border_with_percent)
        writer.sheets['Group Quality'].set_column(13,  13, None, left_thick_border)
        writer.sheets['Group Quality'].set_column(15,  15, None, left_thick_border_with_percent)
        writer.sheets['Group Quality'].set_column(17,  17, None, left_thick_border_with_percent)
        writer.sheets['Group Quality'].set_column(20,  20, None, left_thick_border)
        writer.sheets['Group Quality'].set_column(24,  24, 28, left_thick_border)
        writer.sheets['Group Quality'].set_column(28,  28, None, left_thick_border)
        writer.sheets['Group Quality'].set_column(8,  8, None, cv_format)
        writer.sheets['Group Quality'].set_column(12,  12, None, cv_format)
        writer.sheets['Group Quality'].set_column(16,  16, None, cv_format)
        writer.sheets['Group Quality'].set_column(18,  18, None, cv_format)
        writer.sheets['Group Quality'].set_column(19,  19, None, cv_format)
        
        temp_id = None
        row_idx = 0
        for index, row in group_df.iterrows():
          row_idx += 1
          id = f"{row['molecule']}"
          if temp_id is None:
            temp_id = id
          elif temp_id != id:
            writer.sheets['Group Quality'].set_row(row_idx, 14, top_border)
            writer.sheets['Group Quality'].conditional_format(row_idx,  2,  row_idx,  2, {'type': 'no_errors', 'format': thick_left_top_border})
            writer.sheets['Group Quality'].conditional_format(row_idx,  5,  row_idx,  5, {'type': 'no_errors', 'format': thick_left_top_border})
            writer.sheets['Group Quality'].conditional_format(row_idx,  7,  row_idx,  7, {'type': 'no_errors', 'format': thin_left_top_border_with_percent})
            writer.sheets['Group Quality'].conditional_format(row_idx,  9,  row_idx,  9,  {'type': 'no_errors', 'format': thick_left_top_border})
            writer.sheets['Group Quality'].conditional_format(row_idx,  11, row_idx,  11, {'type': 'no_errors', 'format': thin_left_top_border_with_percent})
            writer.sheets['Group Quality'].conditional_format(row_idx,  13, row_idx,  13, {'type': 'no_errors', 'format': thick_left_top_border})
            writer.sheets['Group Quality'].conditional_format(row_idx,  15, row_idx,  15, {'type': 'no_errors', 'format': thick_left_top_border_with_percent})
            writer.sheets['Group Quality'].conditional_format(row_idx,  17, row_idx,  17, {'type': 'no_errors', 'format': thick_left_top_border_with_percent})
            
            writer.sheets['Group Quality'].conditional_format(row_idx,  20, row_idx,  20, {'type': 'no_errors', 'format': thick_left_top_border})
            writer.sheets['Group Quality'].conditional_format(row_idx,  24, row_idx,  24, {'type': 'no_errors', 'format': thick_left_top_border})
            writer.sheets['Group Quality'].conditional_format(row_idx,  28, row_idx,  28, {'type': 'no_errors', 'format': thick_left_top_border})
            writer.sheets['Group Quality'].conditional_format(row_idx, 8, row_idx, 8, {'type': 'no_errors', 'format': top_border_with_percent})
            writer.sheets['Group Quality'].conditional_format(row_idx, 12, row_idx, 12, {'type': 'no_errors', 'format': top_border_with_percent})
            writer.sheets['Group Quality'].conditional_format(row_idx, 16, row_idx, 16, {'type': 'no_errors', 'format': top_border_with_percent})
            writer.sheets['Group Quality'].conditional_format(row_idx, 18, row_idx, 18, {'type': 'no_errors', 'format': top_border_with_percent})
            writer.sheets['Group Quality'].conditional_format(row_idx, 19, row_idx, 19, {'type': 'no_errors', 'format': top_border_with_percent})
            temp_id = id
        writer.sheets['Group Quality'].set_row(row_idx + 1, 14, top_border)
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  2,  row_idx+1,  2,  {'type': 'no_errors', 'format': thick_left_top_border})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  5,  row_idx+1,  5, {'type': 'no_errors', 'format': thick_left_top_border})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  7,  row_idx+1,  7, {'type': 'no_errors', 'format': thin_left_top_border_with_percent})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  9,  row_idx+1,  9,  {'type': 'no_errors', 'format': thick_left_top_border})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  11, row_idx+1,  11, {'type': 'no_errors', 'format': thin_left_top_border_with_percent})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  13, row_idx+1,  13, {'type': 'no_errors', 'format': thick_left_top_border})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  15, row_idx+1,  15, {'type': 'no_errors', 'format': thick_left_top_border_with_percent})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  17, row_idx+1,  17, {'type': 'no_errors', 'format': thick_left_top_border_with_percent})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  20, row_idx+1,  20, {'type': 'no_errors', 'format': thick_left_top_border})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  24, row_idx+1,  24, {'type': 'no_errors', 'format': thick_left_top_border})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  28, row_idx+1,  28, {'type': 'no_errors', 'format': thick_left_top_border})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  8,  row_idx+1,  8, {'type': 'no_errors', 'format': top_border_with_percent})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  12, row_idx+1,  12, {'type': 'no_errors', 'format': top_border_with_percent})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  16, row_idx+1,  16, {'type': 'no_errors', 'format': top_border_with_percent})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  18, row_idx+1,  18, {'type': 'no_errors', 'format': top_border_with_percent})
        writer.sheets['Group Quality'].conditional_format(row_idx+1,  19, row_idx+1,  19, {'type': 'no_errors', 'format': top_border_with_percent})
        writer.sheets['Group Quality'].freeze_panes(1, 0)
      if self._hasGroup:
        type1_col = 'D'
        type2_col = 'E'
        type3_col = 'F'
        qv_start_col_sample = 6
      else: 
        type1_col = 'C'
        type2_col = 'D'
        type3_col = 'E'
        qv_start_col_sample = 5
      writer.sheets['Sample Quality'].conditional_format(type1_col + '2:' + type1_col + str(sample_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      #writer.sheets['Sample Quality'].conditional_format(type1_col + '2:' + type1_col + str(sample_df_row_num + 1), type1_color_format)
      writer.sheets['Sample Quality'].conditional_format(type1_col + '2:' + type1_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type1_color_format['min_value'], 'maximum': type1_color_format['max_value'], 'format': warn_format})
      writer.sheets['Sample Quality'].conditional_format(type1_col + '2:' + type1_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type1_color_format['max_value'], 'format': good_format})
      writer.sheets['Sample Quality'].conditional_format(type1_col + '2:' + type1_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type1_color_format['min_value'], 'format': bad_format})
      
      writer.sheets['Sample Quality'].conditional_format(type2_col + '2:' + type2_col + str(sample_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      #writer.sheets['Sample Quality'].conditional_format(type2_col + '2:' + type2_col + str(sample_df_row_num + 1), type2_color_format)
      writer.sheets['Sample Quality'].conditional_format(type2_col + '2:' + type2_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type2_color_format['min_value'], 'maximum': type2_color_format['max_value'], 'format': warn_format})
      writer.sheets['Sample Quality'].conditional_format(type2_col + '2:' + type2_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type2_color_format['max_value'], 'format': good_format})
      writer.sheets['Sample Quality'].conditional_format(type2_col + '2:' + type2_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type2_color_format['min_value'], 'format': bad_format})
      
      writer.sheets['Sample Quality'].conditional_format(type3_col + '2:' + type3_col + str(sample_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      #writer.sheets['Sample Quality'].conditional_format(type3_col + '2:' + type3_col + str(sample_df_row_num + 1), type3_color_format)
      writer.sheets['Sample Quality'].conditional_format(type3_col + '2:' + type3_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type3_color_format['min_value'], 'maximum': type3_color_format['max_value'], 'format': warn_format})
      writer.sheets['Sample Quality'].conditional_format(type3_col + '2:' + type3_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type3_color_format['max_value'], 'format': good_format})
      writer.sheets['Sample Quality'].conditional_format(type3_col + '2:' + type3_col + str(sample_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type3_color_format['min_value'], 'format': bad_format})
      
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 0, sample_df_row_num + 1, qv_start_col_sample + 5, {'type': 'no_errors', 'format': digit_format})
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 0, sample_df_row_num + 1, qv_start_col_sample + 0, peaktop_color_format)
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 1, sample_df_row_num + 1, qv_start_col_sample + 1, peaktop_color_format)
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 3, sample_df_row_num + 1, qv_start_col_sample + 3, peakarea_color_format)
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 4, sample_df_row_num + 1, qv_start_col_sample + 4, peakarea_color_format)

      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 12, sample_df_row_num + 1, qv_start_col_sample + 18, {'type': 'no_errors', 'format': digit2_format})
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 12, sample_df_row_num + 1, qv_start_col_sample + 12, peakloc_color_format)
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 13, sample_df_row_num + 1, qv_start_col_sample + 13, peakloc_color_format)
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 14, sample_df_row_num + 1, qv_start_col_sample + 14, peakloc_color_format)
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 15, sample_df_row_num + 1, qv_start_col_sample + 15, peakloc_color_format)

      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 16, sample_df_row_num + 1, qv_start_col_sample + 16, peakloc_color_format)
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 17, sample_df_row_num + 1, qv_start_col_sample + 17, peakloc_color_format)
      writer.sheets['Sample Quality'].conditional_format(1, qv_start_col_sample + 18, sample_df_row_num + 1, qv_start_col_sample + 18, ratio_consistency_color_format)

      grouped_df_row_num = group_df.shape[0]
      writer.sheets['Group Quality'].conditional_format('C2:G' + str(grouped_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      #writer.sheets['Group Quality'].conditional_format('C2:C' + str(grouped_df_row_num + 1), type1_color_format)
      writer.sheets['Group Quality'].conditional_format('C2:C' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type1_color_format['max_value'], 'format': good_format})
      writer.sheets['Group Quality'].conditional_format('C2:C' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type1_color_format['min_value'], 'format': bad_format})
      writer.sheets['Group Quality'].conditional_format('C2:C' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type1_color_format['min_value'], 'maximum': type1_color_format['max_value'], 'format': warn_format})
      #writer.sheets['Group Quality'].conditional_format('D2:D' + str(grouped_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      #writer.sheets['Group Quality'].conditional_format('D2:D' + str(grouped_df_row_num + 1), type2_color_format)
      writer.sheets['Group Quality'].conditional_format('D2:D' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type2_color_format['max_value'], 'format': good_format})
      writer.sheets['Group Quality'].conditional_format('D2:D' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type2_color_format['min_value'], 'format': bad_format})
      writer.sheets['Group Quality'].conditional_format('D2:D' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type2_color_format['min_value'], 'maximum': type2_color_format['max_value'], 'format': warn_format})
      #writer.sheets['Group Quality'].conditional_format('E2:E' + str(grouped_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      #writer.sheets['Group Quality'].conditional_format('E2:E' + str(grouped_df_row_num + 1), type3_color_format)
      writer.sheets['Group Quality'].conditional_format('E2:E' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'greater than or equal to', 'value': type3_color_format['max_value'], 'format': good_format})
      writer.sheets['Group Quality'].conditional_format('E2:E' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'less than', 'value': type3_color_format['min_value'], 'format': bad_format})
      writer.sheets['Group Quality'].conditional_format('E2:E' + str(grouped_df_row_num + 1), {'type': 'cell', 'criteria': 'between', 'minimum': type3_color_format['min_value'], 'maximum': type3_color_format['max_value'], 'format': warn_format})
      writer.sheets['Group Quality'].conditional_format('F2:G' + str(grouped_df_row_num + 1), peaktop_color_format)
      writer.sheets['Group Quality'].conditional_format('H2:I' + str(grouped_df_row_num + 1), cv_color_format)
      writer.sheets['Group Quality'].conditional_format('J2:K' + str(grouped_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      writer.sheets['Group Quality'].conditional_format('J2:K' + str(grouped_df_row_num + 1), peakarea_color_format)
      writer.sheets['Group Quality'].conditional_format('N2:O' + str(grouped_df_row_num + 1), {'type': 'no_errors', 'format': digit_format})
      writer.sheets['Group Quality'].conditional_format('L2:M' + str(grouped_df_row_num + 1), cv_color_format)
      writer.sheets['Group Quality'].conditional_format('P2:T' + str(grouped_df_row_num + 1), cv_color_format)
      writer.sheets['Group Quality'].conditional_format('U2:X' + str(grouped_df_row_num + 1), {'type': 'no_errors', 'format': digit2_format})
      writer.sheets['Group Quality'].conditional_format('U2:X' + str(grouped_df_row_num + 1), density_color_format)
  
  def output_chrom_plot(self, ncol=6, nrow=6, dpi=200, mol_separation=True, reorder=True, figW=None, figH=None):
    output_file = os.path.join(self.output_folder, 'chromatogram_plots.pdf')
    print("Outputing chromatogram plots to " + output_file)
    plt.rcParams['figure.dpi'] = dpi
    if figW is None:
      figW = ncol * 7
    if figH is None:
      figH = nrow * 5
    plt.rcParams['figure.figsize'] = (figW, figH)
    plt.rcParams.update({'font.size': 10})
    chrom_list = []
    target_list = self.chrom_db.target_list
    file_list = self.chrom_db.filename_list
    page_size = ncol * nrow
    total_page = 1
    pdf = PdfPages(output_file)
    page_index = 1

    if reorder and self._hasGroup:
      for mol, target_chrom_list in groupby(self.chrom_list, key=lambda x: x[0]):
        #if self._hasGroup:
        sorted_by_group = sorted(target_chrom_list, key=lambda x: (x[12] is None, x[12]))
        #else:
        #  sorted_by_group = target_chrom_list
        group_iter = groupby(sorted_by_group, key=lambda x: (x[12] is None, x[12])) if self._hasGroup else enumerate([target_chrom_list])
        for idx, (group_key, each_group_chrom_list) in enumerate(group_iter):
          group = group_key[1]
          sorted_chrom_list = sorted(each_group_chrom_list, key=lambda x: (x[1], x[12] is None, x[12]))
          for chrom_data in sorted_chrom_list:
            chrom_list.append(chrom_data)
        if mol_separation:
          current_length = len(chrom_list)
          remainder = current_length % page_size
          if remainder > 0:
            chrom_list = chrom_list + ([None] * (page_size - remainder))
    else:
      for mol in target_list:
        for fn in file_list:
          if mol in self.chrom_dict and fn in self.chrom_dict[mol]:
            chrom_list.append(self.chrom_dict[mol][fn])
        if mol_separation:
          current_length = len(chrom_list)
          remainder = current_length % page_size
          if remainder > 0:
            chrom_list = chrom_list + ([None] * (page_size - remainder))
          
    # end for    
    total_page = math.floor(len(chrom_list) / page_size)
    if (len(self.chrom_list) % page_size) > 0:
        total_page += 1
    page_input_options = []
    print(f'Total page: {total_page}')
    for page_idx in range(total_page):
      page_start = page_idx * page_size
      page_end = page_start + page_size
      page_items = chrom_list[page_start:page_end]
      if len(page_items) > 0:
        page_input_options.append({'page_items': page_items, 'nrow': nrow, 'ncol': ncol, 'dpi': dpi, 'figW': figW, 'figH': figH, 'page_num': page_idx + 1, 'total_page': total_page,'mol_separation': mol_separation})
    page_num = 1
    for each_page_option in tqdm(page_input_options):
      fig = self._plot_each_page(each_page_option)
      pdf.savefig(fig, bbox_inches='tight')
      plt.clf()
      plt.close(fig)
      page_num += 1
    pdf.close()

  def _plot_each_page(self, options):
    page_items = options['page_items']
    nrow = options['nrow']
    ncol = options['ncol']
    plt.rcParams['figure.dpi'] = options['dpi']
    plt.rcParams['figure.figsize'] = (options['figW'], options['figH'])
    plt.rcParams.update({'font.size': 10})
    fig, axs = plt.subplots(nrow*2, ncol, layout="constrained")
    item_idx = 0
    if len(page_items) > 0 and page_items[0] is not None:
      if options['mol_separation'] :
        fig.suptitle(f"{page_items[0][0]} (Page {options['page_num']} of {options['total_page']})", fontsize=14, fontweight='bold', ha='left', x=0.01)
      else:
        fig.subtitle(f"Page {options['page_num']} of {options['total_page']})", fontsize=10, ha='right', x=0.98)
    for row_idx in range(nrow):
      for col_idx in range(ncol):
        if item_idx >= len(page_items) or page_items[item_idx] is None:
          axs[2*row_idx, col_idx].remove()
          axs[2*row_idx + 1, col_idx].remove()
          continue
        chrom_data = page_items[item_idx]
        (target, fn) = (chrom_data[0], chrom_data[1])
        if chrom_data[3] is not None:
          try:
            score_table = self.sample_df[(self.sample_df['molecule'] == target) & (self.sample_df['sample'] == fn)][['type1_score_median', 'type2_score', 'type3_score_median']].iloc[0]
          except:
            score_table = None
        else:
          score_table = None
        self._plot_each_chrom(chrom_data, score_table, axs[2*row_idx, col_idx], axs[2*row_idx + 1, col_idx], options = options, mol_separation = options['mol_separation'])
        item_idx += 1
    return fig

  def _plot_each_chrom(self, chrom_data, score_table, ax1, ax2, options=None, mol_separation=True):
    top_n = self.top_n_transitions
    (ps, fn) = (chrom_data[0], chrom_data[1])
    rt_time = chrom_data[8]
    transitions = chrom_data[2][0]
    has_boundary = True if chrom_data[3] is not None else False
    (peak_start, peak_end) = chrom_data[3] if has_boundary else (None, None)
    (light_intensity, heavy_intensity) = np.hsplit(chrom_data[9], 2)
    light_global_max_intensity = np.max(light_intensity)
    heavy_global_max_intensity = np.max(heavy_intensity)
    light_exp = len(str(int(light_global_max_intensity))) - 2
    heavy_exp = len(str(int(heavy_global_max_intensity))) - 2
    (type1_score, type2_score, type3_score) = (score_table['type1_score_median'], score_table['type2_score'], score_table['type3_score_median']) if score_table is not None else (None, None, None)
    if light_exp < 0 :
      light_exp = 0
    if heavy_exp < 0:
      heavy_exp = 0
    light_int = (light_intensity/(10**light_exp))
    heavy_int = (heavy_intensity/(10**heavy_exp))
    filtered_trans_idx_list = None
    if len(transitions) > top_n:
      if has_boundary:
        (light_peak_ints, heavy_peak_ints) = np.hsplit(chrom_data[4], 2)
        if heavy_peak_ints.shape[0] > 0:
          heavy_max_peak_ints = np.max(heavy_peak_ints, axis=0)
          filtered_trans_idx_list = np.flip(np.argsort(heavy_max_peak_ints)[-top_n:])
      else:
        heavy_max_ints = np.max(heavy_intensity, axis=0)
        filtered_trans_idx_list = np.flip(np.argsort(heavy_max_ints)[-top_n:])
    for idx, fragment in enumerate(chrom_data[2][1]):
      if filtered_trans_idx_list is not None:
        linewidth = 1 if idx in filtered_trans_idx_list else 0.5
      else:
        linewidth = 1
      ax1.plot(rt_time, light_int[:, idx],  linewidth=1, label=fragment)
    for idx, fragment in enumerate(chrom_data[2][2]):
      if filtered_trans_idx_list is not None:
        linewidth = 1 if idx in filtered_trans_idx_list else 0.5
      else:
        linewidth = 1
      ax2.plot(rt_time, heavy_int[:, idx],  linewidth=1, label=fragment)
    if not mol_separation:
      ax1.set_title(ps + '\n', fontsize=12, fontweight='bold')
    ax1.annotate(fn, xy=(0.5, 1.02), xycoords='axes fraction', ha='center', va='bottom', fontsize=10, fontweight='normal')
    ax1.set_ylabel('Intensity(10^%s)'%light_exp, fontsize=10)
    ax1.set_xlabel('Retention Time', fontsize = 10)
    if peak_start is not None:
      ax1.axvline(peak_start, linestyle= '--', linewidth=1,color='black')
      ax2.axvline(peak_start, linestyle= '--', linewidth=1,color='black')
    if peak_end is not None:
      ax1.axvline(peak_end, linestyle= '--', linewidth=1,color='black')  
      ax2.axvline(peak_end, linestyle= '--', linewidth=1,color='black')  
    if filtered_trans_idx_list is None:
      ax1.legend(fontsize=7, loc='best', frameon=False)
    else:
      h, l = ax1.get_legend_handles_labels()
      ax1.legend(fontsize=7, loc='best', frameon=False, handles=[h[frag_idx] for frag_idx in filtered_trans_idx_list], labels = [ l[frag_idx] for frag_idx in filtered_trans_idx_list])  
    ax2.set_ylabel('Intensity(10^%s)'%heavy_exp, fontsize = 10)
    ax2.xaxis.tick_top()
    if filtered_trans_idx_list is None:
      ax2.legend(fontsize=7,loc='best', frameon=False)
    else:
      h, l = ax2.get_legend_handles_labels()
      ax2.legend(fontsize=7, loc='best', frameon=False, handles=[h[frag_idx] for frag_idx in filtered_trans_idx_list], labels = [ l[frag_idx] for frag_idx in filtered_trans_idx_list])
    #Coloring
    good_color = '#198754'
    warn_color = '#cc9a06' 
    bad_color = '#FF0000'
    if type1_score is not None:
      if type1_score >= 7.7:
        type1_color = good_color
        type1_quality = 'Good'
      elif type1_score < 6.7:
        type1_color = bad_color
        type1_quality = 'Poor'
      else:
        type1_color = warn_color
        type1_quality = 'Acceptable'
    else:
      type1_color = 'black'
      type1_quality = 'N/A'
    
    if type2_score is not None:
      if type2_score >= 7.2:
        type2_color = good_color
        type2_quality = 'Good'
      elif type2_score < 5.0:
        type2_color = bad_color
        type2_quality = 'Poor'
      else:
        type2_color = warn_color
        type2_quality = 'Acceptable'
    else:
      type2_color = 'black'
      type2_quality = 'N/A'
  
    
    if type3_score is not None:
      if type3_score > 8.665:
        type3_color = good_color
        type3_quality = 'Good'
      elif type3_score < 7.901:
        type3_color = bad_color
        type3_quality = 'Poor'
      else:
        type3_color = warn_color
        type3_quality = 'Acceptable'
    else:
      type3_color = 'black'
      type3_quality = 'N/A'
    if type1_score is not None:
      ax2.annotate('Type 1: ' + str(round(type1_score, 2)) + '\n' + type1_quality + '\n\n', xy=(-0.05, -0.03), xycoords='axes fraction', ha='left', va='top', fontweight="bold", fontsize=10, color=type1_color)
    else:
      ax2.annotate('Type 1: N/A\n \n\n', xy=(-0.05, -0.03), xycoords='axes fraction', ha='left', va='top', fontweight="bold", fontsize=10, color=type1_color)
    if type2_score is not None:
      ax2.annotate('Type 2: ' + str(round(type2_score, 2)) + '\n' + type2_quality + '\n\n', xy=(0.5, -0.03), xycoords='axes fraction', ha='center', va='top', fontweight="bold", fontsize=10, color=type2_color)
    else:
      ax2.annotate('Type 2: N/A\n \n\n', xy=(0.5, -0.03), xycoords='axes fraction', ha='center', va='top', fontweight="bold", fontsize=10, color=type2_color)
    if type3_score is not None:
      ax2.annotate('Type 3: ' + str(round(type3_score, 2)) + '\n' + type3_quality + '\n\n', xy=(1.05, -0.03), xycoords='axes fraction', ha='right', va='top', fontweight="bold", fontsize=10, color=type3_color)
    else:
      ax2.annotate('Type 3: N/A\n \n\n', xy=(1.05, -0.03), xycoords='axes fraction', ha='right', va='top', fontweight="bold", fontsize=10, color=type3_color)
    if type2_score is not None:
      if type2_score >= 7.2:
          ax1.set_facecolor('mintcream')
          ax2.set_facecolor('mintcream')
      elif type2_score < 7.2 and type2_score >= 5.0:
          ax1.set_facecolor('cornsilk')
          ax2.set_facecolor('cornsilk')
      elif type2_score < 5.0:
          ax1.set_facecolor('mistyrose')
          ax2.set_facecolor('mistyrose')
    else:
      ax1.set_facecolor('lightgray')
      ax2.set_facecolor('lightgray')    

  def output_peak_location_distribution(self):
    output_file = os.path.join(self.output_folder, 'peak_location_distribution.pdf')
    print("Outputing peak location distributions to " + output_file)
    pdf = PdfPages(output_file)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = (16.0, 9.0)
    row_number = 3
    col_number = 4
    counter = 0
    pdf = PdfPages(output_file)
    chrom_db = self.chrom_db
    for mol in self.peak_location_kde.keys():
      if counter == 0:
        fig, ax = plt.subplots(row_number, col_number, layout="constrained")
      rt_list = chrom_db.chrom[chrom_db.chrom['PeptideModifiedSequence'] == mol].iloc[:, 8].str.split(',')
      rt_min = rt_list.apply(lambda x: float(min(x))).min()
      rt_max = rt_list.apply(lambda x: float(max(x))).max()
      x_range = np.linspace(rt_min, rt_max, 1000)
      row = int(counter/col_number)
      col = counter%col_number
      x1_pdf = self.peak_location_kde[mol]['start']
      x2_pdf = self.peak_location_kde[mol]['end']
      start_rt = self.peak_location_kde[mol]['start_rt']
      end_rt = self.peak_location_kde[mol]['end_rt']
      ax[row][col].plot(x_range, x1_pdf(x_range), label=f'Peak start ({start_rt})', alpha=0.8)
      ax[row][col].axvline(start_rt, linestyle= '--', linewidth=1, color='C0')
      ax[row][col].plot(x_range, x2_pdf(x_range), label=f'Peak end ({end_rt})',  alpha=0.8)
      ax[row][col].axvline(end_rt, linestyle= '--', linewidth=1, color='C1')
      ax[row][col].set_title(mol)
      ax[row][col].legend(fontsize=7)
      counter += 1
      if counter % (col_number*row_number) == 0:
        pdf.savefig(fig, bbox_inches='tight')
        plt.clf()
        plt.close(fig)
        counter = 0
    pdf.savefig(fig, bbox_inches='tight')
    plt.clf()
    plt.close(fig)
    pdf.close()
