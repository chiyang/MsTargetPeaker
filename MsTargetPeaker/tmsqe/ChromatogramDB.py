from time import time
from pandas import read_csv
from matplotlib import pyplot as plt
from numpy import (unique as np_unique, interp as np_interp, array as np_array, float32 as np_float32, linspace as np_linspace, zeros as np_zeros,
                  min as np_min, max as np_max,
                  apply_along_axis as np_apply_along_axis, sort as np_sort, argsort as np_argsort, isnan as np_isnan, trapz as np_trapz, hsplit as np_hsplit, sum as np_sum,
                  concatenate as np_concatenate, max as np_max, flip as np_flip, dot as np_dot, max as np_max, min as np_min, where as np_where)
from numpy.random import choice as np_random_choice
from numpy.linalg import norm
from math import (acos as math_acos, pi as math_pi)

class ChromatogramDB():
  def __init__(self,  chromatogramFile, boundaryFile=None, internal_standard_type='heavy', groupFile = None):
    start_time = time()
    self.timepoint = 1024
    self.chrom_path = chromatogramFile
    self.boundary_path = boundaryFile
    self.chrom = read_csv(chromatogramFile, sep='\t')
    self.chrom.columns = self.chrom.columns.str.replace(' ', '')
    self.chrom['IsotopeLabelType'] = self.chrom['IsotopeLabelType'].str.lower() # should be light | heavy
    chromatogram_data_load_time = time()
    print('Chromatogram data loaded in %.2f seconds' % (chromatogram_data_load_time - start_time))
    self.peak_boundary = None
    if boundaryFile is not None:
      try:
        self.peak_boundary = read_csv(boundaryFile)
        self.peak_boundary.columns = self.peak_boundary.columns.str.replace(' ', '')
      except:
        print(f'Error reading the boundary file: {boundaryFile}')
        self.peak_boundary = None  
    peak_boundary_load_time = time()
    print('Peak boundary data loaded in %.2f seconds' % (peak_boundary_load_time - chromatogram_data_load_time))
    self.internal_standard_type = 'light' if internal_standard_type == 'light' else 'heavy'
    self.filename_list = np_unique(self.chrom['FileName'].astype(str))
    self.target_list = np_unique(self.chrom['PeptideModifiedSequence'].astype(str))
    self.chromDB = dict()
    self.group_info = read_csv(groupFile) if groupFile is not None else None
    if self.group_info is not None:
      self.group_info.columns = self.group_info.columns.str.replace(' ', '')
      if 'Group' in self.group_info:
        self.group_types = self.group_info['Group'].unique().tolist()
    else:
      self.group_info = None
      self.group_types = None
    self.mol2trans_list = {}
    self.process_transitions()

  def hasGroup(self):
    return True if self.group_info is not None and 'Group' in self.group_info else False

  def process_transitions(self):
    chrom_table = self.chrom[['PeptideModifiedSequence', 'FileName', 'PrecursorCharge', 'FragmentIon', 'ProductCharge']].drop_duplicates()
    chrom_table['fragment'] = chrom_table['PrecursorCharge'].astype(str) + '.' + chrom_table['FragmentIon'] + '.' + chrom_table['ProductCharge'].astype(str)
    self.mol2trans_list = {}
    for mol, row in chrom_table.groupby('PeptideModifiedSequence'):
      trans_list = list(sorted(set(row['fragment'].unique())))
      self.mol2trans_list[mol] = trans_list

  def getSampleList(self):
    sample_list = []
    for idx, row in self.chrom[['PeptideModifiedSequence', 'FileName']].drop_duplicates().iterrows():
      pep = row['PeptideModifiedSequence']
      file = row['FileName']
      sample_list.append((file, pep))
    return sample_list

  def getChromListFromSample(self, sample_list):
    chrom_list = []
    for fn, pep in sample_list:
      chrom = self.getChromData(fn, pep)
      if chrom is not None:
        chrom_list.append(chrom)
    return chrom_list
  
  def getChromList(self, target=None):
    chrom_list = []
    for ps in self.target_list:
      if target and ps != target:
        continue
      for fn in self.filename_list:
        chrom = self.getChromData(fn, ps)
        if chrom is None:
          continue
        chrom_list.append(chrom)
    return chrom_list

  def _interpolate_intensity(self, col_data, interpolated_rt, original_rt):
    return np_interp(interpolated_rt, original_rt, col_data, left=0, right=0)
  
  def saveChromData(self, chrom_data):
    #(chrom['peptideModifiedSequence'], chrom['fileName'], (chrom['transitions'], endogenous_ions, standard_ions), (chrom['start'], chrom['end']), chrom['peak_intensity'], chrom['peak_time'], chrom['Area2SumRatio'], chrom['area'], chrom['time'], chrom['intensity'])
    key = chrom_data[0] + '.' + chrom_data[1]
    self.chromDB[key] = chrom_data
  def loadChromData(self, filename, target):
    key = target + '.' + filename
    return self.chromDB[key] if key in self.chromDB else None
  def getRandomBoundary(self, chrom_data):
    all_time = chrom_data[8]
    inValid = True
    while inValid:
      draw_rts = np_random_choice(all_time, 2, replace=False)
      draw_rts.sort()
      if draw_rts[0] != draw_rts[1]:
        inValid = False
    return draw_rts[0], draw_rts[1]
    
  def getChromData(self, fileName, pepModSeq, start=None, end=None, chrom_only=False, missing_transitions=False):
    chrom = self.loadChromData(fileName, pepModSeq)
    if chrom is None:
      sample = self.chrom[(self.chrom['FileName'] == fileName) & (self.chrom['PeptideModifiedSequence'] == pepModSeq)].reset_index()
      sample['fragment_names'] = sample['PrecursorCharge'].astype(str) + '.' + sample['FragmentIon'] + '.' + sample['ProductCharge'].astype(str) + '.' + sample['IsotopeLabelType']
      sample.index = sample['fragment_names']
      fragment_names = np_sort(np_unique(sample['PrecursorCharge'].astype(str) + '.' + sample['FragmentIon'] + '.' + sample['ProductCharge'].astype(str)))
      sample_intensities = sample['Intensities']
      sample_times = sample['Times']
      group_df = self.group_info[(self.group_info['FileName']==fileName.split('::')[0])&(self.group_info['PeptideModifiedSequence']==pepModSeq)].reset_index() if self.hasGroup() else None
      groupName = group_df.loc[0, 'Group'] if group_df is not None and len(group_df) > 0 else None
      endogenous_cols = []
      standard_cols = []
      intersect_frag_names = []
      for frag_name in fragment_names:
        endogenous_col = f"{frag_name }.light" if self.internal_standard_type == 'heavy' else f"{frag_name }.heavy"
        standard_col = f"{frag_name }.heavy" if self.internal_standard_type == 'heavy' else f"{frag_name }.light"
        if endogenous_col in sample_intensities and standard_col in sample_intensities:
            endogenous_cols.append(endogenous_col)
            standard_cols.append(standard_col)
            intersect_frag_names.append(frag_name)
      if len(standard_cols) == 0 or len(endogenous_cols) == 0:
        return None
      endogenous_intensity = sample_intensities[endogenous_cols].str.split(',')
      endogenous_time = sample_times[endogenous_cols].str.split(',')
      standard_intensity = sample_intensities[standard_cols].str.split(',')
      standard_time = sample_times[standard_cols].str.split(',')
      intersect_time_set = None
      for endo_frag, std_frag in zip(endogenous_cols, standard_cols):
        endo_time = endogenous_time[endo_frag]
        std_time = standard_time[std_frag]
        intersection_time = set(endo_time).intersection(set(std_time))
        if not intersect_time_set:
            intersect_time_set = intersection_time
        else:
            intersect_time_set = intersect_time_set.intersection(intersection_time)
      intersect_time_arr = list(intersect_time_set)
      intersect_time_arr.sort(key=float)
      intensity_dict = {}
      for endo_frag, std_frag in zip(endogenous_cols, standard_cols):
        endo_time = endogenous_time[endo_frag]
        endo_ints = endogenous_intensity[endo_frag]
        std_time = standard_time[std_frag]
        std_ints = standard_intensity[std_frag]
        endo_dict = dict(zip(endo_time, endo_ints))
        std_dict = dict(zip(std_time, std_ints))
        intensity_dict[endo_frag] = [endo_dict[time] for time in intersect_time_arr]
        intensity_dict[std_frag] = [std_dict[time] for time in intersect_time_arr]
      if missing_transitions:
        all_endogenous_cols = []
        all_standard_cols = []
        for frag_name in self.mol2trans_list[pepModSeq]:
          endogenous_col = f"{frag_name }.light" if self.internal_standard_type == 'heavy' else f"{frag_name }.heavy"
          standard_col = f"{frag_name }.heavy" if self.internal_standard_type == 'heavy' else f"{frag_name }.light"
          all_endogenous_cols.append(endogenous_col)
          all_standard_cols.append(standard_col)
          if endogenous_col not in intensity_dict:
            intensity_dict[endogenous_col] = np_zeros(len(intersect_time_arr))
          if standard_col not in intensity_dict:
            intensity_dict[standard_col] = np_zeros(len(intersect_time_arr))
        all_fragments = all_endogenous_cols + all_standard_cols
      else:
        all_fragments = endogenous_cols + standard_cols
      intensity_mx = [ intensity_dict[frag] for frag in all_fragments]
      all_time = np_array([float(rt) for rt in intersect_time_arr])
      intensity = np_array(intensity_mx, dtype=np_float32).T
      if intensity.shape[0] != all_time.shape[0]:
        print(f"Chromatogram {fileName} {pepModSeq} has inconsistent lengths of retention time ({all_time.shape[0]}) and intensity ({intensity.shape[0]}).")
        return None
      interpolated_rt = np_linspace(all_time[0], all_time[-1], self.timepoint)
      interpolated_intensity = np_apply_along_axis(self._interpolate_intensity, 0, intensity, interpolated_rt, all_time)
      if missing_transitions:
        chrom = (pepModSeq, fileName, (self.mol2trans_list[pepModSeq], all_endogenous_cols, all_standard_cols), None, None, None, None, None, all_time, intensity, interpolated_rt, interpolated_intensity, groupName)  
      else:  
        chrom = (pepModSeq, fileName, (intersect_frag_names, endogenous_cols, standard_cols), None, None, None, None, None, all_time, intensity, interpolated_rt, interpolated_intensity, groupName)
      self.saveChromData(chrom)
    else:
      try:
        all_time = chrom[8]
        intensity = chrom[9]
      except Exception as e:
        print(e)
        print(chrom)
        return None
    if start is None or end is None:  
      if self.peak_boundary is None:
        (start, end) = self.getRandomBoundary(chrom)
      else:
        start = None
        end = None
        start_list = list(self.peak_boundary[(self.peak_boundary['FileName']==fileName)&(self.peak_boundary['PeptideModifiedSequence']==pepModSeq)]['MinStartTime'])
        end_list = list(self.peak_boundary[(self.peak_boundary['FileName']==fileName)&(self.peak_boundary['PeptideModifiedSequence']==pepModSeq)]['MaxEndTime'])
        if len(start_list) == 0 or len(end_list) == 0:
            fileName2 = fileName.split('::')[0]
            start_list = list(self.peak_boundary[(self.peak_boundary['FileName']==fileName2)&(self.peak_boundary['PeptideModifiedSequence']==pepModSeq)]['MinStartTime'])
            end_list = list(self.peak_boundary[(self.peak_boundary['FileName']==fileName2)&(self.peak_boundary['PeptideModifiedSequence']==pepModSeq)]['MaxEndTime'])
        if (len(start_list) == 0 or len(end_list) == 0) and chrom_only:
          return chrom
        else:
          start = start_list[0]
          end = end_list[0]
        try:
          start = float(start)
          end = float(end)
          if np_isnan(start) or np_isnan(end):
            return chrom if chrom_only else None
        except:
          return chrom if chrom_only else None
    return self.updateChromData(chrom, start=start, end=end)

  def updateChromData(self, chrom_data, start=None, end=None):
    if start is None or end is None:
      (start, end) = self.getRandomBoundary(chrom_data)
    if start > end:
      (start, end) = (end, start)
    global_min_rt = np_min(chrom_data[8])
    global_max_rt = np_max(chrom_data[8])
    if start < global_min_rt:
        start = global_min_rt
    if end > global_max_rt:
        end = global_max_rt
    start = round(start, 3)
    end = round(end, 3)
    all_time = chrom_data[8]
    intensity = chrom_data[9]
    peak_filter = (all_time >= start) & (all_time <= end)
    peak_time = all_time[peak_filter]
    peak_intensity = intensity[peak_filter, :]
    area = np_apply_along_axis(np_trapz, 0, peak_intensity, peak_time)
    (light_area, heavy_area) = np_hsplit(area, 2)
    light_area2sum_ratio = light_area/np_sum(light_area) if np_sum(light_area) > 0 else np_array([0]*int(peak_intensity.shape[1]/2))
    heavy_area2sum_ratio = heavy_area/np_sum(heavy_area) if np_sum(heavy_area) > 0 else np_array([0]*int(peak_intensity.shape[1]/2))
    area2sum_ratio = np_concatenate((light_area2sum_ratio, heavy_area2sum_ratio))
    if len(chrom_data) == 13:
      new_chrom_data = (chrom_data[0], chrom_data[1], chrom_data[2], (start, end), peak_intensity, peak_time, area2sum_ratio, area, chrom_data[8], chrom_data[9], chrom_data[10], chrom_data[11], chrom_data[12])
    else:
      new_chrom_data = (chrom_data[0], chrom_data[1], chrom_data[2], (start, end), peak_intensity, peak_time, area2sum_ratio, area, chrom_data[8], chrom_data[9])
    return new_chrom_data
  def transitionOrderConsistency(self, chrom_data):
    if not chrom_data[3]:
      return -1, -1, -1, -1, -1, -1
    peak_time = chrom_data[5]
    area = chrom_data[7]
    if len(peak_time) == 0:
      return -1, -1
    (light_area, heavy_area) = np_hsplit(area, 2)
    dotp = np_dot(light_area, heavy_area)/(norm(light_area) * norm(heavy_area))
    contrast_angle = 1 - (2 * math_acos(dotp)/math_pi)
    return dotp, contrast_angle

  def plotChromData(self, chrom = None, fn=None, ps=None, transitions=None, zoom=None, top_n=5):
    if chrom is None:
      chrom_data = self.getChromData(fn, ps)
    else:
      chrom_data = chrom
    if chrom_data is None:
      return None
    fn = chrom_data[0]
    ps = chrom_data[1]
    fig, (ax1, ax2) = plt.subplots(2, 1)
    rt_time = chrom_data[8]
    intensity = chrom_data[9]
    (start, end) = chrom_data[3]
    if zoom is not None:
      mid_width = (end - start)/2
      view_start = start - zoom * mid_width
      view_start =  view_start if view_start >= rt_time[0] else rt_time[0]
      view_end = end + zoom * mid_width
      view_end = view_end if view_end <= rt_time[-1] else rt_time[-1]
      rt_filter = (rt_time >= view_start) & (rt_time <= view_end)
      rt_time = rt_time[rt_filter]
      intensity = intensity[rt_filter, :]
    light_intensity, heavy_intensity = np_hsplit(intensity, 2)
    light_global_max_intensity = np_max(light_intensity)
    heavy_global_max_intensity = np_max(heavy_intensity)
    light_exp = len(str(int(light_global_max_intensity))) - 2
    heavy_exp = len(str(int(heavy_global_max_intensity))) - 2
    if light_exp < 0 :
      light_exp = 0
    if heavy_exp < 0:
      heavy_exp = 0
    light_int = (light_intensity/(10**light_exp))
    heavy_int = (heavy_intensity/(10**heavy_exp))
    filtered_trans_idx_list = None
    if len(chrom_data[2][0]) > top_n:
      (_, heavy_peak_ints) = np_hsplit(chrom_data[4], 2)
      if heavy_peak_ints.shape[0] > 0:
        heavy_max_peak_ints = np_max(heavy_peak_ints, axis=0)
        filtered_trans_idx_list = np_argsort(heavy_max_peak_ints)[-top_n:]
        filtered_trans_idx_list = np_flip(filtered_trans_idx_list)
    for idx, fragment in enumerate(chrom_data[2][1]):
      if transitions is None or chrom_data[2][0][idx] in transitions:
        ax1.plot(rt_time, light_int[:, idx],  linewidth=1, label=fragment)
    for idx, fragment in enumerate(chrom_data[2][2]):
      if transitions is None or chrom_data[2][0][idx] in transitions:
        ax2.plot(rt_time, heavy_int[:, idx],  linewidth=1, label=fragment)
    ax1.set_title(fn + ': ' + ps, fontsize=8)
    ax1.set_ylabel('Intensity(10^%s)'%light_exp)
    ax1.axvline(start, linestyle= '--', linewidth=1,color='black')
    ax1.axvline(end, linestyle= '--', linewidth=1,color='black')
    if filtered_trans_idx_list is None:
      ax1.legend(fontsize=8, loc='best', frameon=False)
    else:
      h, l = ax1.get_legend_handles_labels()
      ax1.legend(fontsize=8, loc='best', frameon=False, handles=[h[frag_idx] for frag_idx in filtered_trans_idx_list], labels = [ l[frag_idx] for frag_idx in filtered_trans_idx_list])
    ax2.set_ylabel('Intensity(10^%s)'%heavy_exp)
    ax2.set_xlabel('Retention Time')
    ax2.axvline(start, linestyle= '--', linewidth=1,color='black')
    ax2.axvline(end, linestyle= '--', linewidth=1,color='black')
    if filtered_trans_idx_list is None:
      ax2.legend(fontsize=8,loc='best', frameon=False)
    else:
      h, l = ax2.get_legend_handles_labels()
      ax2.legend(fontsize=8, loc='best', frameon=False, handles=[h[frag_idx] for frag_idx in filtered_trans_idx_list], labels = [ l[frag_idx] for frag_idx in filtered_trans_idx_list])
    return fig
