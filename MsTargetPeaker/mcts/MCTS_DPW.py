# Modified from https://github.com/martinobdl/MCTS and reorganized for MsTargetPeaker
from numpy import array as np_array, sum as np_sum, clip as np_clip, sqrt as np_sqrt, log as np_log, inf as np_inf, argmax as np_argmax
from numpy.random import choice as np_random_choice, random_sample as np_random_sample
import copy

class StateNode:
  def __init__(self, state=None, chrom=None, father=None, step=0, is_root=False, is_final=False):
    self.state = state
    self.chrom = chrom
    self.children = {}
    self.is_final = is_final
    self.visits = 0
    self.reward = 0
    self.step = step
    self.father = father
    self.is_root = is_root
  def add_children(self, action_node, hash_preprocess):
    self.children[hash_preprocess(action_node.action)] = action_node
  def next_action_node(self, action, hash_preprocess):
    if hash_preprocess(action) not in self.children.keys():
      new_action_node = ActionNode(action, father=self)
      self.add_children(new_action_node, hash_preprocess)
    else:
      new_action_node = self.children[hash_preprocess(action)]
    return new_action_node
class ActionNode:
  def __init__(self, action, father=None):
    self.action = action
    self.children = {}
    self.cumulative_reward = 0
    self.visits = 0
    self.father = father
  def add_children(self, x, hash_preprocess):
    self.children[hash_preprocess(x.state)] = x

class MCTS_DPW():
  def __init__(self, alpha, beta, initial_obs, initial_chrom, env, K, eval_mode='policy', selection_noise = 0):
    self.env = env
    self.alpha = alpha
    self.beta = beta
    self.K = K
    self._serialize_action = lambda x: tuple(x)
    self._serialize_state = lambda x: tuple(x[1,:])
    self.selection_noise = selection_noise
    self.root = StateNode(state=initial_obs, chrom = initial_chrom, step=0, is_root=True)
    if eval_mode == 'policy':
      self.mode = 'policy'
    elif eval_mode == 'value':
      self.mode = 'value'
    else:
      self.mode = 'random'
    score_obj = env.get_chrom_score(initial_chrom)
    self.history = [dict(
      sampleName = initial_chrom[1],
      peptide=initial_chrom[0],
      peak_start = initial_chrom[3][0],
      peak_end = initial_chrom[3][1],
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
    )]

  def select_outcome(self, state_node, action_node):
    if action_node.visits ** self.beta >= len(action_node.children):
      obs_mx = state_node.state
      chrom = state_node.chrom
      next_obs, r, done, info = self.env.step(obs_mx, chrom, state_node.step, action_node.action, self.history)
      step = state_node.step + 1
      new_chrom = info['chrom']
      return StateNode(state=next_obs, chrom=new_chrom, step=step, father=action_node, is_final=done), r
    else:
      unnorm_probs = [child.visits for child in action_node.children.values()]
      probs = np_array(unnorm_probs)/np_sum(unnorm_probs)
      chosen_state = np_random_choice(list(action_node.children.values()), p=probs)
      return (chosen_state, chosen_state.reward)
  def select(self, x):
    if x.visits**self.alpha >= len(x.children):
      if self.selection_noise < 0:
        a = 2 * np_random_sample((2, )) - 1
      else:
        obs_mx = x.state
        action = self.env.predict(obs_mx)
        noise = (2*np_random_sample(2)-1)*self.selection_noise
        a = np_clip(action + noise, -1, 1)
    else:
      def scoring(k):
        if x.children[k].visits > 0:
          return (x.children[k].cumulative_reward / x.children[k].visits) + self.K * np_sqrt(np_log(x.visits)/x.children[k].visits)
        else:
          return np_inf
      a = max(x.children, key=scoring)
    return a
  def evaluate(self, state_node):
    R = 0
    done = False
    obs_mx = state_node.state
    chrom = state_node.chrom
    step = state_node.step
    if self.mode == 'value':
      v = self.env.predict_val(obs_mx)
      return v
    while not done:
      if self.mode == 'random':
        a = 2 * np_random_sample((2, )) - 1
      elif self.mode == 'policy':
        a = self.env.predict(obs_mx)
      obs_mx, r, done, info = self.env.step(obs_mx, chrom, step, a, history = self.history)
      chrom = info['chrom']
      step += 1
      R += r
    return R
  def forward(self, action, new_state, new_chrom, new_step):
    if self._serialize_action(action) in self.root.children.keys():
      rnd_node = self.root.children[self._serialize_action(action)]
      if len(rnd_node.children) > 1:
        self.root = StateNode(state=new_state, chrom=new_chrom, step=new_step, is_root=True)
      else:
        next_state_node = np_random_choice(list(rnd_node.children.values()))
        next_state_node.father = None
        self.root.children.pop(self._serialize_action(action))
        self.root = next_state_node
        self.root.is_root = True
    else:
      raise RuntimeWarning("Action taken: {} is not in the children of the root node.".format(action))
  def update_state_node(self, state_node, action_node, hash_preprocess):
    if hash_preprocess(state_node.state) not in action_node.children.keys():
      state_node.father = action_node
      action_node.add_children(state_node, hash_preprocess)
    else:
      state_node = action_node.children[hash_preprocess(state_node.state)]
    return state_node
  def best_action(self):
    children_nodes = list(self.root.children.values())
    number_of_visits_children = [node.visits for node in children_nodes]
    index_best_action = np_argmax(number_of_visits_children)
    a = children_nodes[index_best_action].action
    return a
  def get_best_peak_from_history(self):
    best_peak = None
    for history in self.history:
      if best_peak is None or history['final_reward'] > best_peak['final_reward']:
        best_peak = copy.deepcopy(history)
    return best_peak
  def learn(self, Nsim):
    iterations = range(Nsim)
    for _ in iterations:
      self.grow_tree()
  def grow_tree(self):
    state_node = self.root
    while (not state_node.is_final) and state_node.visits > 1:
      a = self.select(state_node)
      new_action_node = state_node.next_action_node(a, self._serialize_action)
      (new_state_node, r) = self.select_outcome(state_node, new_action_node)
      new_state_node = self.update_state_node(new_state_node, new_action_node, self._serialize_state)
      new_state_node.reward = r
      new_action_node.reward = r
      state_node = new_state_node
    state_node.visits += 1
    cumulative_reward = self.evaluate(state_node)
    while not state_node.is_root:
      action_node = state_node.father
      cumulative_reward += action_node.reward
      action_node.cumulative_reward += cumulative_reward
      action_node.visits += 1
      state_node = action_node.father
      state_node.visits += 1
