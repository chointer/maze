import numpy as np

class RewardManager:
    def __init__(self):
        self.rewards_use = {}      # { r_name: {fn: fn, reward: float, penalty: float} }
        self.reward_inputs = {
            'terminated': None,             # goal
            'action': None,                 # friction
            'manual_mode': None,            # friction
            'manhattan_map': None,          # manhattan_dist
            'distance_map': None,           # shortest_path
            'agent_location': None,         # manhattan_dist, shortest_path
            'agent_location_before': None,  # manhattan_dist, shortest_path
        }
        self.available_rewards = {
            'goal': self._goal_fn,
            'friction': self._friction_fn,
            'manhattan_dist': self._manhattan_fn,
            'shortest_path': self._shortest_fn,
        }
        #self.storage = {}

    
    def __call__(self):        # kwargs로 바꾸는게 나으려나?
        reward_total = 0.0
        rewards = {}
        for r_name, r in self.rewards_use.items():
            reward_element = r['fn'](reward=r['reward'], penalty=r['penalty'])
            rewards[r_name] = reward_element
            reward_total += reward_element
        return reward_total, rewards
    

    def add(self, reward_name, reward, penalty):
        if reward_name not in self.available_rewards.keys():
            raise ValueError(f"Unsupported: {reward_name}. Availables: {list(self.available_rewards.keys())}")
        self.rewards_use[reward_name] = {
            'fn': self.available_rewards[reward_name], 
            'reward': reward,
            'penalty': penalty
        }            


    def adjust(self, reward_name, reward=None, penalty=None):
        if reward_name not in self.rewards_use.keys():
            raise ValueError(f"Not included: {reward_name}. Used: {list(self.rewards_use.keys())}")
        if isinstance(reward, float):
            self.rewards_use[reward_name]['reward'] = reward
        if isinstance(penalty, float):
            self.rewards_use[reward_name]['penalty'] = penalty


    def _goal_fn(self, reward=1, penalty=None):
        if self.reward_inputs['terminated'] is None:
            raise ValueError("Set reward_inputs['terimiated']")
        return reward if self.reward_inputs['terminated'] else 0.
    
    
    def _friction_fn(self, reward=None, penalty=0.01):
        if self.reward_inputs['manual_mode'] is None or self.reward_inputs['action'] is None:
            raise ValueError("Set 'manual_mode' and 'action'")
        if self.reward_inputs['manual_mode'] is False or self.reward_inputs['action'] != -1:
            return -penalty
        else:
            return 0

    
    def _manhattan_fn(self, reward=0.1, penalty=None):
        # 벽을 무시한 manhattan distance 기준, 멀어지면 페널티 / 가까워지면 보상
        if self.reward_inputs['manhattan_map'] is None or self.reward_inputs['agent_location'] is None or self.reward_inputs['agent_location_before'] is None:
            raise ValueError("Set 'manhattan_map', 'agent_location' and 'agent_location_before'")
        d0 = self.reward_inputs['manhattan_map'][*self.reward_inputs['agent_location_before']]
        d1 = self.reward_inputs['manhattan_map'][*self.reward_inputs['agent_location']]
        dd = float(d1 - d0)
        if penalty is None:
            penalty = reward
        return -dd*reward if dd <= 0 else -dd*penalty

    
    def _shortest_fn(self, reward=0.1, penalty=None):
        # 최단 거리 기준, 멀어지면 페널티 / 가까워지면 보상
        if self.reward_inputs['distance_map'] is None or self.reward_inputs['agent_location'] is None or self.reward_inputs['agent_location_before'] is None:
            raise ValueError("Set 'distance_map', 'agent_location' and 'agent_location_before'")
        d0 = self.reward_inputs['distance_map'][*self.reward_inputs['agent_location_before']]
        d1 = self.reward_inputs['distance_map'][*self.reward_inputs['agent_location']]
        dd = float(d1 - d0)
        if penalty is None:
            penalty = reward
        return -dd*reward if dd <= 0 else -dd*penalty
        


    """
    def add_maze_revisit(self, obs, penalty=0.05, revisit_multiplier=1):
        # 이미 방문한 장소에 재방문 시 페널티 부여. 
        # 재방문 횟수가 증가할 때마다 reward**revisit_multipiler
        # 신경망 입력으로부터 과거 정보를 파악할 수 없다면 사용하지 않는 것이 좋을 것 같다.
        self.storage['visit_map'] = np.zeros(obs.shape[:2])
        agent_loc_y, agent_loc_x = np.where(obs[:, :, -2])
        self.storage['visit_map'][agent_loc_y, agent_loc_x] += 1

        def revisit_fn():
            if self.reward_inputs['agent_location'] is None:
                raise ValueError("Set 'agent_location'")
            r = -(penalty * revisit_multiplier ** self.storage['visit_map'][*self.reward_inputs['agent_location']])
            self.storage['visit_map'][*self.reward_inputs['agent_location']] += 1
            return r
        self.add_term('revisit', revisit_fn)
        self.reward_inputs['agent_location']= None
    """