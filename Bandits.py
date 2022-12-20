#for loading csvs and initializing zero and one vectors, not for main logic
import numpy as np
import pandas as pd

#alg file contains main algos
#reward_probability contains functions for calculating reward probability matrices

import alg
import reward_probability




#loading csvs for dataframes
df_bool = pd.read_csv("BoolData.csv")
df_range_data = pd.read_csv("RangeData.csv")
reward_prob_vect_bool = reward_probability.reward_prob_arr(df_bool)
reward_prob_vect_rd = reward_probability.reward_prob_arr(df_range_data)
rewards_bool = [1.0] * len(reward_prob_vect_bool)
rewards_rd = [1.0] * len(reward_prob_vect_rd)

#environment definition
class Env(object):
    def __init__(self,reward_prob_vect,reward_vect):

        self.reward_probabilities=reward_prob_vect
        self.rewards=reward_vect
        self.num_arms=len(reward_prob_vect)

    def choose_arm(self,arm):

        random_probability=np.random.random()
        if random_probability<self.reward_probabilities[arm]:
            return rewards_bool[arm]
        else:
            return 0.0


#driver code
#environment one for bool csv, var environment2 for weighted csv
environment1=Env(reward_prob_vect_bool,rewards_bool)
environment2=Env(reward_prob_vect_rd,rewards_rd)

#stationary average agent for environment1 and environment2
stat_agent=alg.statAlgAgent(environment1)
stat_agent.act()
stat_agent=alg.statAlgAgent(environment2)
stat_agent.act()



# rolling average agent for environment1 and environment2
environment2=Env(reward_prob_vect_rd,rewards_rd)
roll_agent=alg.MovingAvgAgent(environment1)
roll_agent.act()
roll_agent=alg.MovingAvgAgent(environment2)
roll_agent.act()

#exponential recency agent for environment1 and environment2
exp_recency_agent=alg.ExponentialRecencyAgent(environment1)
exp_recency_agent.act()
exp_recency_agent=alg.ExponentialRecencyAgent(environment2)
exp_recency_agent.act()






