#I have these functions
import numpy as np
import pandas as pd
import alg
import reward_probability

class statAlgAgent(object):

    def __init__(self,env,iterations=400):
        self.env=env
        self.iterations=iterations

        self.q_values=np.zeros(self.env.num_arms)
        self.arm_counts=np.zeros(self.env.num_arms)
        self.arm_rewards=np.zeros(self.env.num_arms)

        self.rewards=[0.0]
        self.rewards_cumulative=[0.0]

    def act(self):
        for i in range(self.iterations):
            arm=np.argmax(self.q_values)
            print("the chosen are is ",arm)
            reward=self.env.choose_arm(arm)

            self.arm_counts[arm]=self.arm_counts[arm] + 1
            self.arm_rewards[arm]=self.arm_rewards[arm] +reward

            self.q_values[arm]=self.q_values[arm] + 1/(self.arm_counts[arm])*\
                               (reward-self.q_values[arm])
            self.rewards.append(reward)
            self.rewards_cumulative.append(sum(self.rewards)/len(self.rewards))
            print(f"for STAT algo, current step is: {i}, current reward is :{self.rewards[-1]}, current cumulative reward is :{self.rewards_cumulative[-1]}")
        print("End of iterations\n")
class MovingAvgAgent(object):
    def __init__(self, env,max_iterations=400,epsilon=0.01,decay=0.01,window_size=10):
        self.env=env
        self.iterations=max_iterations
        self.epsilon=epsilon
        self.decay=decay
        self.window=window_size

        self.q_values=np.zeros(self.env.num_arms)
        self.arm_counts=np.zeros(self.env.num_arms)
        self.arm_rewards=np.zeros(self.env.num_arms)

        self.rewards=[0.0]
        self.rewards_cumulative=[0.0]

    def act(self):
        for i in range(self.iterations):
            arm=np.random.choice(self.env.num_arms) if np.random.random() <self.epsilon else np.argmax(self.q_values)
            reward=self.env.choose_arm(arm)

            self.arm_counts[arm]=self.arm_counts[arm]+1
            self.arm_rewards[arm]=self.arm_rewards[arm]+reward

            self.q_values[arm]=self.q_values[arm] + (1/self.arm_counts[arm])*(reward -self.q_values[arm])
            self.rewards.append(reward)
            self.rewards_cumulative.append(sum(self.rewards) / len(self.rewards))

            if i % self.window == 0:
                self.epsilon = self.epsilon * self.decay
            print(
                f"For ROLL algo, the current step is {i},current reward is {self.rewards[-1]},current cumulative reward is: {self.rewards_cumulative[-1]}")
        print("End of iterations\n")
class ExponentialRecencyAgent:
    def __init__(self,env, max_iterations=400,epsilon=0.01,decay=0.01,window_size=10):
        self.env=env
        self.iterations=max_iterations
        self.epsilon=0.01
        self.decay=decay
        self.window=window_size

        self.arm_counts=np.zeros(self.env.num_arms)
        self.arm_rewards=np.zeros(self.env.num_arms)
        self.q_values = np.zeros(self.env.num_arms)
        self.rewards=[0.0]
        self.rewards_cumulative=[0.0]

    def act(self):
        for i in range(self.iterations):
            arm=np.random.choice(self.env.num_arms) if np.random.random()<self.epsilon else np.argmax(self.q_values)
            reward=self.env.choose_arm(arm)

            self.arm_counts[arm]=self.arm_counts[arm]+1
            self.arm_rewards[arm]=self.arm_counts[arm]+reward

            self.q_values[arm]=((1-self.epsilon)**(i-1))*self.q_values[arm]+self.epsilon*(1-self.epsilon)*(reward-self.q_values[arm])
            self.rewards.append(reward)
            self.rewards_cumulative.append(sum(self.rewards)/len(self.rewards))

            if i% self.window==0:
                self.epsilon=self.epsilon*self.decay

            print(f"For exponential recency algo, the current step is {i},current reward is {self.rewards[-1]},current cumulative reward is: {self.rewards_cumulative[-1]}")
        print("End of iterations\n")












