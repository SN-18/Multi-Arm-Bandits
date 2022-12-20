## Multi-Arm-Bandits
-Multi-arm bandit problem is a reinforcement ml-problem whose objective is allocating resources among competing needs. These competing needs can be thought of as 'arms'.<br> 
-The name comes from the Gambler's dillemma of choosing between exploration of 'arms' and exploitation or using his knowledge gained in order to maximize the gain. <br>
-More detailed description and applications, such as in clinical trials,portfolio management, routing, etc  can be found on [this page](https://en.wikipedia.org/wiki/Multi-armed_bandit). <br>


## Three types of Multi-Arm-Bandits-Agents are implemented in this project:

1. Stationary Avg. Agent: For this, choosing an arm takes place as a stationary process, meaning that in this one assumes the probability distribution for the rewards for the arms, is not dependent on time, that is for a given window size, the mean would not fluctuate (due to time shocks, read more [on this page.](https://en.wikipedia.org/wiki/Stationary_process) 
2. Rolling Avg. Agent: In this, one takes into account the role of a trend in the mean of rewards for choosing an arm. One assumes a linear trend, with some parameter vector [epsilon1,epsilon2...], that is one adjusts according to changes in mean with regards to time shocks in distribution of time (time series analysis, read more [here.](https://en.wikipedia.org/wiki/Time_series) 
3. Exponential Recency Agent: In this, one gives more weight to more recent mean distributions of reward, and assume our epsilon coeffecients decay with a linear rate (an adjustable hyperparameter, read more on [hyperparameters here.](https://en.wikipedia.org/wiki/Hyperparameter_machine_learning)

##How to run the code
1. Check if you have Python 3.X installed. For this, open a command line environment (terminal for Mac OS X, cmd for windows, etc).
2. If Python is not installed, instructions to download and install can be found on [this page.](https://www.python.org/downloads/)
3. Fork the given repo using the fork button above (Right hand side): <br>
<img width="542" alt="image" src="https://user-images.githubusercontent.com/83748468/208619480-85380070-ce4d-4246-b6ab-5bbc459f4373.png">
4. Make sure you have git installed, instructions can be found [on this page.](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
5. Clone the fork using <br>
   $ git clone <url_of_fork>
To get url of fork, click on code button and then copy:
<img width="1006" alt="image" src="https://user-images.githubusercontent.com/83748468/208620047-c938eae6-0005-45d4-b131-8fdf4a595c00.png">
6. Run the command:
  $ python bandits.py <br>
This will invoke the three agents, stationary, rolling and exponential and will walk you through choosing an arm based on the reward distribution.

##Description of Files 
1. bandits.py - Contains driver code. Invokes the three reinforcement agents.
2. alg.py - Contains src code for three agent classes and is invoked by bandits.py
3. Bool.csv and RangeData.csv: Contains the reward distribution for discrete and continuous rewards, respectively






