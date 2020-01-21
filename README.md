# Spectrum sharing in vehicular networks based on multi-agent reinforcement learning

L. Liang, H. Ye, and G. Y. Li, "Spectrum sharing in vehicular networks based on multi-agent reinforcement learning," IEEE Journal on Selected Areas in Communications, vol. 37, no. 10, pp. 2282-2292, Oct. 2019. 

Trained and tested with Python 3.6 + TensorFlow 1.12.0. 

How to use the code:

- To train the multi-agent RL model: main_marl_train.py + Environment_marl.py + replay_memory.py
- To train the benchmark single-agent RL model: main_sarl_train.py + Environment_marl.py + replay_memory.py
- To test all models in the same environment: main_test.py + Environment_marl_test.py + replay_memory.py + '/model'. 
  - Figures 3 and 4 in the paper can be directly reproduced from running "main_test.py". Change the V2V payload size by "self.demand_size" in "Environment_marl_test.py".
  - Figure 5 can only be obtained from recording returns during training. 
  - Figures 6-7 show performance of an arbitrary episode (but with failed random baseline and successful MARL transmission). In fact, most of such episodes exhibit some interesting observations demonstrating multi-agent cooperation. Interpretation is up to the readers. 
  - Use of "Test" mode in "main_marl_train.py" is not recommended. 


Please send all questions/inquires to lliang@gatech.edu.
