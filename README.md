This is the companion repository for the following paper. Trained and tested with Python 3.6 + TensorFlow 1.12.0. 

L. Liang, H. Ye, and G. Y. Li, "Spectrum sharing in vehicular networks based on multi-agent reinforcement learning," IEEE Journal on Selected Areas in Communications, vol. 37, no. 10, pp. 2282-2292, Oct. 2019. Available at: https://ieeexplore.ieee.org/document/8792382


How to use the code:

- To train the multi-agent RL model: main_marl_train.py + Environment_marl.py + replay_memory.py
- To train the benchmark single-agent RL model: main_sarl_train.py + Environment_marl.py + replay_memory.py
- To test all models in the same environment: main_test.py + Environment_marl_test.py + replay_memory.py + '/model'. Change the V2V payload size using the variable "self.demand_size" in Environment_marl_test.py. 


Please send all questions/inquires to lliang@gatech.edu.
