See writeup.txt for the writeup.

I changed the number of episodes aggregated per data points in the plot to 25 because the total number of training episodes was 1000, which roughly corresponded to the number of total training steps from the DQN paper (~ 1 million) 


In run.py, change `algo` and/or `n_games` to toggle between DQN and Double DQN (use the class names found in `DQN.py`) and to change the number of training steps.