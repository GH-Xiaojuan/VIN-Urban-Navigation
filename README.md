# VIN-Urban-Navigation
Code for Learning Urban Navigation via Value Iteration Network.

Requires:

* Python3
* TensorFlow

You should run the ./data/data.py to get the training data at first,
then run the train.py to train the model.
The data contains two part: rewards.dat and decisions.dat.

The rewards.dat contains the TTM of 8 direction of each time slice, and decisions.dat contains the [slice, goal_x, goal_y, cur_x, cur_y, action].
