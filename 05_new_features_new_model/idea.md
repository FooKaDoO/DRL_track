# Features

Firstly, we removed the following 28 metrics

    mean_height
    hole_depth_deviation
    hole_vertical_instability
    mean_hole_horizontal_clusterdness
    hole_edge_deviation
    hole_horizontal_instability
    mean_hole_distance
    general_hole_clusterdness
    horizontal_hole_clusterdness_row_0-19


After removing 28 features, we have room for new features.

One main feature, overlooked previously, is the next piece. This indicates heavily, what the possible moves are and what we should choose. Therefore, we make it a possible feature, a classification feature (for each piece, we have a feature and if the next piece is it, it is 1, otherwise 0)

# Model

Secondly, we might want to edit the model. The current model is very simple, no layernorm or anything.

We make model more complex: each "group" has an encoder, which encodes the information into something maybe more meaningful.

Then these encoded results are given to the head (original model structure) which gives final score.

Additionally, we add LayerNorm, which normalizes the activations per sample, across the layer's features, but it is still a trainable model, which learns scale and shift, so it can use useful values.

Additionally, we replace ReLU with SiLU, which doesn't remove, but reduces negative values and therefore neurons that would usually be dead, are kept alive. 

Finally, we add conv1d over column features as a 5th group so we can learn more spacial details (bumps, cliffs, etc.)

# Training

Finally, training. We want to train a lot longer. Firstly, we optimize the training loop to work a lot faster. Then, we change some parameters.

    EPISODES = 10000 (I benchmarked and this is around 5.5 hours)
    EPSILON_DECAY = 0.9992 (so the model is a lot less greedy, reaches epsilon floor on episode 8632)

Then changed nn.MSELoss() to nn.SmoothL1Loss(), which makes huge rewards/penalties smaller (not squared like MSE), to avoid overly large changes.

Same reason is for clipping the gradient, since it is such a long-running task, we clip the gradient to not mess the entire training up due to 1 overly large reward/penalty.

Finally, we add checkpoint saving to see results later. The checkpoint logic saves every 1000 episodes until episode 5000, then every 500 episodes until episode 2000, then every 200 episodes until episode 600 and then every 100 episodes until the end (10000th episode). Then it can be nicely seen afterwards, how the results look like.