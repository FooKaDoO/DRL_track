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