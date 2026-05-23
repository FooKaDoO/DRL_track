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