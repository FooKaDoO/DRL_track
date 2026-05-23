# What features to remove


First, we look at which global metrics were not useful.

    mean_height - low score drop (min max and deviation were a lot higher impact)
    hole_depth_deviation - relatively high deviation compared to mean (gets negative sometimes)
    hole_vertical_instability - relatively high deviation compared to mean (gets negative sometimes)
    mean_hole_horizontal_clusterdness - relatively high deviation compared to mean (gets negative sometimes)
    hole_edge_deviation - low score drop
    hole_horizontal_instability - low score drop
    mean_hole_distance - relatively high deviation compared to mean (gets negative sometimes)
    general_hole_clusterdness - low score drop

Secondly, let's try to minimize the features.

It seems that horizontal hole clusterdness has relatively high deviation compared to it's mean, therefore we remove these, since the mean edge distance has a pretty high value.
