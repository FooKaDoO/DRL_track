# Idea

Firstly, let's keep it simple: let's evaluate a board, no comparison for now.

## Ideas for inputs

To evaluate a board, we might want to have some info about the board.

Giving the entire board is inefficient and doesn't help it understand the structure.

That is why we are giving it inputs like column heights, bumpiness, total_height and completed rows.

Firstly, the current inputs

### Heights

It is just the numeric values for the heights for each column. They should be normalized to the full height, since ML models perform better with values between 0 and 1.

Another problem, is that it doesn't describe the column's very well, how many holes there are and how are the holes placed.

### Vertical holeyness

For that we might want to calculate the % of empty cells inside the height (not entire column). Also, to understand how they are placed, we need some metrics.

For that we can define the holes as distance from bottom. Then, if the holes are mostly high or low can be interpeted as the average of that distance.

Since we want to weight bad things more heavy, we might instead of looking at the average height, we look at average depth mean(1 - height).

The hole clusterdness can be defined as standard deviation of this value. Low standard devitation indicates heavy clustering, meanwhile high standard deviation spreadness. Since the values are from 0 to 1, then the highest standard deviation can be 0.5, and therefore, we want to normalize this value. Also, in tetris, it is considered that individual holes are significantly better than clustered holes, therefore we invert this value. So our value would become clusterdness = 1 - 2 * std(height).

### Horizontal holeyness

Same as vertical holeyness but horizontally. A little difference is, that the nr distance from left, or right doesn't give much info. Or it does, but the model would have to learn that 0 and 1 are bad (or good), and the middle is better (or vice versa), which is more complicated than a linear value (good to bad, or vice versa). Therefore, we define the a distance from middle instead, which is: distance_from_middle = abs(1 - 2 * hole_position_normalized).

For clusterdness, we still use the left to right metric hole_position_normalized, since if a point is far left and far right, then they have high distance_from_middle, and would appear clustered.

### Bumpiness

The current bumpiness is the sum of all the height diffs of 2 consecutive columns. This is a good metric, but a much better metric, used in statistics, to evaluate deviations like this, is mean +- std. Meaning, we add the mean height and standard deviation of height as inputs. We can call the standard deviation of heights as bumpiness, although it is not quite it anymore.

This actually inspired to me the idea of horizontal holeyness

### Total height

We already have heights, and bumpiness and holes that will describe the state pretty well. Generally, total height, I don't think it explains the state very well.

We have mean and std of heights, we add min height and max height instead of total height. These 2 metrics tie the picture together, I think.

### Actually, we want some global metrics

```python
total_filledness = sum(fill_heights) / len(fill_heights) # not sure what to name this

mean_hole_depth = mean(vertical_hole_depths)
mean_hole_vertical_clusterdness = mean(vertical_hole_clusterdness)
hole_depth_deviation = std(vertical_hole_depths)
hole_vertical_instability = std(vertical_hole_clusterdness) # High std indicates inconsistent strategy

mean_hole_distance = mean(horizontal_hole_distances)
mean_hole_horizontal_clusterdness = mean(horizontal_hole_clusterdness)
hole_distance_deviation = std(horizontal_hole_distances)
hole_horizontal_instability = std(horizontal_hole_clusterdness) # High std indicates inconsistent strategy
```

total_filledness is same as mean_height, so scrap it.


## Small conclusion

Current metrics:

```python
fill_heights = [highest_cell/len(col) for col in cols]
mean_height = mean(fill_heights)
bumpiness = std(fill_heights)
lowest_point = min(fill_heights)
highest_point = max(fill_heights)

hole_height_per_col = hole_height_in_col / (fill_height[col] * len(col)) # multiplied by len(col), since we normalized it. When calculating, probably unnecessary, we can use the unnormalized value probably
vertical_hole_depths = [
    mean(1 - hole_heights_in_col) # 0 if no holes
    for col in cols
]
vertical_hole_clusterdness = [
    1 - 2 * std(hole_heights_in_col) # 0 if no holes
    for col in cols
]

hole_distance_from_middle_per_row = abs(1 - 2 * hole_position_normalized)
horizontal_hole_distances = [
    mean(distance_from_middle) # distance from middle, 0 if no holes
    for row in rows
]
horizontal_hole_clusterdness = [
    1 - 2 * std(hole_position_normalized) # 0 if no holes
    for row in rows
]

mean_hole_depth = mean(vertical_hole_depths)
mean_hole_vertical_clusterdness = mean(vertical_hole_clusterdness)
hole_depth_deviation = std(vertical_hole_depths)
hole_vertical_instability = std(vertical_hole_clusterdness) # High std indicates inconsistent strategy

mean_hole_edge_distance = mean(horizontal_hole_distances)
mean_hole_horizontal_clusterdness = mean(horizontal_hole_clusterdness)
hole_edge_distance_deviation = std(horizontal_hole_distances)
hole_horizontal_instability = std(horizontal_hole_clusterdness) # High std indicates inconsistent strategy
```

## Other metrics

### General holeyness

I actually added some metrics, that combbine the vertical and horizontal metrics, therefore total holeyness, is renamed to general holeyness, where I take the middle column as the middle and the average fill_height as the row. Then do the same calculations as for horizontal, and find the metrics for distance from middle. I also wrote my idea to ChatGPT and it started writing the euclidean distance using vertical depth instead. I like that solution more, because instead of scoring the distance from the center, it scores the distance from the surface.

So we get per hole
```python
vertical_depth = 1 - hole_height_normalized
horizontal_distance = abs(1 - 2 * hole_position_normalized)

hole_distance = sqrt((vertical_depth**2 + horizontal_distance**2) / 2) # divide by 2, since the sum can be 1 + 1 = 2
hole_clustering_metric = sqrt((vertical_depth**2 + hole_position_normalized**2) / 2)

mean_hole_distance = mean(hole_distances)
general_hole_clusterdness = 1 - 2 * std(hole_clustering_metric)
```

Also we probably want to define

```python
total_holeyness = total_holes / sum(highest_cell for col in cols)
```

## Conclusion

We have metrtics

```python
fill_heights = [highest_cell/len(col) for col in cols]

mean_height = mean(fill_heights)
bumpiness = std(fill_heights)
lowest_point = min(fill_heights)
highest_point = max(fill_heights)

hole_height_per_col = hole_height_in_col / (fill_height[col] * len(col)) # multiplied by len(col), since we normalized it. When calculating, probably unnecessary, we can use the unnormalized value probably
vertical_hole_depths = [
    mean(1 - hole_heights_in_col) # 0 if row/col has fewer than 2 holes
    for col in cols
]
vertical_hole_clusterdness = [
    1 - 2 * std(hole_heights_in_col) # 0 if row/col has fewer than 2 holes
    for col in cols
]

hole_distance_from_middle_per_row = abs(1 - 2 * hole_position_normalized)
horizontal_hole_distances = [
    mean(distance_from_middle) # distance from middle, 0 if row/col has fewer than 2 holes
    for row in rows
]
horizontal_hole_clusterdness = [
    1 - 2 * std(hole_position_normalized) # 0 if row/col has fewer than 2 holes
    for row in rows
]

mean_hole_depth = mean(vertical_hole_depths)
mean_hole_vertical_clusterdness = mean(vertical_hole_clusterdness)
hole_depth_deviation = std(vertical_hole_depths)
hole_vertical_instability = std(vertical_hole_clusterdness) # High std indicates inconsistent strategy

mean_hole_edge_distance = mean(horizontal_hole_distances)
mean_hole_horizontal_clusterdness = mean(horizontal_hole_clusterdness)
hole_edge_distance_deviation = std(horizontal_hole_distances)
hole_horizontal_instability = std(horizontal_hole_clusterdness) # High std indicates inconsistent strategy

vertical_depth = 1 - hole_height_normalized
horizontal_distance = abs(1 - 2 * hole_position_normalized)
hole_distance = sqrt((vertical_depth**2 + horizontal_distance**2) / 2) # divide by 2, since the sum can be 1 + 1 = 2
hole_clustering_metric = sqrt((vertical_depth**2 + hole_position_normalized**2) / 2)

total_holeyness = total_holes / sum(highest_cell for col in cols)
mean_hole_distance = mean(hole_distances)
general_hole_clusterdness = 1 - 2 * std(hole_clustering_metric)
```
