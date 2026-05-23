# Features

Previously, we removed the old "bumpiness" metric. After looking at how it plays, I understood one thing:

It seems to try to get line-piece holes (even though it doesn't have a direct visual input, it seems to be able to infer it from the 95 features it has), which is a common strategy in tetris. But there is also a lot of noise. I think it evaluates this noise also as those lines.

Therefore, I want to introduce a new metric: "bumpiness", but I rename it to "height_diff". This would allow it to better understand what is going on.