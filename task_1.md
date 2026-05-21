# Task 1 - Gameplay flow / Game loop

## 1

Most of the logic is pretty straight forward. The first interesting part comes with the logic of `get_possible_actions()`

An action is `(rotation_index, column)`

It uses method `_check_hard_drop_valid(self, shape, col)` which from the naming seems that it checks if it can hard drop there, but in reality, it just checks that if the action's starting position is inside the game area horisontally.

Actually, it is not very complicated, but I had the wrong idea what it does at first and it took quite some time to understand what it is doing because of that.

## 2

The second most important part is what are we training.

Reading what is being done, is that we generate all possible actions, and those act as an input.

The agent's job is to evaluate these states and choose the best state.

The core principle of the agent:

1. evaluate the states
2. output probability distribution of the states
3. the highest probability is given to the state that is supposedly the best

## 3 Some conclusions/things that caught my eye

A piece can have at most `4` unique rotations.

With `4` unique rotations, the piece must be at least `3` pieces wide. Take for example:

    _ x _
    x _ x
where _ is empty and x is piece.

Then it has the following rotations:

    _ _ _    x _ _    x _ x    _ _ x
    _ x _    _ x _    _ x _    _ x _
    x _ x    x _ _    _ _ _    _ _ x
2 of these take up 3 pieces while 2 take up only 2.

Therefore, the total amount of options is maximally:

    2 * (cols - 2) + 2 * (cols - 1) =
    = 4 * cols - 6

In the case of this, if we have a piece with an even length, then the center would have to be one of the corners. Therefore, removing one corner of the 3 piece tetrominoe, will give us a 2 piece tetrominoe, which takes 2 length everywhere. Example:

    _ _ _    _ _ _    x _ _    _ _ x
    _ x _    _ x _    _ x _    _ x _
    _ _ x    x _ _    _ _ _    _ _ _
Therefore, the total amount of options is maximally:

    2 * (cols - 1) + 2 * (cols - 1) =
    = 4 * cols - 4

Actually, we can reduce this again:

    _ _ _    _ _ _    _ x _    _ _ _
    _ x _    x x _    _ x _    _ x x
    _ x _    _ _ _    _ _ _    _ _ _

Although, the last solution would be more likely defined as 2 positons, and the same for the one before it, we should still consider this option. The amount of options is maximally:

    2 * cols + 2 * (cols - 1) =
    = 4 * cols - 2

Theoretically, the following is possible:

    _ _ _    _ _ _    _ x _    _ _ _
    _ _ _    x _ _    _ _ _    _ _ x
    _ x _    _ _ _    _ _ _    _ _ _
which would make the final amount of options maximally:
    
    4 * cols

This leads me away from my first idea. Initially, I thought I would make a model, which takes all the possible inputs (if there are not max amount, just use padding values, similarly to language models), and outputs a probability map. The latter is also the motivation for this process I just had, but is not very dynamic. Ideally, I want something dynamic. A model, where I can use varying piece types and counts, and maybe even varying board sizes (not sure yet).

How would I accomplish the latter?

Ideas:
1. Output a score based on board. Most simple, learn to score boards, but this solution doesn't have context. It learns to score boards, but if there are 2 good boards infront of it, it might've learned to score them high, but not relatively correctly (the worse of the 2 gets a higher score). I think it would make a good baseline, as it learns what is good and what is bad, but that is pretty simple.

2. Tournament style: Takes in all boards, divides into groups (depending on input layer size), and then for each group calculates relative scores (which board it thinks is best) (this is like probability distribution). Then puts together a new input from the best performers of each distribution, and compares again. Finally, we can comparatively choose the best option. This allows for dynamic input counts, but I don't think dynamic column and row counts.

I like the second idea quite a bit, but I will see how things go and what I implement.