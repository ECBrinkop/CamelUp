Camel Up Decision Engine
A numerical decision-support engine for the board game Camel Up, computing the expected payoff of every available move given any game state. Supports both the classic 5-camel variant and the extended 7-camel variant.
What it does
Camel Up involves 5 (or 7) camels racing around a board in rounds. Each round, each camel moves once. When camels land on the same tile they stack, causing fast-skip chain movements that create complex, path-dependent probability distributions over race outcomes.
This engine takes a full game state as input — current board positions, camel stacks, player inventories, and available moves — and returns the expected payoff for every legal move. It does this by enumerating and simulating:

Up to 300,000 paths per field composition to compute round and race outcome probabilities
Up to 7,500,000 additional paths across alternative field positions to evaluate leg and spectator card placements

The engine is accelerated using Numba JIT compilation, reducing computation time by orders of magnitude over pure Python implementations.
Why this is interesting
The stacking mechanic in Camel Up creates a combinatorial explosion in possible game states that makes analytical solutions intractable beyond a few moves. The engine sidesteps this by exhaustive path simulation with performance-critical inner loops compiled to machine code — the same approach used in production quantitative systems for pricing path-dependent derivatives.
Contents

CamelUp.py — core simulation and expected value computation engine
CamelUpScenarios.ipynb — worked analysis of common 7-camel starting positions, with optimal move sequences for the first 4 moves of several frequently-encountered field configurations
Some extra files containing additional Experiments and Tests

Getting started
The CamelUpScenarios notebook is the best entry point. It is fully self-contained and walks through several starting positions with commentary. All examples in the notebook are operational and can be modified to analyse custom game states.
Dependencies: Python 3.x, NumPy, Numba, Jupyter
bashpip install numpy numba jupyter
jupyter notebook CamelUpScenarios.ipynb
Background
Built as a personal project to apply expected value reasoning and high-performance numerical computing to a combinatorial game. The engine has been used in live play — opponents have yet to win.
