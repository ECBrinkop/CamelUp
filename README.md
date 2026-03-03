<h1 align="center">Camel Up Decision Engine</h1>

<p align="center">
A numerical decision-support engine for the board game <strong>Camel Up</strong>,<br>
computing the expected payoff of every available move given any game state.
</p>

<p align="center">
Supports both the classic 5-camel variant and the extended 7-camel variant.
</p>

<hr>

<h2>Overview</h2>

<p>
<em>Camel Up</em> involves 5 (or 7) camels racing around a board in rounds. Each round, every camel moves once. When camels land on the same tile, they stack — creating fast-skip chain movements and highly path-dependent race dynamics.
</p>

<p>
The stacking mechanic induces complex probability distributions over round and race outcomes, making analytical solutions intractable beyond a few moves.
</p>

<p>
This engine takes a full game state as input — including:
</p>

<ul>
  <li>Current board positions</li>
  <li>Camel stack configurations</li>
  <li>Player inventories</li>
  <li>Available legal moves</li>
</ul>

<p>
It returns the <strong>expected payoff for every legal move</strong>.
</p>

<hr>

<h2>Computation Methodology</h2>

<p>
Expected values are computed via exhaustive path enumeration and simulation:
</p>

<ul>
  <li>Up to <strong>300,000 simulated paths</strong> per field composition to compute round and race outcome probabilities</li>
  <li>Up to <strong>7,500,000 additional simulated paths</strong> across alternative field positions to evaluate leg and spectator card placements</li>
</ul>

<p>
The engine is accelerated using <strong>Numba JIT compilation</strong>, reducing computation time by orders of magnitude compared to pure Python implementations.
</p>

<p>
This mirrors the approach used in production quantitative systems for pricing path-dependent derivatives — performance-critical inner loops are compiled to machine code.
</p>

<hr>

<h2>Why This Is Interesting</h2>

<p>
The stacking mechanic creates a combinatorial explosion in reachable game states. Closed-form probability derivation quickly becomes infeasible.
</p>

<p>
By combining exhaustive simulation with high-performance numerical acceleration, the engine provides tractable and precise expected value estimates in a setting that would otherwise be analytically unwieldy.
</p>

<hr>

<h2>Repository Structure</h2>

<pre>
CamelUp.py               — Core simulation and expected value computation engine
CamelUpScenarios.ipynb   — Worked analysis of common 7-camel starting positions,
                           including optimal move sequences for the first four moves
Additional files         — Experiments and test cases
</pre>

<hr>

<h2>Getting Started</h2>

<p>
The <strong>CamelUpScenarios.ipynb</strong> notebook is the recommended entry point.
</p>

<ul>
  <li>Fully self-contained</li>
  <li>Includes commentary and worked examples</li>
  <li>All examples are executable and can be modified to analyse custom game states</li>
</ul>

<h3>Dependencies</h3>

<ul>
  <li>Python 3.x</li>
  <li>NumPy</li>
  <li>Numba</li>
  <li>Jupyter Notebook</li>
</ul>

<h3>Installation</h3>

<pre><code>pip install numpy numba jupyter</code></pre>

<p>Then launch:</p>

<pre><code>jupyter notebook CamelUpScenarios.ipynb</code></pre>

<hr>

<h2>Background</h2>

<p>
Built as a personal project to apply expected value reasoning and high-performance numerical computing to a combinatorial board game.
</p>

<p>
The engine has been used in live play — opponents have yet to win.
</p>
