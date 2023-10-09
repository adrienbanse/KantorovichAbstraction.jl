# KantorovichAbstraction.jl

This repository contains all the code necessary to implement the experiments in 
- [BRAJ23] Adrien Banse, Licio Romao, Alessandro Abate, Raphaël M. Jungers - <em>Data-driven abstractions via adaptive refinements and a Kantorovich metric</em> - Online [here](https://adrienbanse.github.io/assets/pdf/cdc23_extended.pdf)

<p align="center">
  <img width="774" alt="Screenshot 2023-04-07 at 11 46 05" src="https://user-images.githubusercontent.com/45042779/230587083-0f890634-12e6-4857-b3c6-42a1be0512c6.png">  
</p>
<p align="center">[BRAJ23, Fig. 3.] <em>Illustration of the execution of [BRAJ23, Algorithm 2].</em></p>

## Content

<code>src/Kantorovich.jl</code> module is composed of the following files:
- <code>src/metric.jl</code> - contains an efficient implementation of the Kantorovich metric [BRAJ23, Algorithm 1], as well as structures for Markov chains
- <code>src/abstraction.jl</code> - contains an implementation of [BRAJ23, Algorithm 2]
- <code>src/system.jl</code> - contains structures for dynamical systems
- <code>src/utils.jl</code> - contains util functions

And <code>experiments/refine_memory_example.ipynb</code> is a notebook implementing [BRAJ23, Section III-C], and recovers all the numerical values exposed in this section.

<p align="center">
  <img width="813" alt="Screenshot 2023-04-07 at 11 44 55" src="https://user-images.githubusercontent.com/45042779/230586888-60317fbb-629d-4496-b734-44e8f4474ab8.png">
</p>
<p align="center">[BRAJ23, Fig. 5.] <em>Illustration of the last partitioning given by [BRAJ23, Algorithm 2] for [BRAJ23, Example 1].</em></p>
