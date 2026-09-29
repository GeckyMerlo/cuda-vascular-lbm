# References

These sources informed the project design and the interpretation of the model.
They are background references, not evidence that this implementation reproduces
every method described in the papers.

## Blood-flow and particle modeling

- C. Sun and L. L. Munn, [“Lattice Boltzmann simulation of blood flow in digitized vessel networks”](https://pubmed.ncbi.nlm.nih.gov/19343080/), 2008: an example of LBM applied to vascular flow.
- F. Janoschek, F. Toschi, and J. Harting, [“Simulations of blood flow in plain cylindrical and constricted vessels with single cell resolution”](https://onlinelibrary.wiley.com/doi/10.1002/mats.201100013), 2011: background for particulate blood-flow modeling in confined vessels.
- T. Wang and Z. Xing, [“A fluid-particle interaction method for blood flow with special emphasis on red blood cell aggregation”](https://doi.org/10.3233/BME-141065), 2014: background on coupled fluid-particle blood models.

## Numerical methods

- [Lattice Boltzmann methods](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods): introductory overview of the method and its macroscopic moments.
- [Stokes drag](https://en.wikipedia.org/wiki/Drag_%28physics%29#Very_low_Reynolds_numbers:_Stokes'_drag): background for the low-Reynolds-number drag law used by the particle model.
- [lbmpy boundary conditions](https://pycodegen.pages.i10git.cs.fau.de/lbmpy/sphinx/boundary_conditions.html): reference material used while comparing LBM boundary treatments.
- M. Schwarz and H.-P. Seidel, [“Fast Parallel Surface and Solid Voxelization on GPUs”](https://michael-schwarz.com/research/publ/2010/vox/), ACM Transactions on Graphics 29(6), 2010: source cited in the presentation for triangle-box voxelization.

## Project material

- [Final presentation](HESP-Project-Presentation.pptx)
- [Model and mathematics](MODEL_AND_MATHEMATICS.md)
- [Experimental results](RESULTS.md)
- [Particle implementation notes](../PARTICLE_SYSTEM.md)
