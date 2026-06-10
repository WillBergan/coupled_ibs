# coupled_ibs
Simple code for calculating IBS growth rates for a lattice with coupling.

This implements the method developed by V. Lebedev and S. Nagaitsev in "Multiple Intrabeam Scattering in x-y Coupled Focusing Systems," arXiv:1812.09275v5

Run with:

python ibs_with_coupling_many_hsr.py hsr_cs_params_1k.txt

The input file is for an uncoupled lattice, so we add in coupling artificially as a test.
