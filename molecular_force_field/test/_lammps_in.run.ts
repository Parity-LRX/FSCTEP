# Matched A/B: same model as core_real663.pt2 but as a TorchScript core (energy-only,
# force via C++ autograd) -> drives the engine's eager + cuda-graph paths. Identical
# system/cutoff/freeze to in.run.aoti so PE and timing are directly comparable.
units metal
atom_style atomic
boundary p p p

region box block 0 30 0 30 0 30
create_box 2 box
create_atoms 1 random 150 12345 box
create_atoms 2 random 100 12346 box
mass 1 1.008
mass 2 15.999

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch 6.0 cuda
pair_coeff * * /home/ylzhang/lrx/mff_md_work/core_real_ts.pt H O

fix 1 all nve
fix 2 all setforce 0.0 0.0 0.0
thermo 20
thermo_style custom step temp pe etotal
run 0
run 100
