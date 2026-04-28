This code evaluates some aspects of the paper Magnetic reconnection mediated by hyper-resistive plasmoid instability  by Yi-Min Huang, A. Bhattacharjee, and Terry G. Forbes

Report submitted via google doc: https://docs.google.com/document/d/1_LTZVEtPM8AIFoO3AqhFmZ-xdfRqUbaZDCGLqH0tk9g/edit?usp=sharing

I'd recommend making a new conda env like this:
conda create -n mhx_env python=3.10 -y

conda activate mhx_env

conda install -c conda-forge numpy matplotlib scipy -y

pip install "jax[cuda12]"

####install MHX in editable mode####

conda install -c conda-forge conda-pack -y

conda pack -n mhx_env -o mhx_env.tar.gz



Then, move this tarball to your CHTC dir, alongside run_mhx.sh

To run "run_plasmoid.py", move run_plasmoid.py and run_plasmoid.sub into your CHTC dir, and run it with condor_submit run_plasmoid.sub
scp out history.npz, and the png it creates, move history.npz into your MHX directory, and then run:
python scripts/legacy/mhd_tearing_island_evolution.py     --input history.npz     --make-movie

To run "sweep_ky.py", move sweep_ky.py and sweep_ky.sub into your CHTC dir, and run it with condor_submit sweep_ky.sub
scp out the outputs, and you are done

To run "linear.py", configure which of the 3 runs you want to do (Nx, Ny) = (64,64), (64,128),or (128,128) in your python file by changing the inputs at cfg = TearingSimConfig, and then just run on cpu.
