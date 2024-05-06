# prepare dataset
cp -r /public/dsy/AtrialSeg ~/AtrialSeg

# Unpack environment into directory `xm`
mkdir -p xm_pack
tar -xzf /public/dsy/xm.tar.gz -C xm_pack

# Use Python without activating or fixing the prefixes. Most Python
# libraries will work fine, but things that require prefix cleanups
# will fail.
./xm_pack/bin/python

# Activate the environment. This adds `xm/bin` to your path
source xm_pack/bin/activate

# Deactivate the environment
# source xm_pack/bin/deactivate

# # Cleanup prefixes from in the active environment.
# # Note that this command can also be run without activating the environment
# # as long as some version of Python is already installed on the machine.
# # 'pip install conda-pack' first
# conda-unpack
