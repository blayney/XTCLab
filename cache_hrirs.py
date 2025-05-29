# build_hrir_cache.py
import numpy as np
import xtc

# pick the sofa file and all azimuths you’ll ever need
sofa_file = "P0275_FreeFieldComp_48kHz.sofa"
azs = np.arange(0, 360, 5)  # or whatever rounding you use

cache = {"left_az": [], "right_az": [], "HRIR_LL": [], "HRIR_LR": [], "HRIR_RL": [], "HRIR_RR": []}
for az in azs:
    ll, lr, _ = xtc.extract_hrirs_sam(sofa_file,  az)   # left speaker
    rl, rr, _ = xtc.extract_hrirs_sam(sofa_file, -az)   # right speaker (or wrap to 0–360)
    cache["left_az"].append(az)
    cache["right_az"].append(az)
    cache["HRIR_LL"].append(ll)
    cache["HRIR_LR"].append(lr)
    cache["HRIR_RL"].append(rl)
    cache["HRIR_RR"].append(rr)

# save everything in one go
np.savez("hrir_cache.npz", **cache)