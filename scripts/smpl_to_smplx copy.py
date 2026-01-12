# python
import numpy as np
p = "/home/msi/Desktop/AMASS/raw/CMU/22_23_justin/22_18_poses.npz"
with np.load(p, allow_pickle=True) as data:
    keys = list(data.keys())
    print(keys)
    for k in keys:
        v = data[k]
        try:
            print(k, getattr(v, "shape", None), type(v))
        except Exception:
            print(k, type(v))