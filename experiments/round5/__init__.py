"""Round 5 experiments: principled fixes for calibrated and miscalibrated maps.

Importing the package pins the BLAS libraries to one thread, so that the parallel workers
of the drivers do not oversubscribe the cores.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
