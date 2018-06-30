from .version import __version__
__all__ = ["mps_gen", "mps_uniform", "mps_sandwich", "tdvp_gen", "tdvp_uniform", 
           "tdvp_sandwich", "mps_uniform_pinv", "mps_uniform_excite", "dynamics",
           "core_common", "tdvp_common", "tdvp_calc_C", "sexpmv",
           "cuda_alternatives",
           "allclose", "matmul", "nullspace", "version"]

import logging

logging.basicConfig(level=logging.INFO)  # This is probably not the right
# place to do the logger configuration, but I  figure it's better to have a
# logger configuration by default, that one can overwrite by calling
# basicConfig before loading the module.
