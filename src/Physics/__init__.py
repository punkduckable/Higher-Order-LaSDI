"""
Physics models and full-order problem wrappers for LaSDI.

`Physics` defines the common full-order model interface. `Burgers`, `BurgersSecondOrder`,
`Burgers2D`, `Explicit`, `ExplicitSecondOrder`, and `Thermal` are importable without MFEM.
`Advection`, `WaveEquation`, `KleinGordon`, `Telegraphers`, and `NonlinearElasticity` wrap the
optional PyMFEM solvers and should be imported directly when the `pymfem` extra is available.
"""

from    importlib               import  import_module;

from    .Physics                import  Physics;

_MODULE_EXPORTS : dict[str, str] = {    "Burgers"               : ".Burgers",
                                        "BurgersSecondOrder"    : ".BurgersSecondOrder",
                                        "Burgers2D"             : ".Burgers2D",
                                        "Explicit"              : ".Explicit",
                                        "ExplicitSecondOrder"   : ".ExplicitSecondOrder",
                                        "Thermal"               : ".Thermal"};

__all__ = [    "Physics",
               "Burgers",
               "BurgersSecondOrder",
               "Burgers2D",
               "Explicit",
               "ExplicitSecondOrder",
               "Thermal"];


def __getattr__(name : str):
    if(name in _MODULE_EXPORTS):
        module = import_module(_MODULE_EXPORTS[name], __name__);
        globals()[name] = module;
        return module;

    raise AttributeError("module %s has no attribute %s" % (__name__, name));
