"""
Plotting, metric, and animation helpers for LaSDI experiments.

`Animate` builds 2D scalar/vector solution movies. `Metrics` computes heatmap data and melt-pool
dimensions. `Plot` draws melt-pool summaries, latent trajectories, heatmaps, and train-space
relative-error diagnostics from trainer and interpolation outputs.
"""

from    importlib               import  import_module;

_EXPORTS : dict[str, tuple[str, str]] = {
    "make_solution_movies"                 : (".Animate", "make_solution_movies"),
    "Animate_2D_Grid_Scalar"               : (".Animate", "Animate_2D_Grid_Scalar"),
    "Generate_Heatmap_Data"                : (".Metrics", "Generate_Heatmap_Data"),
    "Compute_Meltpool_Dimensions"          : (".Metrics", "Compute_Meltpool_Dimensions"),
    "Plot_Meltpool_Dimensions"             : (".Plot", "Plot_Meltpool_Dimensions"),
    "Plot_Latent_Trajectories"             : (".Plot", "Plot_Latent_Trajectories"),
    "Plot_Heatmap"                         : (".Plot", "Plot_Heatmap"),
    "trainSpace_RelativeErrors_Heatmap"    : (".Plot", "trainSpace_RelativeErrors_Heatmap")};

__all__ = [    "make_solution_movies",
               "Animate_2D_Grid_Scalar",
               "Generate_Heatmap_Data",
               "Compute_Meltpool_Dimensions",
               "Plot_Meltpool_Dimensions",
               "Plot_Latent_Trajectories",
               "Plot_Heatmap",
               "trainSpace_RelativeErrors_Heatmap"];


def __getattr__(name : str):
    if(name in _EXPORTS):
        module_name, attr_name = _EXPORTS[name];
        value = getattr(import_module(module_name, __name__), attr_name);
        globals()[name] = value;
        return value;

    raise AttributeError("module %s has no attribute %s" % (__name__, name));
