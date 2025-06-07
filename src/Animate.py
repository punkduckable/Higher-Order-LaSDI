# -------------------------------------------------------------------------------------------------
# Import and Setup
# -------------------------------------------------------------------------------------------------

from    __future__                  import  annotations
from    pathlib                     import  Path

import  numpy;
import  matplotlib.pyplot           as      plt
from    matplotlib.animation        import  FuncAnimation, FFMpegWriter;





# -------------------------------------------------------------------------------------------------
# movie making function
# -------------------------------------------------------------------------------------------------

def make_solution_movies(   U_True          : numpy.ndarray,
                            U_Pred          : numpy.ndarray,
                            X               : numpy.ndarray,
                            T               : numpy.ndarray,
                            *,
                            save_dir        : str | Path    = ".",
                            fname_prefix    : str           = "solution",
                            fps             : int           = 20,
                            dpi             : int           = 150,
                            cmap            : str           = "viridis") -> tuple[Path, Path, Path]:
    """
    Create three movies visualising a spatio-temporal solution: the true field, the predicted 
    field, and their difference.

    
    -----------------------------------------------------------------------------------------------
    Parameters
    -----------------------------------------------------------------------------------------------

    U_True, U_Pred
        Arrays of shape ``(N_t, 1 or 2, N_x)`` holding the true and predicted
        signal, respectively.  One component → scalar field, two components →
        2-D vector field.
    
    X
        Array of shape ``(2, N_x)``.  Each column gives the *(x, y)* coordinates
        of a sensor point.
    
    T
        1-D array of length ``N_t`` with times for each frame.
    
    save_dir
        Directory in which to write the resulting ``.mp4`` files.
    
    fname_prefix
        Prefix for the filenames (e.g. *prefix*`_True.mp4`).
    
    fps
        Frames per second for the saved movies.
    
    dpi
        Resolution of the rendered frames.
   
    cmap
        Matplotlib colour-map for scalar plots.

    
        
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    (true_path, pred_path, err_path)
        Paths of the three generated movie files.

    

    -----------------------------------------------------------------------------------------------
    Notes
    -----------------------------------------------------------------------------------------------
    
    * Requires **matplotlib** and **FFmpeg**.
    * The function modifies a handful of *rcParams* for cleaner aesthetics
      (larger font, transparent axes spines, gentle grid, prettier colours).
    """
    
    # ---------------------------------------------------------------------------------------------
    # basic checks 
    # ---------------------------------------------------------------------------------------------
    if U_True.shape != U_Pred.shape:
        raise ValueError("U_True and U_Pred must have identical shape")

    N_t, n_comp, N_x = U_True.shape
    if n_comp not in (1, 2):
        raise ValueError("Second dimension of U_* must be 1 (scalar) or 2 (vector)")
    if X.shape != (2, N_x):
        raise ValueError("X must have shape (2, N_x)")
    if T.shape != (N_t,):
        raise ValueError("T must have shape (N_t,)")

    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents = True, exist_ok = True)



    # ---------------------------------------------------------------------------------------------
    # nicer default style 
    # ---------------------------------------------------------------------------------------------
    
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({   "figure.dpi"        : dpi,
                            "font.size"         : 12,
                            "axes.grid"         : True,
                            "grid.alpha"        : 0.3,
                            "axes.spines.top"   : False,
                            "axes.spines.right" : False})



    # ---------------------------------------------------------------------------------------------
    # internal helpers 
    # ---------------------------------------------------------------------------------------------
    
    def _scalar_anim(   data: numpy.ndarray, title: str, fname: str ) -> Path:  # data shape (N_t, N_x)
        vmin, vmax  = data.min(), data.max()
        fig, ax     = plt.subplots()
        scat = ax.scatter(  X[0],
                            X[1],
                            c           =   data[0],
                            cmap        =   cmap,
                            vmin        =   vmin,
                            vmax        =   vmax,
                            s           =   60,
                            linewidths  =   0.4,
                            edgecolors  =   "k")
        ax.set_aspect("equal")
        cb = fig.colorbar(scat, ax = ax)
        cb.set_label(title.replace("\n", " "))
        time_text = ax.set_title(f"{title}\n$t$ = {T[0]:.3f}")

        def update(frame: int):
            scat.set_array(data[frame])
            time_text.set_text(f"{title}\n$t$ = {T[frame]:.3f}")
            return scat, time_text

        ani = FuncAnimation(fig, 
                            update, 
                            frames  = N_t, 
                            blit    = True, 
                            repeat  = False);
        out_path = save_dir / fname
        ani.save(out_path, writer = FFMpegWriter(fps = fps, codec = "libx264"))
        plt.close(fig)
        return out_path



    def _vector_anim(data: numpy.ndarray, title: str, fname: str) -> Path:  # data shape (N_t, 2, N_x)
        # Arrow colour encodes vector magnitude (helps readability)
        magnitudes = numpy.linalg.norm(data, axis = 1)
        vmin, vmax = magnitudes.min(), magnitudes.max()

        fig, ax = plt.subplots()
        q = ax.quiver(
            X[0],
            X[1],
            data[0, 0],
            data[0, 1],
            magnitudes[0],
            cmap            =   cmap,
            clim            =   (vmin, vmax),
            angles          =   "xy",
            scale_units     =   "xy",
            scale           =   1.0,
            width           =   0.007)
        
        ax.set_aspect("equal")
        cb = fig.colorbar(q, ax = ax)
        cb.set_label("|value|")
        time_text = ax.set_title(f"{title}\n$t$ = {T[0]:.3f}")

        def update(frame: int):
            q.set_UVC(data[frame, 0], data[frame, 1], magnitudes[frame])
            time_text.set_text(f"{title}\n$t$ = {T[frame]:.3f}")
            return q, time_text

        ani = FuncAnimation(fig, 
                            update, 
                            frames  = N_t, 
                            blit    = False, 
                            repeat  = False)
        out_path = save_dir / fname
        ani.save(out_path, writer = FFMpegWriter(fps = fps, codec = "libx264"))
        plt.close(fig)
        return out_path



    # ---------------------------------------------------------------------------------------------
    # dispatch based on scalar / vector 
    # ---------------------------------------------------------------------------------------------
    
    if n_comp == 1:
        t_path = _scalar_anim(  U_True[:, 0], 
                                "True field", 
                                f"{fname_prefix}_True.mp4")
        p_path = _scalar_anim(  U_Pred[:, 0], 
                                "Predicted field", 
                                f"{fname_prefix}_Pred.mp4")
        e_path = _scalar_anim(  (U_Pred - U_True)[:, 0],
                                "Prediction error",
                                f"{fname_prefix}_error.mp4")
    
    else:  # 2 components → vector field
        t_path = _vector_anim(  U_True, 
                                "True vector field", 
                                f"{fname_prefix}_True.mp4")
        p_path = _vector_anim(  U_Pred, 
                                "Predicted vector field", 
                                f"{fname_prefix}_Pred.mp4")
        e_path = _vector_anim(  U_Pred - U_True, 
                                "Error vector field", 
                                f"{fname_prefix}_error.mp4")

    return t_path, p_path, e_path



# -------------------------------------------------------------------------------------------------
# Example (commented out – uncomment to test)
# -------------------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     N_t, N_x = 50, 100
#     X = numpy.random.rand(2, N_x) * 2 - 1        # points in [-1,1]²
#     T = numpy.linspace(0, 2 * numpy.pi, N_t)
#
#     # synthetic scalar example
#     U_True = numpy.sin(T)[:, None, None] * numpy.exp(-numpy.linalg.norm(X, axis=0)[None, None, :])
#     noise = 0.05 * numpy.random.randn(*U_True.shape)
#     U_Pred = U_True + noise
#     make_solution_movies(U_True, U_Pred, X, T, fname_prefix="demo_scalar")
#
#     # synthetic vector example
#     U_True_vec = numpy.empty((N_t, 2, N_x))
#     for i, t in enumerate(T):
#         U_True_vec[i, 0] = numpy.cos(t) * X[0] - numpy.sin(t) * X[1]
#         U_True_vec[i, 1] = numpy.sin(t) * X[0] + numpy.cos(t) * X[1]
#     U_Pred_vec = U_True_vec + 0.1 * numpy.random.randn(*U_True_vec.shape)
#     make_solution_movies(U_True_vec, U_Pred_vec, X, T, fname_prefix="demo_vector")