# -------------------------------------------------------------------------------------------------
# Import and Setup
# -------------------------------------------------------------------------------------------------

from    __future__                  import  annotations;
from    pathlib                     import  Path;
import  os;

import  numpy;
import  matplotlib.pyplot           as      plt;
from    matplotlib.animation        import  FuncAnimation, FFMpegWriter;





# ---------------------------------------------------------------------------------------------
# internal helpers 
# ---------------------------------------------------------------------------------------------

def _scalar_anim(   data        : numpy.ndarray,
                    title       : str,
                    fname       : str,
                    X           : numpy.ndarray,
                    T           : numpy.ndarray,
                    save_dir    : Path          = Path("."),
                    fps         : int           = 20,
                    dpi         : int           = 150,
                    cmap        : str           = "viridis") -> Path:  # data shape (N_t, N_x)
    """
    Create an MP4 showing the evolution of a **scalar** field sampled on a
    point cloud.



    -------------------------------------------------------------------------------------------
    Arguments
    -------------------------------------------------------------------------------------------

    data : ndarray, shape (N_t, N_x)
        Scalar values at each sensor for every time step.  
        ``data[i, j]`` corresponds to time *``T[i]``* and position *``X[:, j]``*.
    
    title : str
        Text for the figure title & colour-bar label.
    
    fname : str
        File name (without directory) of the resulting movie.
    
    X : ndarray, shape (2, N_x), optional
        Sensor coordinates.
    
    T : ndarray, shape (N_t,), optional
        Time stamps. 
    
    save_dir : pathlib.Path, default ``Path('.')``
        Directory in which the movie is written.
    
    fps : int, default 20
        Frames per second.
    
    dpi : int, default 150
        Dots-per-inch for the figure canvas.
    
    cmap : str, default ``'viridis'``
        Matplotlib colour-map used to encode the scalar amplitude.

        

    -------------------------------------------------------------------------------------------
    Returns
    -------------------------------------------------------------------------------------------
    
    pathlib.Path
        Absolute path of the saved MP4.

    
    -------------------------------------------------------------------------------------------
    Notes
    -------------------------------------------------------------------------------------------

    * Uses :class:`matplotlib.animation.FuncAnimation` together with an
    :class:`matplotlib.animation.FFMpegWriter`.  A working **FFmpeg**
    installation must be on ``$PATH``.
    * Colours are normalised globally (``vmin``, ``vmax`` from *all* frames)
    so that colour is comparable across time.
    """
    
    """
    data : numpy.ndarray, shape = (N_t, N_x)
        An array whose i,j element holds the value we want to plot at the j'th position in the
        i'th frame.

    title : str
        The title for the movie we make
    
    fname : str
        The name of the file where we want to save the animation.
    """

    N_t         : int   = T.shape[0];

    # Determine axis scales. 
    vmin, vmax  = data.min(), data.max();

    # Make the plot.
    fig, ax     = plt.subplots()
    scat = ax.scatter(  X[0],
                        X[1],
                        c           =   data[0],
                        cmap        =   cmap,
                        vmin        =   vmin,
                        vmax        =   vmax,
                        s           =   15,
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
    out_path = os.path.join(save_dir, fname);
    print(out_path);
    print(save_dir / fname);
    ani.save(out_path, writer = FFMpegWriter(fps = fps, codec = "libx264"));
    plt.close(fig);

    # All done!
    return out_path;



def _vector_anim(   data        : numpy.ndarray,
                    title       : str,
                    fname       : str,
                    X           : numpy.ndarray,
                    T           : numpy.ndarray,
                    save_dir    : Path          = Path("."),
                    fps         : int           = 20,
                    dpi         : int           = 150,
                    cmap        : str           = "viridis") -> Path:
    """
    Create an MP4 showing the evolution of a **2-D vector** field sampled
    on a point cloud.

    
    
    -------------------------------------------------------------------------------------------
    Arguments
    -------------------------------------------------------------------------------------------

    data : ndarray, shape (N_t, 2, N_x)
        Vector values (``u, v`` components) for every time step and sensor.
    
    title : str
        Text for the figure title & colour-bar label.
    
    fname : str
        File name (without directory) of the resulting movie.
    
    X : ndarray, shape (2, N_x), optional
        Sensor coordinates. 
    
    T : ndarray, shape (N_t,), optional
        Time stamps.
    
    save_dir : pathlib.Path, default ``Path('.')``
        Directory in which the movie is written.
    
    fps : int, default 20
        Frames per second.
    
    dpi : int, default 150
        Dots-per-inch for the figure canvas.
    
    cmap : str, default ``'viridis'``
        Matplotlib colour-map used to encode arrow magnitude.

        
    
    -------------------------------------------------------------------------------------------
    Returns
    -------------------------------------------------------------------------------------------
    
    pathlib.Path
        Absolute path of the saved MP4.

    
    
    -------------------------------------------------------------------------------------------
    Notes
    -------------------------------------------------------------------------------------------
    
    * Arrow colour represents vector magnitude (‖(u, v)‖) so that both
    direction and strength are visible.
    * As for `_scalar_anim`, FFmpeg must be available.
    """
    
    # Arrow colour encodes vector magnitude (helps readability)
    magnitudes          = numpy.linalg.norm(data, axis = 1);
    vmin                = magnitudes.min();
    vmax                = magnitudes.min();
    N_t         : int   = T.shape[0];

    # Make a quiver plot using the data
    fig, ax = plt.subplots();
    q = ax.quiver(  X[0],
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
    
    ax.set_aspect("equal");
    cb = fig.colorbar(q, ax = ax);
    cb.set_label("|value|");
    time_text = ax.set_title(f"{title}\n$t$ = {T[0]:.3f}");

    def update(frame: int):
        q.set_UVC(data[frame, 0], data[frame, 1], magnitudes[frame]);
        time_text.set_text(f"{title}\n$t$ = {T[frame]:.3f}");
        return q, time_text;

    ani = FuncAnimation(fig, 
                        update, 
                        frames  = N_t, 
                        blit    = False, 
                        repeat  = False);
    out_path = save_dir / fname;
    ani.save(out_path, writer = FFMpegWriter(fps = fps, codec = "libx264"));
    plt.close(fig);

    # All done!
    return out_path;



# -------------------------------------------------------------------------------------------------
# movie making function
# -------------------------------------------------------------------------------------------------

def make_solution_movies(   U_True          : numpy.ndarray,
                            U_Pred          : numpy.ndarray,
                            X               : numpy.ndarray,
                            T               : numpy.ndarray,
                            save_dir        : str | Path    = "../Figures/",
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

    U_True, U_Pred : numpy.ndarray, shape = (N_t, 1, N_x) or (N_t, 2, N_x)
        Arrays of shape holding the true and predicted signal, respectively. If they have shape 
        (N_t, 1, N_x) then the solution should be a scalar field. If it is 2 then the solution 
        should be a 2-D vector field. 

    X : numpy.ndarray, shape = (2, N_x)
        Each column gives the (x, y) coordinates of a sensor point.
    
    T : numpy.ndarray, shape = (N_t)
        i'th element holds the value of the i'th time step.

    save_dir : str
        Directory in which to write the resulting ``.mp4`` files.
    
    fname_prefix : str
        Prefix for the filenames (e.g. *prefix*`_True.mp4`).
    
    fps : int
        Frames per second for the saved movies.
    
    dpi : int
        Resolution of the rendered frames.
   
    cmap : str
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

    save_dir = Path(save_dir).expanduser().resolve();
    save_dir.mkdir(parents = True, exist_ok = True);



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
    # dispatch based on scalar / vector 
    # ---------------------------------------------------------------------------------------------

    if n_comp == 1:
        t_path = _scalar_anim(  data        = U_True[:, 0, :], 
                                title       = "True field", 
                                fname       = f"{fname_prefix}_True.mp4",
                                save_dir    = save_dir,
                                X           = X,
                                T           = T);
        
        p_path = _scalar_anim(  data        = U_Pred[:, 0, :], 
                                title       = "Predicted field", 
                                fname       = f"{fname_prefix}_Pred.mp4",
                                save_dir    = save_dir,
                                X           = X,
                                T           = T);
        e_path = _scalar_anim(  data        = (U_Pred - U_True)[:, 0, :],
                                title       = "Prediction error",
                                fname       = f"{fname_prefix}_error.mp4",
                                save_dir    = save_dir,
                                X           = X,
                                T           = T);
    
    else:  # 2 components → vector field
        t_path = _vector_anim(  data        = U_True, 
                                title       = "True vector field", 
                                fname       = f"{fname_prefix}_True.mp4",
                                save_dir    = save_dir,
                                X           = X,
                                T           = T);
        p_path = _vector_anim(  data        = U_Pred, 
                                title       = "Predicted vector field", 
                                fname       = f"{fname_prefix}_Pred.mp4",
                                save_dir    = save_dir,
                                X           = X,
                                T           = T);
        e_path = _vector_anim(  data        = U_Pred - U_True, 
                                title       = "Error vector field", 
                                fname       = f"{fname_prefix}_error.mp4",
                                save_dir    = save_dir,
                                X           = X,
                                T           = T);

    return t_path, p_path, e_path
