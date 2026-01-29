# -------------------------------------------------------------------------------------------------
# Import and Setup
# -------------------------------------------------------------------------------------------------

from    __future__                  import  annotations;
from    pathlib                     import  Path;
import  os;

import  numpy;
import  matplotlib.pyplot           as      plt;
from    matplotlib.animation        import  FuncAnimation, FFMpegWriter;
from    matplotlib.colors           import  LinearSegmentedColormap;
from    mpl_toolkits.mplot3d        import  Axes3D;  # noqa: F401 (registers 3D projection)

# Set up logging.
import  logging;
LOGGER = logging.getLogger(__name__);



# ---------------------------------------------------------------------------------------------
# internal helpers 
# ---------------------------------------------------------------------------------------------

def _scalar_anim(   data        : numpy.ndarray,
                    title       : str,
                    fname       : str,
                    X           : numpy.ndarray,
                    T           : numpy.ndarray,
                    save_dir    : Path          = Path("."),
                    fps         : int           = 30,
                    dpi         : int           = 150,
                    cmap        : str           = "viridis") -> Path:  # data shape (N_t, N_x)
    """
    Create an MP4 showing the evolution of a **scalar** field sampled on a point cloud in 
    2d or 3d domain.



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
    
    X : ndarray, shape (2, N_x) or (3, N_x)
        Sensor coordinates. Can be a 2d or 3d array.
    
    T : ndarray, shape (N_t,)
        Time stamps. 
    
    save_dir : pathlib.Path, default ``Path('.')``
        Directory in which the movie is written.
    
    fps : int, default 30
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

    # Checks
    assert isinstance(data, numpy.ndarray), "type(data) = %s" % str(type(data));
    assert isinstance(X, numpy.ndarray),    "type(X) = %s" % str(type(X));
    assert isinstance(T, numpy.ndarray),    "type(T) = %s" % str(type(T));
    assert len(data.shape) == 2,            "data.shape = %s" % str(data.shape);
    assert len(X.shape) == 2,               "X.shape = %s" % str(X.shape);
    assert len(T.shape) == 1,               "T.shape = %s" % str(T.shape);
    assert data.shape[0] == T.shape[0],     "data.shape = %s, T.shape = %s" % (str(data.shape), str(T.shape));
    assert data.shape[1] == X.shape[1],     "data.shape = %s, X.shape = %s" % (str(data.shape), str(X.shape));
    assert X.shape[0] in (2, 3),            "X.shape = %s" % str(X.shape);
    assert T.shape[0] == data.shape[0],     "T.shape = %s, data.shape = %s" % (str(T.shape), str(data.shape));

    # Setup.
    N_t         : int   = T.shape[0];
    vmin                = data.min();
    vmax                = data.max();

    # Ensure output directory exists.
    save_dir = Path(save_dir);
    save_dir.mkdir(parents = True, exist_ok = True);

    # Create a new figure/axes object (2D or 3D depending on X).
    if(X.shape[0] == 2):
        fig, ax = plt.subplots();
    else:
        fig = plt.figure();
        ax = fig.add_subplot(111, projection = "3d");

    # Plot the initial frame of the data as a filled triangular contour plot
    if(X.shape[0] == 2):
        scat = ax.scatter(  X[0],                   # array of x coordinates for the triangulation
                            X[1],                   # array of y coordinates for the triangulation
                            c           = data[0],  # data values at those (x,y) positions for the first time slice
                            cmap        = cmap,     # what colormap we use
                            vmin        = vmin,     # lower bounds for color scaling
                            vmax        = vmax,     # upper bounds for color scaling
                            s           = 60);      # Area of the markers.
    else:
        scat = ax.scatter3D(X[0],                   # array of x coordinates for the triangulation
                            X[1],                   # array of y coordinates for the triangulation
                            X[2],                   # array of z coordinates for the triangulation
                            c           = data[0],  # data values at those (x,y,z) positions for the first time slice
                            cmap        = cmap,     # what colormap we use
                            vmin        = vmin,     # lower bounds for color scaling
                            vmax        = vmax,     # upper bounds for color scaling
                            s           = 20);      # Area of the markers.
   
    # Force equal aspect (2D) / box aspect (3D) if available.
    if X.shape[0] == 2:
        ax.set_aspect("equal");
    else:
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1));

    # Add a colorbar to the figure, linked to the contour collection `scat`
    cb = fig.colorbar(scat, ax = ax);

    # Label the colorbar, using the provided title (strip out any newlines)
    cb.set_label(title.replace("\n", " "));

    # Set the initial title of the axes, including the time at T[0]
    time_text = ax.set_title(f"{title}\n$t$ = {T[0]:.3f}");

    def update(frame: int):
        """
        This function will be called for each frame of the animation.
        - frame : the current frame index (0 <= frame < N_t)
        Inside, we:
        1. Update the contour data to the new time slice
        2. Update the title to display the new time
        3. Return the updated artists so FuncAnimation knows what to redraw
        """
        
        # Replace the contour array with the new frame's data
        scat.set_array(data[frame]);
        
        # Update the title text to show the current time
        time_text.set_text(f"{title}\n$t$ = {T[frame]:.3f}");
        
        return scat, time_text;

    # Create the animation
    ani = FuncAnimation(fig,                # the Figure object to animate
                        update,             # the function that draws each frame
                        frames  = N_t,      # total number of frames (N_t)
                        blit    = (X.shape[0] == 2),  # blitting is unreliable for 3D artists
                        repeat  = False);   # whether the animation should loop when it reaches the end

    # Build the full output path.
    out_path = save_dir / fname;

    # Save the animation to a file using ffmpeg
    ani.save(   str(out_path),
                writer      =   FFMpegWriter(   fps = fps,              # frames per second
                                                codec = "libx264"));    # video codec (libx264 for H.264)

    # Close the figure to free up memory.
    plt.close(fig);

    # Return the path to the saved animation file
    return out_path;



def _vector_anim(   data        : numpy.ndarray,
                    title       : str,
                    fname       : str,
                    X           : numpy.ndarray,
                    T           : numpy.ndarray,
                    save_dir    : Path          = Path("."),
                    fps         : int           = 30,
                    dpi         : int           = 150,
                    cmap        : str           = "viridis") -> Path:
    """
    Create an MP4 showing the evolution of a **2-D vector** field sampled
    on a point cloud.

    
    
    -------------------------------------------------------------------------------------------
    Arguments
    -------------------------------------------------------------------------------------------

    data : ndarray, shape (N_t, 2, N_x) or (N_t, 3, N_x)
        Vector values (``u, v`` components if 2d, or ``u, v, w`` components if 3d) for every 
        time step and sensor.
    
    title : str
        Text for the figure title & colour-bar label.
    
    fname : str
        File name (without directory) of the resulting movie.
    
    X : ndarray, shape (2, N_x) or (3, N_x)
        Sensor coordinates. Can be a 2d or 3d array.
    
    T : ndarray, shape (N_t,)
        Time stamps.
    
    save_dir : pathlib.Path, default ``Path('.')``
        Directory in which the movie is written.
    
    fps : int, default 30
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
    
    # Checks
    assert isinstance(data, numpy.ndarray), "type(data) = %s" % str(type(data));
    assert isinstance(X, numpy.ndarray),    "type(X) = %s" % str(type(X));
    assert isinstance(T, numpy.ndarray),    "type(T) = %s" % str(type(T));
    assert len(data.shape) == 3,            "data.shape = %s" % str(data.shape);
    assert len(X.shape) == 2,               "X.shape = %s" % str(X.shape);
    assert len(T.shape) == 1,               "T.shape = %s" % str(T.shape);
    assert data.shape[1] in (2, 3),              "data.shape = %s" % str(data.shape);
    assert X.shape[0] in (2, 3),            "X.shape = %s" % str(X.shape);
    assert data.shape[0] == T.shape[0],     "data.shape = %s, T.shape = %s" % (str(data.shape), str(T.shape));
    assert data.shape[2] == X.shape[1],     "data.shape = %s, X.shape = %s" % (str(data.shape), str(X.shape));

    # Setup.
    save_dir = Path(save_dir);
    save_dir.mkdir(parents = True, exist_ok = True);

    N_t         : int   = T.shape[0];
    magnitudes          = numpy.linalg.norm(data, axis = 1);
    vmin                = magnitudes.min();
    vmax                = magnitudes.max();

    # Make a quiver plot using the data
    if X.shape[0] == 2:
        fig, ax = plt.subplots();
    else:
        # 3D vector-field animation is not robustly supported by the current update logic.
        # (Matplotlib 3D quiver objects do not support q.set_UVC like 2D quiver does.)
        raise NotImplementedError("3D vector-field animation is not supported. X.shape = %s" % str(X.shape));

    # Create a new figure and a single subplot (axes) object
    if(X.shape[0] == 2):
        q = ax.quiver(  X[0],                           # 1D array of x-coordinates for arrow bases
                        X[1],                           # 1D array of y-coordinates for arrow bases
                        data[0, 0, :],                  # 1D array of x-components of vectors at frame 0
                        data[0, 1, :],                  # 1D array of y-components of vectors at frame 0
                        magnitudes[0],                  # scalar values used to color each arrow by its length
                        cmap            = cmap,         # colormap mapping magnitudes → colors
                        clim            = (vmin, vmax), # set color limits
                        angles          = "xy",         # interpret U/V in Cartesian coords
                        scale_units     = "xy",         # scale arrow lengths in data units
                        scale           = 1.0,          # no additional scaling
                        width           = 0.007);       # arrow shaft width
    else: # 3d
        q = ax.quiver(  X[0],                           # 1D array of x-coordinates for arrow bases
                        X[1],                           # 1D array of y-coordinates for arrow bases
                        X[2],                           # 1D array of z-coordinates for arrow bases
                        data[0, 0, :],                  # 1D array of x-components of vectors at frame 0
                        data[0, 1, :],                  # 1D array of y-components of vectors at frame 0
                        data[0, 2, :],                  # 1D array of z-components of vectors at frame 0
                        magnitudes[0],                  # scalar values used to color each arrow by its length
                        cmap            = cmap,         # colormap mapping magnitudes → colors
                        clim            = (vmin, vmax), # set color limits
                        scale           = 1.0,          # no additional scaling
                        width           = 0.007);       # arrow shaft width

    # Keep x and y axes at the same scale so arrows aren’t distorted
    ax.set_aspect("equal");

    # Add and label a colorbar for the magnitudes
    cb = fig.colorbar(q, ax = ax);
    cb.set_label("|value|", rotation = 0);

    # Set initial plot title including the time stamp for frame 0
    time_text = ax.set_title(f"{title}\n$t$ = {T[0]:.3f}");

    def update(frame: int):
        """
        Called once per frame by FuncAnimation.
        - Updates arrow U, V components and colors for the given frame index.
        - Updates the title text to reflect the current time.
        """
        
        # Update the quiver vectors and their color values
        if(X.shape[0] == 2):
            q.set_UVC(data[frame, 0, :], data[frame, 1, :], magnitudes[frame]);
        else: # 3d
            q.set_UVC(data[frame, 0, :], data[frame, 1, :], data[frame, 2, :], magnitudes[frame]);
        
        # Update the title with the new time
        time_text.set_text(f"{title}\n$t$ = {T[frame]:.3f}");
        
        return q, time_text;

    
    ani = FuncAnimation(fig, 
                        update, 
                        frames  = N_t,      # total frames to animate
                        blit    = False,    # redraw entire frame each update
                        repeat  = False);   # do not loop
    
    # Construct output filepath and save the animation using ffmpeg
    out_path = save_dir / fname;
    ani.save(str(out_path), writer = FFMpegWriter(fps = fps, codec = "libx264"));

    # Close the figure to free memory
    plt.close(fig);

    # Return the path to the saved animation file
    return out_path;



# -------------------------------------------------------------------------------------------------
# movie making function
# -------------------------------------------------------------------------------------------------

def make_solution_movies(   U_True          : numpy.ndarray,
                            U_Pred          : numpy.ndarray,
                            X               : numpy.ndarray,
                            T               : numpy.ndarray,
                            save_dir        : str | Path | None  = None,
                            fname_prefix    : str           = "solution",
                            fps             : int           = 30,
                            dpi             : int           = 150,
                            cmap            : str           = "viridis") -> tuple[Path, Path, Path]:
    """
    Create three movies visualising a spatio-temporal solution: the true field, the predicted 
    field, and their difference. The solution can be a scalar or vector field in a 2d or 3d domain.
    
    -----------------------------------------------------------------------------------------------
    Parameters
    -----------------------------------------------------------------------------------------------

    U_True, U_Pred : numpy.ndarray, shape = (N_t, 1, N_x), (N_t, 2, N_x), or (N_t, 3, N_x)
        Arrays of shape holding the true and predicted signal, respectively. If they have shape 
        (N_t, 1, N_x) then the solution should be a scalar field. If it is 2 or 3 then the 
        solution should be a 2-D or 3-D vector field, respectively. 

    X : numpy.ndarray, shape = (2, N_x) or (3, N_x)
        Each column gives the (x, y) (2D) or (x, y, z) (3d) coordinates of a sensor point.
    
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

    assert isinstance(U_True, numpy.ndarray),   "type(U_True) = %s" % str(type(U_True));
    assert isinstance(U_Pred, numpy.ndarray),   "type(U_Pred) = %s" % str(type(U_Pred));
    assert isinstance(X, numpy.ndarray),        "type(X) = %s" % str(type(X));
    assert isinstance(T, numpy.ndarray),        "type(T) = %s" % str(type(T));
    # Accept scalar fields as either (N_t, N_x) or (N_t, 1, N_x).
    if len(U_True.shape) == 2:
        U_True = U_True[:, None, :];
    if len(U_Pred.shape) == 2:
        U_Pred = U_Pred[:, None, :];
    assert len(U_True.shape) == 3,              "U_True.shape = %s" % str(U_True.shape);
    assert len(U_Pred.shape) == 3,              "U_Pred.shape = %s" % str(U_Pred.shape);
    assert len(X.shape) == 2,                   "X.shape = %s" % str(X.shape);
    assert len(T.shape) == 1,                   "T.shape = %s" % str(T.shape);
    assert U_True.shape == U_Pred.shape,        "U_True.shape = %s, U_Pred.shape = %s" % (str(U_True.shape), str(U_Pred.shape));
    assert X.shape[0] in (2, 3),                "X.shape = %s" % str(X.shape);
    assert T.shape[0] == U_True.shape[0],       "T.shape = %s, U_True.shape = %s" % (str(T.shape), str(U_True.shape));
    assert X.shape[1] == U_True.shape[2],       "X.shape = %s, U_True.shape = %s" % (str(X.shape), str(U_True.shape));
    
    N_t, n_comp, N_x = U_True.shape;
    assert n_comp in (1, 2, 3),                 "n_comp = %s" % n_comp;
    
    # Default save directory: <Higher-Order-LaSDI>/Figures (independent of CWD).
    if save_dir is None:
        project_dir = Path(__file__).resolve().parent.parent;
        save_dir = project_dir / "Figures";
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
        LOGGER.info("Making scalar field movie for %s" % fname_prefix);
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
    
    else:  # 2 or 3 components: vector field
        LOGGER.info("Making vector field movie for %s" % fname_prefix);
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




# -------------------------------------------------------------------------------------------------
# Specialized function for when the solution is a scalar field on a 2D grid
# -------------------------------------------------------------------------------------------------

def Animate_2D_Grid_Scalar( U           : numpy.ndarray,
                            x_values    : numpy.ndarray,
                            y_values    : numpy.ndarray,
                            T           : numpy.ndarray,
                            save_dir    : str = None,
                            fname       : str = "Burgers2D.mp4",
                            levels      : int = 300,
                            fps         : int = 30,
                            dpi         : int = 150) -> str:
    """
    Create an MP4 animation of the true scalar 2D field using filled contour plots.


    Arguments
    -------------------------------------------------------------------------------------------------

    U : numpy.ndarray, shape = (N_t, 1, N_x*N_y)
        Solution values over time flattened along space.

    x_values, y_values : numpy.ndarray
        1D arrays of spatial coordinates in x and y of lengths N_x and N_y, respectively.

    T : numpy.ndarray, shape = (N_t)
        Time stamps corresponding to the frames.

    save_dir : str
        Directory where the MP4 is saved. Defaults to '<project>/Figures/Burgers2D'.

    fname : str
        Output movie file name.

    levels : int
        Number of contour levels for contourf.

    fps : int
        Frames per second for the animation.

    dpi : int
        Resolution of the figure.


    Returns
    -------------------------------------------------------------------------------------------------

    out_path : str
        Absolute path to the saved MP4 file.
    """

    # Checks
    assert isinstance(U, numpy.ndarray),            "U must be a numpy array";
    assert isinstance(x_values, numpy.ndarray),     "x_values must be a numpy array";
    assert isinstance(y_values, numpy.ndarray),     "y_values must be a numpy array";
    assert isinstance(T, numpy.ndarray),            "T must be a numpy array";
    assert U.ndim == 3 and U.shape[1] == 1,         "U must have shape (N_t, 1, N_x*N_y)";
    assert x_values.ndim == 1 and y_values.ndim == 1 and T.ndim == 1;

    N_t : int = U.shape[0];
    N_x : int = x_values.shape[0];
    N_y : int = y_values.shape[0];
    assert U.shape[2] == N_x * N_y,                 "U's spatial dimension must be N_x*N_y";
    assert T.shape[0] == N_t,                       "Length of T must equal number of frames";

    # Reshape to (N_t, N_x, N_y)
    U_grid : numpy.ndarray = U[:, 0, :].reshape(N_t, N_x, N_y);

    # Prepare coordinates and global color limits
    X, Y = numpy.meshgrid(x_values, y_values, indexing = 'ij');
    vmin = float(U_grid.min());
    vmax = float(U_grid.max());

    # Gray-to-orange colormap
    cmap = LinearSegmentedColormap.from_list(
            'gray_to_orange',
            ["midnightblue", "white", "slategray"],
            N = 256);

    # Figure and first frame
    plt.rcParams.update({   "figure.dpi"        : dpi,
                            "font.size"         : 12,
                            "axes.grid"         : True,
                            "grid.alpha"        : 0.25,
                            "axes.spines.top"   : False,
                            "axes.spines.right" : False});
    fig, ax = plt.subplots();
    cont = ax.contourf(X, Y, U_grid[0], levels = levels, cmap = cmap, vmin = vmin, vmax = vmax);
    cb = fig.colorbar(cont, ax = ax);
    ax.set_xlabel("x");
    ax.set_ylabel("y", rotation = 0, labelpad = 10);
    title_text = ax.set_title(f"2D Burgers (true)\n$t$ = {T[0]:.3f}");
    ax.set_aspect('equal');

    # Define the update function for the animation.
    def update(frame: int):
        nonlocal cont;
        # Remove old contour collections
        for c in cont.collections:
            c.remove();

        # Draw new frame
        cont = ax.contourf(X, Y, U_grid[frame], levels = levels, cmap = cmap, vmin = vmin, vmax = vmax);
        title_text.set_text(f"2D Burgers (true)\n$t$ = {T[frame]:.3f}");
        return cont.collections + [title_text];

    # Create the animation.
    ani = FuncAnimation(fig, update, frames = N_t, blit = False, repeat = False);

    # Save output
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(os.path.pardir)), "Figures", "Burgers2D");
    os.makedirs(save_dir, exist_ok = True);
    out_path = os.path.join(save_dir, fname);
    ani.save(out_path, writer = FFMpegWriter(fps = fps, codec = "libx264"));
    plt.close(fig);
    return out_path;

