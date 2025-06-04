# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import      os;
import      sys;
import      logging;

import      mfem.ser    as      mfem;
from        os.path     import  expanduser, join, dirname;
import      numpy;

utils_path : str        = os.path.join(os.path.join(os.path.pardir, os.path.pardir), "Utilities");
sys.path.append(utils_path);
import      Logging;


# Logger Setup 
LOGGER : logging.Logger = logging.getLogger(__name__);

mfem_version : int = (mfem.MFEM_VERSION_MAJOR*100 + mfem.MFEM_VERSION_MINOR*10+
                      mfem.MFEM_VERSION_PATCH);





# -------------------------------------------------------------------------------------------------
# Wave Equation class
# -------------------------------------------------------------------------------------------------

class WaveOperator(mfem.SecondOrderTimeDependentOperator):
    def __init__(self, fespace : mfem.FiniteElementSpace, ess_bdr : mfem.intArray, speed : float) -> None:
        """
        Initialize the WaveOperator.

        ---------------------------------------------------------------------------------------------
        Arguments
        ---------------------------------------------------------------------------------------------

        fespace : mfem.FiniteElementSpace
            The finite element space to use. 

        ess_bdr : mfem.intArray
            The essential boundary conditions. This is an array of integers, where each integer 
            corresponds to a boundary attribute. If the integer is 1, then the boundary condition 
            is essential (i.e., fixed). If the integer is 0, then the boundary condition is natural 
            (i.e., free).

        speed : float   
            The speed of the wave in the direction of the wave's propagation.
        """ 

        # Run the super class initializer
        mfem.SecondOrderTimeDependentOperator.__init__(self, fespace.GetTrueVSize(), 0.0)

        # Get the essential boundary conditions
        self.ess_tdof_list : mfem.intArray = mfem.intArray();
        fespace.GetEssentialTrueDofs(ess_bdr, self.ess_tdof_list);

        # Define the diffusion coefficient
        c2  : mfem.ConstantCoefficient  = mfem.ConstantCoefficient(speed*speed);
        K   : mfem.BilinearForm         = mfem.BilinearForm(fespace);
        K.AddDomainIntegrator(mfem.DiffusionIntegrator(c2));
        K.Assemble();

        # Initialize the stiffness and mass matrices
        self.Kmat : mfem.SparseMatrix = mfem.SparseMatrix();
        self.Mmat : mfem.SparseMatrix = mfem.SparseMatrix();

        # Define the mass matrix
        M   : mfem.BilinearForm = mfem.BilinearForm(fespace);
        M.AddDomainIntegrator(mfem.MassIntegrator());
        M.Assemble();

        # Build K and M from the 
        if mfem_version < 471:
            dummy       : mfem.intArray     = mfem.intArray();
            self.Kmat0  : mfem.SparseMatrix = mfem.SparseMatrix();
            K.FormSystemMatrix(dummy, self.Kmat0);
        
        # TODO: What does this function do?  
        K.FormSystemMatrix(self.ess_tdof_list, self.Kmat);
        M.FormSystemMatrix(self.ess_tdof_list, self.Mmat);

        # Store the stiffness and mass matrices
        self.K : mfem.BilinearForm = K;
        self.M : mfem.BilinearForm = M;

        # Define the relative tolerance (for the solver)
        rel_tol : float = 1e-8;

        # Initialize the solver and preconditioner for M
        M_solver     : mfem.CGSolver    = mfem.CGSolver();
        M_prec       : mfem.DSmoother = mfem.DSmoother();
        M_solver.iterative_mode = False;
        M_solver.SetRelTol(rel_tol);
        M_solver.SetAbsTol(0.0);
        M_solver.SetMaxIter(30);
        M_solver.SetPrintLevel(0);
        M_solver.SetPreconditioner(M_prec);
        M_solver.SetOperator(self.Mmat);
        self.M_prec       : mfem.DSmoother  = M_prec;
        self.M_solver     : mfem.CGSolver   = M_solver;

        # Initialize the solver and preconditioner for T
        T_solver     : mfem.CGSolver    = mfem.CGSolver();
        T_prec       : mfem.DSmoother   = mfem.DSmoother();
        T_solver.iterative_mode = False;
        T_solver.SetRelTol(rel_tol);
        T_solver.SetAbsTol(0.0);
        T_solver.SetMaxIter(100);
        T_solver.SetPrintLevel(0);
        T_solver.SetPreconditioner(T_prec);
        self.T_prec     : mfem.DSmoother    = T_prec;
        self.T_solver   : mfem.CGSolver     = T_solver;
        self.T          : mfem.SparseMatrix = None;

        # Initialize the vector
        self.z : mfem.Vector = mfem.Vector(self.Height());


        
    def Mult(self, u : mfem.Vector, du_dt : mfem.Vector, d2udt2 : mfem.Vector) -> None:
        # Compute:
        #    d2udt2 = M^{-1}*-K(u)
        # for d2udt2

        # Compute the stiffness matrix
        z : mfem.Vector = self.z;
        self.K.FullMult(u, z);
        z.Neg();

        # Set the essential boundary conditions to zero
        z.SetSubVector(self.ess_tdof_list, 0.0);

        # Solve for d2udt2
        self.M_solver.Mult(z, d2udt2);

        # Set the essential boundary conditions to zero
        d2udt2.SetSubVector(self.ess_tdof_list, 0.0);



    def ImplicitSolve(self, fac0 : float, fac1 : float, u : mfem.Vector, dudt : mfem.Vector, d2udt2 : mfem.Vector) -> None:
        # Solve the equation:
        #    d2udt2 = M^{-1}*[-K(u + fac0*d2udt2)]
        # for d2udt2
        if self.T is None:
            self.T : mfem.SparseMatrix = mfem.Add(1.0, self.Mmat, fac0, self.Kmat);
            self.T_solver.SetOperator(self.T)

        z : mfem.Vector = self.z;
        self.K.FullMult(u, z);
        z.Neg();
        z.SetSubVector(self.ess_tdof_list, 0.0);

        self.T_solver.Mult(z, d2udt2)

        if mfem_version >= 471:
            d2udt2.SetSubVector(self.ess_tdof_list, 0.0)



    def SetParameters(self, u):
        self.T = None


class cInitialSolution(mfem.PyCoefficient):
    def EvalValue(self, x : numpy.ndarray) -> float:    
        global decay;
        norm2 : float = numpy.sum(x**2);
        return numpy.exp(-norm2*decay);



class cInitialRate(mfem.PyCoefficient):
    def EvalValue(self, x : numpy.ndarray) -> float:
        return 0;



# -------------------------------------------------------------------------------------------------
# Simulate function
# -------------------------------------------------------------------------------------------------

def Simulate(mesh_file          : str   = "star.mesh",
             ref_levels         : int   = 0,
             order              : int   = 2,
             ode_solver_type    : int   = 10,
             t_final            : float = 1.0,
             dt                 : float = 1e-2,
             c                  : float = 0.5,
             k                  : float = 20.0,
             dirichlet          : bool  = True,
             serialization_steps: int   = 1,
             num_positions      : int   = 1000,
             VisIt              : bool  = True) -> tuple[numpy.ndarray,numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Simulate the wave equation.

    ---------------------------------------------------------------------------------------------
    Arguments
    ---------------------------------------------------------------------------------------------

    mesh_file : str
        The mesh file to use.

    ref_levels : int
        The number of times to refine the mesh uniformly before parallelization.

    order : int
        The order of the finite element space (i.e., the degree of the polynomial basis functions).

    ode_solver_type : int
        An integer specifying the type of ODE solver to use. The following options are available:
        - 10: GeneralizedAlpha2Solver(0.1 * s)
        - 11: AverageAccelerationSolver
        - 12: LinearAccelerationSolver
        - 13: CentralDifferenceSolver
        - 14: FoxGoodwinSolver

    t_final : float
        The final time. We solve the wave equation from t = 0 to t = t_final.

    dt : float
        The time step. We solve the wave equation using a time-stepping scheme with time step dt.

    c : float
        The speed of the wave.  

    k : float
        A coefficient used to define the initial solution: u(0, x) = exp(-k*|x|^2).

    dirichlet : bool
        Whether to use Dirichlet boundary conditions. If True, we fix the position of the nodes on the 
        boundary. If False, we allow the nodes to move freely on the boundary.

    serialization_steps : int
        Specifies how frequently we serialize (save) and visualize the solution.
    
    num_positions : int
        Specifies the number of positions at which we will evaluate the solution.
        
    VisIt : bool
        If True, will prompt the code to save the displacement and velocity GridFunctions every 
        time we serialize them. It will save the GridFunctions in a format that VisIt 
        (visit.llnl.gov) can understand/work with.
        

    ---------------------------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------------------------------

    U, DtU, X, T

    D : numpy.ndarray, shape = (Nt, 2, N_Nodes)
        i, j, k element holds the j'th component of the solution at the k'th position (i.e., 
        X[i, :]) at the i'th time step (i.e., T[i]).
    
    DtU : numpy.ndarray, shape = (Nt, 2, N_Nodes)
        i, j, k element holds the j'th component of the time derivative of the solution at the k'th 
        position (i.e., X[i]) at the i'th time step (i.e., T[i]).

    X : numpy.ndarray, shape = (2, N_Nodes)
        i'th row holds the position of the i'th node at which we evaluate the solution.
    
    T : numpy.ndarray, shape = (Nt)
        i'th element holds the j'th time at which we evaluate the solution.
    """
    

    # ---------------------------------------------------------------------------------------------
    # 1. Setup 
    
    LOGGER.info("Setting up wave equation simulation with MFEM.");
    
    # Set the global variable c.
    global decay;
    decay = k;

    # Define the ODE solver used for time integration.
    LOGGER.debug("Defining the ODE solver.");
    if   ode_solver_type <= 10:
        ode_solver : mfem.TimeIntegrator = mfem.GeneralizedAlpha2Solver(ode_solver_type/10);
    elif ode_solver_type == 11:
        ode_solver : mfem.TimeIntegrator = mfem.AverageAccelerationSolver();
    elif ode_solver_type == 12:
        ode_solver : mfem.TimeIntegrator = mfem.LinearAccelerationSolver();
    elif ode_solver_type == 13:
        ode_solver : mfem.TimeIntegrator = mfem.CentralDifferenceSolver();
    elif ode_solver_type == 14:
        ode_solver : mfem.TimeIntegrator = mfem.FoxGoodwinSolver();
    else:
        print("Unknown ODE solver type: " + str(ode_solver_type));



    # ---------------------------------------------------------------------------------------------
    # 2. Setup the mesh.

    # Load the mesh.
    LOGGER.debug("Loading the mesh and its properties");
    meshfile_path   : str   = expanduser(join(dirname(__file__), 'data', mesh_file));
    mesh                    = mfem.Mesh(meshfile_path, 1, 1);

    # Get the dimension (i.e., number of spatial dimensions) of the mesh.
    dim : int = mesh.Dimension();
    LOGGER.debug("mesh dimension = %d" % dim);

    # Serially refine the mesh to increase the resolution. 
    LOGGER.debug("Refining the mesh to increase the resolution.");
    for lev in range(ref_levels):
        mesh.UniformRefinement();



    # ---------------------------------------------------------------------------------------------
    # 3. Define the finite element space and grid functions to hold the solution and the time 
    # derivative of the solution.

    LOGGER.info("Defining the finite element space and grid functions to hold U and (d/dt)U.");

    LOGGER.debug("Defining the finite element space.");
    fe_coll : mfem.FiniteElementCollection  = mfem.H1_FECollection(order, dim);         # Basis functions
    fespace : mfem.FiniteElementSpace       = mfem.FiniteElementSpace(mesh, fe_coll);   # FEM space (span of basis functions).

    # Get the number of degrees of freedom in the finite element space.
    fe_size : int = fespace.GetTrueVSize();
    LOGGER.debug("Number of unknowns to solve for: %d" % fe_size);

    # Initialize the grid functions for the solution and the time derivative of the solution.
    u_gf    : mfem.GridFunction = mfem.GridFunction(fespace);
    dudt_gf : mfem.GridFunction = mfem.GridFunction(fespace);



    # ---------------------------------------------------------------------------------------------
    # 4. Set the initial conditions for U and (d/dt)U.

    LOGGER.info("Setting the initial conditions for U and (d/dt)U.");

    # Set the initial conditions for u. All boundaries are considered natural.
    u_0     : mfem.PyCoefficient = cInitialSolution();
    dudt_0  : mfem.PyCoefficient = cInitialRate();

    # Project the initial conditions onto the finite element space.
    u_gf.ProjectCoefficient(u_0);
    u : mfem.Vector = mfem.Vector();        # Vector that will hold the true degrees of freedom (i.e., the 
                                            # values of U at the nodes) for U.
    u_gf.GetTrueDofs(u);

    # Project the initial conditions onto the finite element space.
    dudt_gf.ProjectCoefficient(dudt_0);
    dudt : mfem.Vector = mfem.Vector();     # Vector that will hold the true degrees of freedom (i.e., the 
                                            # values of (d/dt)U at the nodes) for (d/dt)U.
    dudt_gf.GetTrueDofs(dudt);



    # ---------------------------------------------------------------------------------------------
    # 5. Initialize the wave operator.

    LOGGER.info("Initializing the wave operator.");

    # Define the essential boundary conditions. 
    ess_bdr : mfem.intArray = mfem.intArray();
    if mesh.bdr_attributes.Size():
        ess_bdr.SetSize(mesh.bdr_attributes.Max());

        # If Dirichlet (i.e., fixed position) boundary conditions are used, we set every
        # element of ess_brd to 1.
        if (dirichlet):
            ess_bdr.Assign(1);
        else:
            ess_bdr.Assign(0);

    # Initialize the wave operator.
    oper : WaveOperator = WaveOperator(fespace = fespace, ess_bdr = ess_bdr, speed = c);



    # ---------------------------------------------------------------------------------------------
    # 5. Set up positions at which we will evaluate the solution.

    LOGGER.info("Sampling %d positions in the mesh" % num_positions);

    # Figure out the maximum/minimum x and y coordinates of the mesh.
    bb_min, bb_max = mesh.GetBoundingBox();
    x_min   : float = bb_min[0];
    x_max   : float = bb_max[0];
    y_min   : float = bb_min[1];
    y_max   : float = bb_max[1];
    LOGGER.debug("x_min = %f, x_max = %f, y_min = %f, y_max = %f" % (x_min, x_max, y_min, y_max));

    # Now, sample num_positions points evenly spaced between x_min and x_max, and y_min and y_max.
    # If the mesh has an unusal shape, some of these points may lie outside the mesh. We sample 
    # too many positions to account for this. Any points that lie outside the mesh will be ignored.
    # We will sample new points if the number of points that lie inside the mesh is less than 
    # num_positions.
    Valid_Positions_List : list[numpy.ndarray] = [];
    Elements_List        : list[int]           = [];
    RefCoords_List       : list[numpy.ndarray] = [];
    num_valid_positions  : int = 0;
    
    while(num_valid_positions < num_positions):
        LOGGER.debug("Sampling %d positions" % num_positions);

        # Sample random x,y coordinates
        x_positions : numpy.ndarray = numpy.random.uniform(x_min, x_max, num_positions);
        y_positions : numpy.ndarray = numpy.random.uniform(y_min, y_max, num_positions);

        # Create array of points in format expected by FindPoints
        points : numpy.ndarray = numpy.column_stack((x_positions, y_positions));

        # Find which points are in the mesh.
        count, elem_list, ref_coords = mesh.FindPoints(points, warn = False, inv_trans = None);

        # Check which points are inside elements
        for i in range(num_positions):
            if elem_list[i] >= 0:  # -1 indicates point not found in any element
                Valid_Positions_List.append(points[i]);
                Elements_List.append(elem_list[i]);
                RefCoords_List.append(ref_coords[i]);
                num_valid_positions += 1;

                # If we have enough valid positions, break.
                if num_valid_positions >= num_positions:
                    break;

        # If we have not enough valid positions, sample again.
        if(num_valid_positions < num_positions):
            LOGGER.debug("Not enough valid positions (current = %d, needed = %d), sampling again" % (num_valid_positions, num_positions));

    # Convert the lists to numpy arrays.
    Positions : numpy.ndarray = numpy.array(Valid_Positions_List).T;
    Elements  : numpy.ndarray = numpy.array(Elements_List);
    RefCoords : numpy.ndarray = numpy.array(RefCoords_List);
    LOGGER.debug("Positions has shape %s (dim = %d, num_positions = %d)" % (str(Positions.shape), dim, Positions.shape[1]));


    
    # ---------------------------------------------------------------------------------------------
    # 6. VisIt

    # Store the initial solution to u_gf and dudt_gf.
    u_gf.SetFromTrueDofs(u);
    dudt_gf.SetFromTrueDofs(dudt);

    if(VisIt):
        LOGGER.info("Setting up VisIt visualization.");

        # Create the VisIt data collection.
        visit_dc_path   : str                       = os.path.join(os.path.join(os.path.dirname(__file__), "VisIt"), "waveEq-fom");
        visit_dc        : mfem.VisItDataCollection  = mfem.VisItDataCollection(visit_dc_path, mesh);
        visit_dc.SetPrecision(8);

        # Register U and its time derivative.
        visit_dc.RegisterField("U",     u_gf);
        visit_dc.RegisterField("DtU",   dudt_gf);

        # Set the cycle and time.
        visit_dc.SetCycle(0);
        visit_dc.SetTime(0.0);
        visit_dc.Save();


    # ---------------------------------------------------------------------------------------------
    # 7. Setup lists to store the solution + evaluate the initial solution at the positions.

    LOGGER.info("Setting up lists to store the time, U, and DtU at each time step.");

    # Setup for time stepping.
    times_list          : list[float]           = [];    
    displacements_list  : list[numpy.ndarray]   = [];
    velocities_list     : list[numpy.ndarray]   = [];

    # Evaluate the initial solution at the positions.
    u_Positions_0       = numpy.zeros((1, num_positions));
    dudt_Positions_0    = numpy.zeros((1, num_positions));

    for i in range(num_positions):
        u_Positions_0[0, i]     = u_gf.GetValue(Elements[i], RefCoords[i], dim);
        dudt_Positions_0[0, i]  = dudt_gf.GetValue(Elements[i], RefCoords[i], dim);

    # Append the initial U, DtU, and time to their corresponding lists.
    times_list.append(0);
    displacements_list.append(  u_Positions_0);
    velocities_list.append(     dudt_Positions_0);


    # ---------------------------------------------------------------------------------------------
    # 8. Perform time-integration (looping over the time iterations, ti, with a time-step dt).

    LOGGER.info("Performing time-integration (dt = %g, t_final = %g)" % (dt, t_final));
    
    # Time step!!!!!
    ode_solver.Init(oper);
    t           : float = 0.0;
    ti          : int   = 0;
    last_step   : bool  = False;

    while not last_step:
        # Check if we should stop time stepping (if this time step is within dt/2 of t_final).
        if t + dt >= t_final - dt/2:
            last_step = True

        # Step the ODE solver.
        t, dt = ode_solver.Step(u, dudt, t, dt);
        ti += 1;

        # Should we serialize?
        if last_step or (ti % serialization_steps == 0):
            LOGGER.info("time step: " + str(ti) + ", time: " + str(numpy.round(t, 3)));

            # Update the solution to the grid functions
            u_gf.Assign(u);
            dudt_gf.Assign(dudt);

            # Evaluate the solution at the positions.
            u_Positions_t       = numpy.zeros((1, num_positions));
            dudt_Positions_t    = numpy.zeros((1, num_positions));

            for i in range(num_positions):
                u_Positions_t[0, i]     = u_gf.GetValue(Elements[i], RefCoords[i], dim);
                dudt_Positions_t[0, i]  = dudt_gf.GetValue(Elements[i], RefCoords[i], dim);

            # Append the current U, DtU, and time to their corresponding lists.
            times_list.append(t);
            displacements_list.append(  u_Positions_t);
            velocities_list.append(     dudt_Positions_t);
    
            # If visualizing, Save the solution to the VisIt object.
            if VisIt:
                visit_dc.SetCycle(ti);
                visit_dc.SetTime(t);
                visit_dc.Save();

        # Update the wave operator with the current solution.
        oper.SetParameters(u);



    # ---------------------------------------------------------------------------------------------
    # 9. Package everything up for returning.

    # Turn times, displacements, velocities lists into arrays.
    Times           = numpy.array(times_list,           dtype = numpy.float32);
    U               = numpy.array(displacements_list,   dtype = numpy.float32);
    DtU             = numpy.array(velocities_list,      dtype = numpy.float32);

    # Return everything
    return U, DtU, Positions, Times;



if __name__ == "__main__":
    Logging.Initialize_Logger(level = logging.DEBUG);
    U, DtU, X, T = Simulate();
