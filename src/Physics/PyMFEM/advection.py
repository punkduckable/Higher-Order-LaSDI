# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

from        mfem                    import  path;
import      mfem.par                as      mfem;
from        mfem.par                import  intArray;
from        mpi4py                  import  MPI;
import      numpy;

import      os;
import      sys;
import      logging;
from        os.path                 import  expanduser, join, dirname, exists;

utils_path : str        = os.path.join(os.path.join(os.path.pardir, os.path.pardir), "Utilities");
sys.path.append(utils_path);
import      Logging;


# Logger Setup 
LOGGER : logging.Logger = logging.getLogger(__name__);


# -------------------------------------------------------------------------------------------------
# Coefficient
# -------------------------------------------------------------------------------------------------

class velocity_coeff(mfem.VectorPyCoefficient):
    def __init__(self, dim : int, bb_min : numpy.ndarray, bb_max : numpy.ndarray, gamma : float) -> None:
        """
        Initialize a velocity_coeff object. This class is used to define the velocity field for the 
        advection problem. If dim = 2, the velocity field is defined by

            v(x, y) = d(x, y)* gamma * (y, -x)

        where d(x, y) = max((x + 1)*(1 - x), 0.) * max((y + 1)*(1 - y), 0.) takes on non-zero 
        values in [-1, 1] x [-1, 1] and is zero outside of this domain. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        dim : int
            The dimension of the velocity field.

        bb_min : numpy.ndarray, shape = (dim,)
            The minimum coordinates of the bounding box.

        bb_max : numpy.ndarray, shape = (dim,)
            The maximum coordinates of the bounding box.

        gamma : float
            The parameter in the velocity field. See above.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Run checks
        assert(isinstance(bb_min, numpy.ndarray));
        assert(isinstance(bb_max, numpy.ndarray));
        assert(len(bb_min) == dim);
        assert(len(bb_max) == dim);
        for i in range(dim):
            assert(bb_min[i] < bb_max[i]);

        # Run the super class initializer
        mfem.VectorPyCoefficient.__init__(self, dim);

        # Now set problem specific attributes.
        self.dim        : int           = dim;
        self.bb_min     : numpy.ndarray = bb_min;
        self.bb_max     : numpy.ndarray = bb_max;
        self.gamma      : float         = gamma;



    def EvalValue(self, x : numpy.ndarray) -> numpy.ndarray:
        """
        Evaluate the velocity field at the given position.

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        x : numpy.ndarray, shape = (dim,)
            A 1D array holding the position at which to evaluate the velocity field.   


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        numpy.ndarray, shape = (dim,)
            A 1D array holding the velocity field at the given position.
        """

        # Run checks
        assert(isinstance(x, numpy.ndarray));
        assert(len(x.shape) == 1);
        assert(len(x) == self.dim);

        # Get the bounding box.
        bb_min : numpy.ndarray = self.bb_min;
        bb_max : numpy.ndarray = self.bb_max;

        # Get the center of the bounding box.
        center : float = (bb_min + bb_max)/2.0;

        # map to the reference [-1,1] domain
        X : numpy.ndarray = 2 * (x - center) / (bb_max - bb_min);

        # Get the velocity field. (Clockwise twisting rotation in 2D around the origin
        d : float = max((X[0] + 1.)*(1. - X[0]), 0.) * max((X[1] + 1.)*(1. - X[1]), 0.);
        d : float = d ** 2;

        if self.dim == 1:
            v : list[float] = [1.0];
        elif self.dim == 2:
            v : list[float] = [d*self.gamma*X[1],  - d*self.gamma*X[0]];
        elif self.dim == 3:
            v : list[float] = [d*self.gamma*X[1],  - d*self.gamma*X[0],  0];

        return v;



class Initial_Displacement(mfem.PyCoefficient):
    def __init__(self, bb_min : numpy.ndarray, bb_max : numpy.ndarray, k : float, w : float):
        """
        Initialize a Initial_Displacement object. This class is used to define the initial 
        condition for the advection problem. The initial condition is defined by

            u(0, (x, y)) = exp(-k*(x^2 + y^2)) * sin(pi*w*x~) * sin(pi*w*y~)

        where x~ and y~ are the non-dimensionalized coordinates. Specifically, 

            x~ = 2 * (x - center) / (bb_max - bb_min)
            y~ = 2 * (y - center) / (bb_max - bb_min)

        where center = (bb_min + bb_max) / 2.0.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        bb_min : numpy.ndarray, shape = (2,)
            The minimum coordinates of the bounding box.

        bb_max : numpy.ndarray, shape = (2,)
            The maximum coordinates of the bounding box.

        k : float   
            The decay parameter. See above.

        w : float
            The frequency parameter. See above. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Run checks
        assert(isinstance(bb_min, numpy.ndarray));
        assert(isinstance(bb_max, numpy.ndarray));
        assert(len(bb_min) == 2);
        assert(len(bb_max) == 2);

        # Run the super class initializer
        mfem.PyCoefficient.__init__(self);

        # Now set problem specific attributes.
        self.bb_min : numpy.ndarray = bb_min;
        self.bb_max : numpy.ndarray = bb_max;
        self.k      : float         = k;
        self.w      : float         = w;



    def EvalValue(self, x : numpy.ndarray) -> numpy.ndarray:
        """
        This function returns the initial condition for the advection problem:

            u_0(x, y) = exp(-k(x~^2 + y~^2))*sin(pi*w*x~) * sin(pi*w*y~)

        where x~ and y~ are the non-dimensionalized coordinates. Specifically, 

            x~ = 2 * (x - center) / (bb_max - bb_min)
            y~ = 2 * (y - center) / (bb_max - bb_min)

        where center = (bb_min + bb_max) / 2.0 and k, w are parameters in self. See the initializer 
        for more details.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        x : numpy.ndarray, shape = (2, N) or (2)
            The position at which to evaluate the initial condition. The first element is the 
            x-coordinate, and the second element is the y-coordinate. If x.shape = (2, N), then 
            x[:, j] is the j'th position at which to evaluate the initial condition. If x.shape = (2), 
            then x is a scalar holding the position at which to evaluate the initial condition.

            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        u : numpy.ndarray, shape = (1, N)
            The value of the initial condition at the given position. If x.shape = (2, N), then 
            u[:, j] is the value of the initial condition at the j'th position. If x.shape = (2), 
            then u is a scalar holding the value of the initial condition at the lone position in x.
        """

        assert(isinstance(x, numpy.ndarray));
        assert(x.shape[0] == 2);

        # Get the center and width of the bounding box.
        center : numpy.ndarray = (self.bb_min + self.bb_max)/2.0;       # shape = (2,)
        width  : numpy.ndarray = (self.bb_max - self.bb_min)/2.0;      # shape = (2,)

        # Reshape center, width to have shape (2, 1)
        center = center.reshape(2, 1);
        width  = width.reshape(2, 1);

        # Get the number of points.
        if(len(x.shape) == 1):
            x = x.reshape(2, 1);

        # Get the number of points.
        N : int = x.shape[1];

        # Map to the reference [-1,1] domain.
        X : numpy.ndarray = 2 * (x - center) / width; # shape = (2, N)
        
        # Return the initial condition.
        norm2 : numpy.ndarray = numpy.sum(numpy.square(X), axis = 0); # shape = (N,)
        u     : numpy.ndarray = numpy.exp(-self.k*norm2) * numpy.sin(numpy.pi * self.w * X[0]) * numpy.sin(numpy.pi * self.w * X[1]); # shape = (N)

        if(N == 1):
            return u[0];
        else:
            return u.reshape(1, N);





# Inflow boundary condition (zero for the problems considered in this example)
class inflow_coeff(mfem.PyCoefficient):
    def EvalValue(self, x):
        return 0



# -------------------------------------------------------------------------------------------------
# Advection Class
# -------------------------------------------------------------------------------------------------

class AdvectionOperator(mfem.PyTimeDependentOperator):
    def __init__(self, fespace : mfem.FiniteElementSpace, velocity : velocity_coeff, g : inflow_coeff) -> None:
        """
        This class implements the Advection operator. The Advection operator is given by:
            
            (d/dt)u(t, X) + v(X) \cdot \nabla(u(t, X)) = 0
            
        where u is the scalar field, v(X) is the velocity field. Let { \phi_i } be the basis 
        functions in the finite element space fespace. Then the weak form of the Advection operator 
        is given by:

            (\phi_i(X), (d/dt) u(t, X)) + (\phi_i(X), v(X) \cdot \nabla u(t, X)) = 0
        
        Using integration by parts on the second term gives

            (\phi_i(X), (d/dt) u(t, X)) + (-v(X) \cdot \nabla \phi_i(X), u(t, X)) + \int_{F} \hat{u}(v(X) \cdot n) \phi_i(X) = \int_{Bd(\Omega)} g(v(X) \cdot \phi_i(X))
      
        Where \Omega is the problem domain, g defines the flow into the problem domain, and \hat{u} 
        is decided by the upwind scheme and n is the unit normal vector to the boundary between 
        elements at X. Specifically, 
        
            \hat{u}(v(X) \cdot n) = { u^-(X)            (v \cdot n)(X) \geq 0     
                                    { u^+(X)            (v \cdot n)(X) \leq 0 \hat{u} 
        
        flow into the problem domain. If we assume the solution is of the form 
        
            u(t, X) = \sum_{j = 1}^{N} \phi_j(X) U_j(t)
        
        then the weak form of the Klein-Gordon operator becomes:
        
            (\phi_i(X), \sum_{j = 1}^{N} \phi_j(X) U_j'(t)) 
            + (- v(X) \cdot \phi_i(X), \sum_{j = 1}^{N} \phi_j(X) U_j(t)) 
            + \sum_{j = 1}^N \int_{F} \hat{U_j(t)}(v(X) \cdot n(X)) \phi_i(X) 
            = \int_{Bd(\Omega)} g(v(X) \cdot \eta)

        Here, \hat{u} is decided by the upwind scheme and n is the unit normal vector to the 
        boundary between elements at X. Specifically, 
        
            \hat{u}(v(X) \cdot n) = { u^-(X)            (v \cdot n)(X) \geq 0     
                                    { u^+(X)            (v \cdot n)(X) \leq 0 

        Where u^- is the approximation to the solution in "downstream" of the normal n(X) (the 
        element we end up in if we move from the boundary, F, in the direction of n(X)). 
        u^+ is the opposite element. This equation engenders the following system 
        of equations:

            M*U'(t) = K*U(t) + b

        where M is the mass matrix, K is the stiffness matrix, U(t) is the vector whose j'th 
        entry is U_j(t), and b is a vector. The entires of M, K, and b are defined as follows:
        
            M_{ij}  = (\phi_i(X), \phi_j(X))
            K_{ij}  = (\phi_i(X), v(X) * \nabla \phi_j(X)) + \int_{F} \hat{U_j(t)}(v(X) \cdot n(X)) \phi_i(X) 
            b_i     = \int_{Bd(\Omega)} g(v(X) \cdot \phi_i(X))
                

            
        ---------------------------------------------------------------------------------------------
        Arguments
        ---------------------------------------------------------------------------------------------

        fespace : mfem.FiniteElementSpace
            The finite element space to use. 

        velocity : velocity_coeff
            A velocity_coeff object which defines "v(X)" in the equations above. 

        g : inflow_coeff
            A inflow_coeff object which defines the "g" term in the boundary condition. 
        """ 

        # Define the bilinear form corresponding to the term M in the weak form of the 
        # Advection equation (see the docstring above).
        M : mfem.ParBilinearForm = mfem.ParBilinearForm(fespace);                           # Initialize the bilinear form
        M.AddDomainIntegrator(mfem.MassIntegrator());                                       # Sets m to the bilinear form B(\phi, \eta) = (\phi, \eta)

        # Run the super class initializer. Note that fespace.GetTrueVSize() returns the number of 
        # "true" unknown DOFs after removing those that are constrained by Dirichlet BCs.
        mfem.PyTimeDependentOperator.__init__(self, M.Height())

        # Define the bilinear form corresponding to the term K in the docstring above.
        K : mfem.ParBilinearForm = mfem.ParBilinearForm(fespace);                           # Initialize the bilinear form.
        K.AddDomainIntegrator(mfem.ConvectionIntegrator(velocity, -1.0));                   # Sets k to the bilinear form B(\phi, \eta) = (\phi, -v(X) * \nabla \eta)
        K.AddInteriorFaceIntegrator(
            mfem.TransposeIntegrator(mfem.DGTraceIntegrator(velocity, 1.0, -0.5)));         # Adds \int_{F} \hat{U_j(X)}(v(X) \cdot n) \phi_i(X) for faces between elements
        K.AddBdrFaceIntegrator( 
            mfem.TransposeIntegrator(mfem.DGTraceIntegrator(velocity, 1.0, -0.5)));         # Adds the term \int_{F} \hat{U_j(X)}(v(X) \cdot n) \phi_i(X) for boundary faces

        # Define a linear form corresponding to the vector b in the docstring above. 
        b : mfem.ParLinearForm = mfem.ParLinearForm(fespace);                               # Initialize the linear form.
        b.AddBdrFaceIntegrator(mfem.BoundaryFlowIntegrator(g, velocity, -1.0, -0.5));       # Set b(\phi) = \int_{Bd(\Omega)} g(v(X) \cdot \phi(X))

        # Assemble the forms.
        M.Assemble();                                                                       # Computes B(\phi_i, \phi_j) = (\phi_i, \phi_j) for all i, j using the basis functions in fespace.
        M.Finalize();                                                                       # completes any needed preprocessing.
        skip_zeros = 0; 
        K.Assemble(skip_zeros);                                                             # Computes B(\phi_i, \phi_j) = (\phi_i, -v(X) * \nabla \eta_j) for all i, j using the basis functions in fespace, eliminating any rows/columns corresponding to boundary conditions
        K.Finalize(skip_zeros);                                                             # Completes any needed preprocessing (e.g. imposing Dirichlet zeros).
        b.Assemble();                                                                       # Computes b(\phi_i) = \int_{Bd(\Omega)} g(v(X) \cdot \phi_i(X)) using the basis functions in fespace.

        # Build matrices to hold M, K, and b in the expression above
        self.Mmat : mfem.HypreParMatrix = M.ParallelAssemble();
        self.Kmat : mfem.HypreParMatrix = K.ParallelAssemble();
        self.bvec : mfem.HypreParVector = b.ParallelAssemble();

        # Initialize the solver and preconditioner for Mult (see below).
        self.M_prec     : mfem.HypreSmoother    = mfem.HypreSmoother();
        self.M_prec.SetType(mfem.HypreSmoother.Jacobi);

        self.M_solver   : mfem.CGSolver         = mfem.CGSolver(self.Mmat.GetComm());
        self.M_solver.SetPreconditioner(self.M_prec);           # Tells the CG solver "the linear system I want to solve is Mx = b.
        self.M_solver.SetOperator(self.Mmat);
        self.M_solver.iterative_mode = False;                   
        self.M_solver.SetRelTol(1e-9);                          # Says "stop when the relative residual is < 1e-9"
        self.M_solver.SetAbsTol(0.0);                           # Says "no absolute‐residual stopping condition."
        self.M_solver.SetMaxIter(100);                          # Sets the maximum number of cg iterations.
        self.M_solver.SetPrintLevel(0);                         # Silences all CG output.

        self.z          : mfem.Vector           = mfem.Vector(M.Height());



    def Mult(self, x : mfem.Vector, y : mfem.Vector) -> None:
        # Solve the following equation for U'(t).
        #    M * U'(t) = -K(U(t)) + b

        # Compute K*U(t), store the result to self.z. This sets self.z to K(U(t))
        self.Kmat.Mult(x, self.z)

        # Add b to K(U(t)).
        self.z += self.bvec

        # Solve for U'(t) in M U'(t) = K U(t) + b, store the result in y. 
        self.M_solver.Mult(self.z, y)



# -------------------------------------------------------------------------------------------------
# Simulate function
# -------------------------------------------------------------------------------------------------

def Simulate(   meshfile_name       : str           = "periodic-hexagon.mesh", 
                ser_ref_levels      : int           = 2,
                par_ref_levels      : int           = 1,
                order               : int           = 3,
                ode_solver_type     : int           = 4,
                t_Grid              : numpy.ndarray = numpy.linspace(0, 5.0, 501),
                Positions           : numpy.ndarray = numpy.empty(0),
                g                   : float         = numpy.pi/2,
                k                   : float         = 1.0,
                w                   : float         = 1.0,
                serialization_steps : int           = 2,
                num_positions       : int           = 1000,
                VisIt               : bool          = True) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """     
    This examples solves a time dependent advection problem of the form

        (d/dt)u(t, X) + v(X) * \nabla(u(t, X)) = 0,
    
    where v(X) is a velocity field. We also impose the following initial conditions:
        
        u(0, (x, y))        =  exp(-k*(x~^2 + y~^2)) * sin(pi*w*x~) * sin(pi*w*y~)


    See the "Initial_Displacement" class for more details. We solve this PDE, then return the 
    solution at each time step. 

        

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    meshfile_name : str
        specifies the mesh file to use. This should specify a file in the Physics/PyMFEM/data 
        subdirectory.

    ser_ref_levels : int   
        specifies the number of times to refine the serial mesh uniformly.

    par_ref_levels : int 
        specifies the number of times to refine each parallel mesh.

    order : int 
        specifies the finite element order (polynomial degree of the basis functions).

    ode_solver_type : int 
        specifies which ODE solver we should use
            1   - Backward Euler
            2   - RK2
            4   - RK4
            6   - RK6

    t_Grid : numpy.ndarray, shape = (Nt)
        specifies the time grid. We simulate the dynamics from t_Grid[0] to t_Grid[-1]; we assume 
        that the elements of t_Grid form an increasing sequence.

    Positions : numpy.ndarray, shape = (2, num_positions)
        An optional argument. If empty, we generate new positions from scratch. If it is not empty, 
        then Positions should be a 2D array whose i'th row holds the position of the i'th position 
        at which we evaluate the solution.

    g : float 
        specifies the rotation speed of the velocity field (this becomes the gamma variable in 
        the EvalValue method in the velocity_coeff class).

    w : float
        specifies the frequency of peaks in the initial condition (this becomes the freq variable 
        in the EvalValue method in the Initial_Displacement class).

    k : float
        Specifies the rate of decay in the initial condition (this becomes the decay variable in 
        the EvalValue method in the Initial_Displacement class).

    serialization_steps : int
        Specifies how frequently we serialize (save) and visualize the solution.
    
    num_positions : int
        Specifies the number of positions at which we will evaluate the solution.
        
    VisIt : bool
        If True, will prompt the code to save the displacement and velocity GridFunctions every 
        time we serialize them. It will save the GridFunctions in a format that VisIt 
        (visit.llnl.gov) can understand/work with.
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Sol, X, T, bb_min, bb_max
    
    Sol : numpy.ndarray, shape = (Nt, 1, num_positions)
        i, j, k element holds the j'th component of the solution at the k'th position (i.e., 
        X[i, :]) at the i'th time step (i.e., T[i]).

    X : numpy.ndarray, shape = (2, num_positions)
        i'th row holds the position of the i'th position at which we evaluate the solution.
    
    T : numpy.ndarray, shape = (Nt)
        i'th element holds the i'th time at which we evaluate the solution.

    bb_min : numpy.ndarray, shape = (2,)
        The minimum coordinates of the bounding box.

    bb_max : numpy.ndarray, shape = (2,)
        The maximum coordinates of the bounding box.
    """

    if(Positions.size > 0):
        assert(isinstance(Positions, numpy.ndarray));
        assert(len(Positions.shape)     == 2);
        assert(Positions.shape[0]       == 2);
        assert(Positions.shape[1]       == num_positions);


    # ---------------------------------------------------------------------------------------------
    # 1. Setup 

    LOGGER.info("Setting up advection simulation with MFEM.");

    # Fetch thread information.
    comm                        = MPI.COMM_WORLD;
    myid                : int   = comm.Get_rank();
    num_procs           : int   = comm.Get_size();

    # Define the ODE solver used for time integration. Several explicit Runge-Kutta methods are 
    # available.
    LOGGER.debug("Selecting the ODE solver");
    ode_solver : mfem.ODESolver; 
    if ode_solver_type == 1:
        ode_solver = mfem.ForwardEulerSolver();
    elif ode_solver_type == 2:
        ode_solver = mfem.RK2Solver(1.0);
    elif ode_solver_type == 4:
        ode_solver = mfem.RK4Solver();
    elif ode_solver_type == 6:
        ode_solver = mfem.RK6Solver();
    else:
        print("Unknown ODE solver type: " + str(ode_solver_type));
        exit();



    # ---------------------------------------------------------------------------------------------
    # 2. Setup the mesh. Read the serial mesh from the given mesh file on all processors. We can 
    # handle geometrically periodic meshes in this code.

    # Load the mesh.
    if(myid == 0): LOGGER.debug("Loading the mesh and its properties");
    meshfile_path   : str   = expanduser(join(dirname(__file__), 'data', meshfile_name));
    mesh                    = mfem.Mesh(meshfile_path, 1, 1);
    dim             : int   = mesh.Dimension();

    # Report
    if(myid == 0): LOGGER.debug("meshfile_path = %s" % meshfile_path);
    if(myid == 0): LOGGER.debug("dim = %d" % dim);

    # Serially refine the mesh.
    if(myid == 0): LOGGER.debug("Refining the mesh");
    for lev in range(ser_ref_levels):
        mesh.UniformRefinement();
        if mesh.NURBSext:
            mesh.SetCurvature(max(order, 1));
        bb_min, bb_max = mesh.GetBoundingBox(max(order, 1));
    

    # Setup the parallel mesh and refine it.
    if(myid == 0): LOGGER.debug("Setting up the parallel mesh");
    pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh);
    for i in range(par_ref_levels):
        pmesh.UniformRefinement();



    # ---------------------------------------------------------------------------------------------
    # 3. Define the discontinuous DG finite element space of the given
    #    polynomial order on the refined mesh.

    if(myid == 0): LOGGER.debug("Setting up the FEM space.");
    fec         : mfem.DG_FECollection          = mfem.DG_FECollection(order, dim, mfem.BasisType.GaussLobatto);
    fespace     : mfem.ParFiniteElementSpace    = mfem.ParFiniteElementSpace(pmesh, fec);

    global_vSize : int = fespace.GlobalTrueVSize();
    if(myid == 0): LOGGER.info("Number of unknowns: " + str(global_vSize));

    # Setup the grid function to hold the initial condition.
    if(myid == 0): LOGGER.debug("Setting up the grid function to hold the initial condition.");
    u_gf    : mfem.ParGridFunction                  = mfem.ParGridFunction(fespace);



    # ---------------------------------------------------------------------------------------------
    # 4. Define the initial condition objects.
    
    if(myid == 0): LOGGER.debug("Setting up the coefficient objects.");
    velocity    = velocity_coeff(dim, bb_min, bb_max, g);
    inflow      = inflow_coeff();
    u0          = Initial_Displacement(bb_min, bb_max, k, w); 

    # Project the initial condition onto the finite element space.
    u_gf.ProjectCoefficient(u0);
    U       : mfem.HypreParVector        = u_gf.GetTrueDofs();



    # ---------------------------------------------------------------------------------------------
    # 5. Set up positions at which we will evaluate the solution.

    if(Positions.size == 0):
        if(myid == 0): LOGGER.info("Sampling %d positions in the mesh" % num_positions);
    else:
        if(myid == 0): LOGGER.info("Verifying the columns of Positions are in the problem domain");

    # Figure out the maximum/minimum x and y coordinates of the mesh.
    bb_min, bb_max = mesh.GetBoundingBox();
    if(myid == 0): LOGGER.debug("The bounding box for the mesh is given by bb_min = %s, bb_max = %s" % (str(bb_min), str(bb_max)));
    x_min   : float = bb_min[0];
    x_max   : float = bb_max[0];
    y_min   : float = bb_min[1];
    y_max   : float = bb_max[1];
    if(myid == 0): LOGGER.debug("x_min = %f, x_max = %f, y_min = %f, y_max = %f" % (x_min, x_max, y_min, y_max));

    # If we are sampling new points, then we sample num_positions points evenly spaced between 
    # x_min and x_max, and y_min and y_max. If the mesh has an unusual shape, some of these points 
    # may lie outside the mesh. We sample too many positions to account for this. Any points that 
    # lie outside the mesh will be ignored. We will sample new points if the number of points that 
    # lie inside the mesh is less than num_positions.
    Valid_Positions_List : list[numpy.ndarray] = [];
    Elements_List        : list[int]           = [];
    RefCoords_List       : list[numpy.ndarray] = [];
    num_valid_positions  : int = 0;
    
    while(num_valid_positions < num_positions):
        if(Positions.size == 0):
            if(myid == 0): LOGGER.debug("Sampling %d positions" % num_positions);

            # Sample random x,y coordinates
            x_positions : numpy.ndarray = numpy.random.uniform(x_min, x_max, num_positions);
            y_positions : numpy.ndarray = numpy.random.uniform(y_min, y_max, num_positions);

            # Create array of points in format expected by FindPoints
            points : numpy.ndarray = numpy.column_stack((x_positions, y_positions));

        else:
            points = Positions.T;

        # Find which points are in the mesh.
        count, elem_list, ref_coords = pmesh.FindPoints(points, warn = False, inv_trans = None);

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
            if(Positions.size > 0):
                if(myid == 0): LOGGER.error("%d/%d elements of Positions are invalid. Aborting" % (num_valid_positions, num_positions));
                raise ValueError("Invalid Positions");
            else:
                if(myid == 0): LOGGER.debug("Not enough valid positions (current = %d, needed = %d), sampling again" % (num_valid_positions, num_positions));

    # Convert the lists to numpy arrays.
    Positions : numpy.ndarray = numpy.array(Valid_Positions_List).T;
    Elements  : numpy.ndarray = numpy.array(Elements_List);
    RefCoords : numpy.ndarray = numpy.array(RefCoords_List);
    if(myid == 0): LOGGER.debug("Positions has shape %s (dim = %d, num_positions = %d)" % (str(Positions.shape), dim, Positions.shape[1]));
    


    # ---------------------------------------------------------------------------------------------
    # 6. Setup lists to store the solution + evaluate the initial solution at the positions.

    if(myid == 0): LOGGER.debug("Setting up lists to store the time, solution at each time step.");

    # Setup for time stepping.
    t_list              : list[float]           = [];
    u_list              : list[numpy.ndarray]   = [];

    # Evaluate the initial solution at the positions.
    u_Positions_0       = numpy.zeros((1, num_positions));
    for i in range(num_positions):
        u_Positions_0[0, i]     = u_gf.GetValue(Elements[i], RefCoords[i], dim);

    # Append the initial solution and time to their corresponding lists.
    t_list.append(t_Grid[0]);
    u_list.append(  u_Positions_0);



    # ---------------------------------------------------------------------------------------------
    # 7. VisIt

    # Setup VisIt visualization (if we are doing that)
    if (VisIt == True):
        LOGGER.info("Setting up VisIt visualization.");

        # Create the VisIt data collection.
        visit_dc_path   : str                       = os.path.join(os.path.join(os.path.dirname(__file__), "VisIt"), "Advection-fom");
        visit_dc        : mfem.VisItDataCollection  = mfem.VisItDataCollection(visit_dc_path, pmesh);
        visit_dc.SetPrecision(8);

        # Register U and its time derivative.
        visit_dc.RegisterField("U",    u_gf);

        # Set the cycle and time.
        visit_dc.SetCycle(0);
        visit_dc.SetTime(0.0);
        visit_dc.Save();



    # --------------------------------------------------------------------------------------------- 
    # 8.  Perform time-integration.

    if(myid == 0): LOGGER.info("Running time stepping from t = %f to t = %f with %d time steps" % (t_Grid[0], t_Grid[-1], len(t_Grid)));

    # Setup the ODE solver.
    adv = AdvectionOperator(fespace = fespace, velocity = velocity, g = inflow);

    # Initialize the ODE solver.
    ode_solver.Init(adv);

    # Run the time stepping loop.
    for t_idx in range(1, len(t_Grid)):
        # Step the ODE solver.
        t, dt = ode_solver.Step(U, t_Grid[t_idx - 1], t_Grid[t_idx] - t_Grid[t_idx - 1]);
        u_gf.Assign(U);

        # Should we serialize?
        Last_Step : bool = (t_idx == len(t_Grid) - 1);
        if (Last_Step or (t_idx % serialization_steps == 0)):
            if(myid == 0): LOGGER.debug("time step: " + str(t_idx) + ", time: " + str(numpy.round(t, 3)) + ", dt: " + str(numpy.round(dt, 3)));

            # Update the solution to the grid functions
            u_gf.Assign(U);

            # Evaluate the solution at the positions.
            u_Positions_t       = numpy.zeros((1, num_positions));
            for i in range(num_positions):
                u_Positions_t[0, i]     = u_gf.GetValue(Elements[i], RefCoords[i], dim);

            # Append the current solution and time to their corresponding lists.
            u_list.append(  u_Positions_t);
            t_list.append(t);

            # If visualizing, Save the solution to the VisIt object.
            if(VisIt):
                # Save the mesh, solution, and time.
                visit_dc.SetCycle(t_idx);
                visit_dc.SetTime(t);
                visit_dc.Save();



    # ---------------------------------------------------------------------------------------------
    # 9. Package everything up for returning.

    # Turn times, displacements, velocities lists into arrays.
    Trajectory  : numpy.ndarray = numpy.array(u_list,               dtype = numpy.float32);
    Times       : numpy.ndarray = numpy.array(t_list,               dtype = numpy.float32);

    return Trajectory, Positions, Times, bb_min, bb_max;


if __name__ == "__main__":
    Logging.Initialize_Logger(level = logging.DEBUG);
    Sol, X, T, bb_min, bb_max = Simulate();
