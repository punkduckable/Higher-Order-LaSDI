# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import      os;
import      sys;
import      logging;

import      mfem.par    as      mfem;
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
# Telegraphers Operator class
# -------------------------------------------------------------------------------------------------

class TelegraphersOperator(mfem.SecondOrderTimeDependentOperator):
    def __init__(self, fespace : mfem.FiniteElementSpace, ess_bdr : mfem.intArray, c : float, alpha : float) -> None:
        """
        This class implements the Telegraphers operator. The Telegraphers operator is given by:
            
              (d^2/dt^2)u(t, X) - c^2*laplacian(u(t, X)) + 2*\alpha*(d/dt)u(t, X) = 0
            
        where u is the scalar field, c is the wave speed, and m is the mass of the field. Let 
        { \phi_i } be the basis functions in the finite element space fespace. Then the weak form 
        of the Telegrapher's operator is given by:

            (\phi_i, (d^2/dt^2) u) + c^2*(\nabla \phi_i, \nabla u) + 2*\alpha*(\phi_i, (d/dt) u) = 0
        
        If we assume the solution is of the form u(x, t) = \sum_{j = 1}^{N} \phi_j(x) U_j(t), then 
        the weak form of the Telegrapher's operator becomes:
        
            (\phi_i, \sum_{j = 1}^{N} \phi_j(x) U_j''(t)) + c^2*(\nabla \phi_i, \nabla \sum_{j = 1}^{N} \phi_j(x) U_j(t)) + 2*\alpha*(\phi_i, \sum_{j = 1}^{N} \phi_j(x) U_j'(t)) = 0

        This engenders the following system of equations:

            M*U''(t) + K*U(t) + A*U'(t) = 0

        where U(t) is the vector whose j'th entry is U_j(t). Likewise, M, K, and A are the matrices
        defined by  

            M_{ij}  = (\phi_i, \phi_j)
            K_{ij}  = c^2*(\nabla \phi_i, \nabla \phi_j)
            A_{ij}  = 2*\alpha*(\phi_i, \phi_j)

        where \Omega is the domain.
        

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

        c : float   
            The speed of the wave in the direction of the wave's propagation. See the Telegrapher's 
            equation above.

        alpha : float
            The  of the field. See the Telegrapher's equation above.
        """ 

        # ---------------------------------------------------------------------------------------------
        # 1. Initialize the bilinear forms and matrices.

        # Run the super class initializer. Note that fespace.GetTrueVSize() returns the number of 
        # "true" unknown DOFs after removing those that are constrained by Dirichlet BCs.
        mfem.SecondOrderTimeDependentOperator.__init__(self, fespace.GetTrueVSize(), 0.0); 

        # Get the essential boundary conditions. This ess_bdr is an array of length equal to the 
        # number of boundary attributes. For each boundary attribute a, if ess_bdr[a] == 1 then 
        # that face has u = 0 (Dirichlet) boundary condition. We store the indices of the DOFs 
        # that are constrained by Dirichlet BCs in self.ess_tdof_list.
        self.ess_tdof_list : mfem.intArray = mfem.intArray();
        fespace.GetEssentialTrueDofs(ess_bdr, self.ess_tdof_list);

        # Define the bilinear form corresponding K in the weak form of the Telegrapher's equation 
        # (see the docstring above).
        c2  : mfem.ConstantCoefficient  = mfem.ConstantCoefficient(c*c);                # c^2
        K   : mfem.BilinearForm         = mfem.BilinearForm(fespace);                   # Initialize the bilinear form.
        K.AddDomainIntegrator(mfem.DiffusionIntegrator(c2));                            # Sets K to the bilinear form B(u, v) = c^2*(\nabla u, \nabla v)
        K.Assemble();                                                                   # Computes B(\phi_i, \phi_j) = c^2*(\nabla \phi_i, \nabla \phi_j) for all i, j using the basis functions in fespace.

        # Build a matrix to hold K.
        self.Kmat0 = mfem.SparseMatrix();                                               # Initialize a matrix to hold K. This one is used to build K without setting the essential boundary conditions.
                                                                                        # We need this version when computing the RHS of the linear system (need products of full matrices before zeroing out the BCs).
        self.Kmat  = mfem.SparseMatrix();                                               # Initialize a matrix to hold K. This one is used to build K with the essential boundary conditions set.
                                                                                        # We need this version when we compute T in ImplicitSolve.
        Kdummy     = mfem.intArray();
        K.FormSystemMatrix(Kdummy, self.Kmat0);                                         # This populates Kmat0 such that the i,j'th entry is B(\phi_i, \phi_j) = c^2*(\nabla \phi_i, \nabla \phi_j).
        K.FormSystemMatrix(self.ess_tdof_list, self.Kmat);                              # Does the same thing, but with all of the rows/columns corresponding to essential boundary conditions removed.

        # Define the bilinear form corresponding to M in the weak form of the Telegrapher's equation 
        # (see the docstring above).
        M = mfem.BilinearForm(fespace);                                                 # Initialize the bilinear form.
        M.AddDomainIntegrator(mfem.MassIntegrator());                                   # Sets M to the bilinear form B(u, v) = (u, v)
        M.Assemble();                                                                   # Computes B(\phi_i, \phi_j) = (\phi_i, \phi_j) for all i, j using the basis functions in fespace.
        
        # Build a matrix to hold M.        
        self.Mmat = mfem.SparseMatrix();                                                # Initialize a matrix to hold M.
        M.FormSystemMatrix(self.ess_tdof_list, self.Mmat);                              # Build M, stores it in self.Mmat.

        # Define the bilinear form corresponding to A in the weak form of the Telegrapher's equation 
        # (see the docstring above).
        two_alpha   : mfem.ConstantCoefficient  = mfem.ConstantCoefficient(2*alpha);    # 2 alpha
        A           : mfem.BilinearForm         = mfem.BilinearForm(fespace);           # Initialize the bilinear form.
        A.AddDomainIntegrator(mfem.MassIntegrator(two_alpha));                          # Sets A to the bilinear form B(u, v) = 2*\alpha*(u, v)
        A.Assemble();                                                                   # Computes B(\phi_i, \phi_j) = 2\alpha*(\phi_i, \phi_j) for all i, j using the basis functions in fespace.
        
        # Build a matrix to hold A.
        self.Amat0 = mfem.SparseMatrix();                                               # Initialize a matrix to hold A. This one is used to build K without setting the essential boundary conditions. 
                                                                                        # We need this version when computing the RHS of the linear system (need products of full matrices before zeroing out the BCs).
        self.Amat  = mfem.SparseMatrix();                                               # Initialize a matrix to hold A. This one is used to build A with the essential boundary conditions set. 
                                                                                        # We need this version when we compute T in ImplicitSolve.
        Adummy     = mfem.intArray();
        A.FormSystemMatrix(Adummy, self.Amat0);                                         # Build A, stores it in self.Amat.
        A.FormSystemMatrix(self.ess_tdof_list, self.Amat);                              #Does the same thing, but with all of the rows/columns corresponding to essential boundary conditions removed.

        # In older MFEM (< 4.7.1), FormSystemMatrix(dummy, Kmat0) and/or 
        # A.FormSystemMatrix(Adummy, self.Amat0) may not have worked earlier. 
        # This block simply ensures that self.Kmat0 and self.Amat0 always exists, even in older 
        # releases.
        if mfem_version < 471:
            Kdummy      : mfem.intArray     = mfem.intArray();
            self.Kmat0  : mfem.SparseMatrix = mfem.SparseMatrix();
            K.FormSystemMatrix(Kdummy, self.Kmat0);
        
            Adummy      : mfem.intArray     = mfem.intArray();
            self.Amat0  : mfem.SparseMatrix = mfem.SparseMatrix();
            A.FormSystemMatrix(Adummy, self.Amat0);
            
        # Store the K, M, and A matrices. 
        self.K  : mfem.BilinearForm = K;                                                # i,j entry is c^2*(\nabla \phi_i, \nabla \phi_j)
        self.M  : mfem.BilinearForm = M;                                                # i,j entry is (\phi_i, \phi_j) 
        self.A  : mfem.BilinearForm = A;                                                # i,j entry is 2*alpha*(\phi_i, \phi_j)

        # Define the relative tolerance (for the solver)
        rel_tol : float = 1e-8;



        # ---------------------------------------------------------------------------------------------
        # 2. Initialize the solvers and preconditioners.

        # The Telegrapher's solver needs to invert the mass matrix M many times (e.g., if you do a 
        # Newmark or implicit‐time‐stepping scheme), and it also needs some solver/preconditioner for 
        # whatever "T" operator you will define later. (In many implementations, "T" is something 
        # like M + a*K or some combination used in the time‐update step.) In this code, we 
        # initialize two separate CG (Conjugate Gradient) solvers, each with a diagonal smoother 
        # (DSmoother) as a preconditioner.

        # Initialize the solver and preconditioner for M. See Mult below.
        M_solver     : mfem.CGSolver        = mfem.CGSolver();
        M_prec       : mfem.DSmoother       = mfem.DSmoother();
        M_solver.iterative_mode = False;
        M_solver.SetRelTol(rel_tol);                        # Says "stop when the relative residual is < rel_tol"
        M_solver.SetAbsTol(0.0);                            # says "no absolute‐residual stopping condition.""
        M_solver.SetMaxIter(30);                            # Sets the maximum number of iterations.
        M_solver.SetPrintLevel(0);                          # Silences all CG output.
        M_solver.SetPreconditioner(M_prec);                 # Attaches a diagonal smoother as a pre-conditioner to speed up the solver.
        M_solver.SetOperator(self.Mmat);                    # Tells the CG solver "the linear system I want to solve is Mx = b where M = self.Mmat.
        self.M_prec     : mfem.DSmoother    = M_prec;       # Stash M_solver and M_prec so that whenever the time‐integrator needs to compute M^{-1}b, it can just call M_solver.Mult(b, x).
        self.M_solver   : mfem.CGSolver     = M_solver;   

        # Initialize the solver and preconditioner for T = M + fac0*K (see ImplicitSolve below).
        T_solver        : mfem.CGSolver     = mfem.CGSolver();
        T_prec          : mfem.DSmoother    = mfem.DSmoother();
        T_solver.iterative_mode = False;                    # Tells the CG solver "I want to solve a linear system Tx = b where T = M + fac0*K."
        T_solver.SetRelTol(rel_tol);                        # Says "stop when the relative residual is < rel_tol"
        T_solver.SetAbsTol(0.0);                            # says "no absolute‐residual stopping condition.""
        T_solver.SetMaxIter(100);                           # Sets the maximum number of iterations.
        T_solver.SetPrintLevel(0);                          # Silences all CG output.     
        T_solver.SetPreconditioner(T_prec);                 # Attaches a diagonal smoother as a pr-conditioner to speed up the solver.
        self.T_prec     : mfem.DSmoother    = T_prec;       # Stash T_solver and T_prec so that whenever the time‐integrator needs to compute T^{-1}b, it can just call T_solver.Mult(b, x).
        self.T_solver   : mfem.CGSolver     = T_solver;  
        self.T          : mfem.SparseMatrix = None;         # Initialize T to None. This will force us to build T = M + fac0*K in ImplicitSolve.


        
    def Mult(self, u : mfem.Vector, du_dt : mfem.Vector, d2udt2 : mfem.Vector) -> None:
        # Solve the following equation for U''(t):
        #    M * U''(t) = -K*U(t) - A * U'(t)

        # Compute -K*U(t) 
        KU          = mfem.Vector(u.Size());        # Initialize a vector to hold K*U(t).
        self.Kmat.Mult(u,       KU);                # Computes K*U(t) and stores it in KU.
        KU.Neg();                                   # Now holds -K*U(t)    

        # Compute -A*U'(t)
        AdUdt       = mfem.Vector(u.Size());        # Initialize a vector to hold A*U'(t).
        self.Amat.Mult(du_dt,   AdUdt);             # Computes A*U'(t) and stores it in adUdt.                
        AdUdt.Neg();                                # Now holds -A*U'(t)

        # Add the two together
        KU_AdUdt    = mfem.Vector(u.Size());    
        mfem.add_vector(KU, 1.0, AdUdt, KU_AdUdt);  # KU_AdUdt = -K*U - A*U'(t) = KU + 1.0*AdUdt. 

        # Solves M * U''(t) = -K*U(t) - A*U'(t) for U''(t).
        self.M_solver.Mult(KU_AdUdt, d2udt2);



    def ImplicitSolve(self, fac0 : float, fac1 : float, u : mfem.Vector, dudt : mfem.Vector, d2udt2 : mfem.Vector) -> None:
        # Solve the following equation for U''(t):
        #    (M + fac0*K + fac1*A)*U''(t) = -K*U(t) - A*U'(t)

        if self.T is None:
            # Start with T = M + fac0*K.
            self.T : mfem.SparseMatrix = mfem.Add(1.0, self.Mmat, fac0, self.Kmat);

            # Add fac1*A to get T = M + fac0*K + fac1*A
            self.T.Add(fac1, self.Amat)
            
            # Hand T to the linear solver
            self.T_solver.SetOperator(self.T);

        # Compute -K*U(t)
        KU = mfem.Vector(u.Size());                 # Initialize a vector to hold -K*U(t).
        self.Kmat0.Mult(u, KU);                     # Computes K*U(t), stores it in KU
        KU.Neg();                                   # Now holds -K*U(t)

        # Compute -A U'(t)
        AdUdt = mfem.Vector(u.Size());              # Initialize a vector to hold -A*U'(t)
        self.Amat0.Mult(dudt, AdUdt);               # Computes A*U'(t), stores it in AdUdt
        AdUdt.Neg();                                # Now holds -A*U'(t)

        # Compute = -K*U(t) - M2 * U(t), store in z.
        KU_AdUdt : mfem.Vector = mfem.Vector(u.Size());
        mfem.add_vector(KU, 1.0, AdUdt, KU_AdUdt);  # KU_AdUdt = KU + AdUdt = -K*U(t) - A*U'(t)

        # Set the essential boundary conditions to zero. This is necessary because the stiffness 
        # matrix is not symmetric.
        for j in self.ess_tdof_list:
            KU_AdUdt[j] = 0.0;

        # Solves (M + fac0*K + fac1*A) * U''(t) = -K*U(t) - A*U'(t) for U''(t).
        self.T_solver.Mult(KU_AdUdt, d2udt2);
    


    def SetParameters(self, u):
        self.T = None


class cInitialSolution(mfem.PyCoefficient):
    def EvalValue(self, X : numpy.ndarray) -> float:    
        global decay, freq;
        norm2 : float = numpy.sum(numpy.square(X));
        return numpy.exp(-decay*norm2) * numpy.sin(numpy.pi * freq * X[0]) * numpy.sin(numpy.pi * freq * X[1]);



class cInitialRate(mfem.PyCoefficient):
    def EvalValue(self, x : numpy.ndarray) -> float:
        return 0;








# -------------------------------------------------------------------------------------------------
# Simulate function
# -------------------------------------------------------------------------------------------------

def Simulate(mesh_file          : str           = "hexagon.mesh",
             ref_levels         : int           = 3,
             order              : int           = 2,
             ode_solver_type    : int           = 10,
             t_final            : float         = 5.0,
             dt                 : float         = .01,
             Positions          : numpy.ndarray = None,
             c                  : float         = 0.2,
             alpha              : float         = 0.2,
             k                  : float         = 3.0,
             w                  : float         = 2.0,
             dirichlet          : bool          = True,
             serialization_steps: int           = 1,
             num_positions      : int           = 1000,
             VisIt              : bool          = True) -> tuple[numpy.ndarray,numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Simulate the Telegraphers equation:

        (d^2/dt^2)u(t, X) - c^2*laplacian(u(t, X)) + 2*\alpha*(d/dt)u(t, X) = 0

    We also impose the following initial conditions:
        
        u(0, (x, y))        = exp(-k*(x^2 + y^2)) * sin(pi*w*x) * sin(pi*w*y)
        (d/dt)u(0, (x, y))  = 0

    We solve this PDE, then return the solution at each time step. 

    

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
        The final time. We solve the Telegrapher's equation from t = 0 to t = t_final.

    dt : float
        The time step. We solve the Telegrapher's equation using a time-stepping scheme with time step dt.
    
    Positions : numpy.ndarray, shape = (2, num_positions)
        An optional argument. If None, we generate new positions from scratch. If it is not None, 
        then Positions should be a 2D array whose i'th row holds the position of the i'th position 
        at which we evaluate the solution.

    c : float
        The speed of the wave. See the Telegrapher's equation in the TelegraphersOperator class.

    alpha : float
        See the Telegrapher's equation in the TelegraphersOperator class.

    k : float
        specifies the decay rate of in the initial condition. 

    w : float
        A constant used to specify the freuqnecy of peaks in the initial condition.
        
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

    if(Positions is not None):
        assert(isinstance(Positions, numpy.ndarray));
        assert(len(Positions.shape)     == 2);
        assert(Positions.shape[0]       == 2);
        assert(Positions.shape[1]       == num_positions);  


    # ---------------------------------------------------------------------------------------------
    # 1. Setup 
    
    LOGGER.info("Setting up Telegrapher's equation simulation with MFEM.");
    
    # Set the global variable decay.
    global decay, freq;
    decay   = k;
    freq    = w;

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

    LOGGER.debug("Defining the finite element space and grid functions to hold U(t) and U'(t).");
    fe_coll : mfem.FiniteElementCollection  = mfem.H1_FECollection(order, dim);         # Basis functions
    fespace : mfem.FiniteElementSpace       = mfem.FiniteElementSpace(mesh, fe_coll);   # FEM space (span of basis functions).

    # Get the number of degrees of freedom in the finite element space.
    fe_size : int = fespace.GetTrueVSize();
    LOGGER.info("Number of unknowns to solve for: %d" % fe_size);

    # Initialize the grid functions for the solution and the time derivative of the solution.
    u_gf    : mfem.GridFunction = mfem.GridFunction(fespace);
    dudt_gf : mfem.GridFunction = mfem.GridFunction(fespace);



    # ---------------------------------------------------------------------------------------------
    # 4. Set the initial conditions for U(t) and U'(t).

    LOGGER.debug("Setting the initial conditions for U and U'(t).");

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
                                            # values of U'(t) at the nodes) for U'(t).
    dudt_gf.GetTrueDofs(dudt);



    # ---------------------------------------------------------------------------------------------
    # 5. Initialize the Telegrapher's operator.

    LOGGER.debug("Initializing the Telegrapher's operator.");

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

    # Initialize the Telegrapher's operator.
    oper : TelegraphersOperator = TelegraphersOperator(fespace = fespace, ess_bdr = ess_bdr, c = c, alpha = alpha);



    # ---------------------------------------------------------------------------------------------
    # 6. Set up positions at which we will evaluate the solution.

    if(Positions is None):
        LOGGER.info("Sampling %d positions in the mesh" % num_positions);
    else:
        LOGGER.info("Verifying the columns of Positions are in the problem domain");

    # Figure out the maximum/minimum x and y coordinates of the mesh.
    bb_min, bb_max = mesh.GetBoundingBox();
    LOGGER.debug("The bounding box for the mesh is given by bb_min = %s, bb_max = %s" % (str(bb_min), str(bb_max)));
    x_min   : float = bb_min[0];
    x_max   : float = bb_max[0];
    y_min   : float = bb_min[1];
    y_max   : float = bb_max[1];
    LOGGER.debug("x_min = %f, x_max = %f, y_min = %f, y_max = %f" % (x_min, x_max, y_min, y_max));

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
        if(Positions is None):
            LOGGER.debug("Sampling %d positions" % num_positions);

            # Sample random x,y coordinates
            x_positions : numpy.ndarray = numpy.random.uniform(x_min, x_max, num_positions);
            y_positions : numpy.ndarray = numpy.random.uniform(y_min, y_max, num_positions);

            # Create array of points in format expected by FindPoints
            points : numpy.ndarray = numpy.column_stack((x_positions, y_positions));

        else:
            points = Positions.T;

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
            if(Positions is not None):
                LOGGER.error("%d/%d elements of Positions are invalid. Aborting" % (num_valid_positions, num_positions));
                raise ValueError("Invalid Positions");
            else:
                LOGGER.debug("Not enough valid positions (current = %d, needed = %d), sampling again" % (num_valid_positions, num_positions));

    # Convert the lists to numpy arrays.
    Positions : numpy.ndarray = numpy.array(Valid_Positions_List).T;
    Elements  : numpy.ndarray = numpy.array(Elements_List);
    RefCoords : numpy.ndarray = numpy.array(RefCoords_List);
    LOGGER.debug("Positions has shape %s (dim = %d, num_positions = %d)" % (str(Positions.shape), dim, Positions.shape[1]));


    # ---------------------------------------------------------------------------------------------
    # 7. Setup lists to store the solution + evaluate the initial solution at the positions.

    LOGGER.debug("Setting up lists to store the time, U, and DtU at each time step.");

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
    # 8. VisIt

    # Store the initial solution to u_gf and dudt_gf.
    u_gf.SetFromTrueDofs(u);
    dudt_gf.SetFromTrueDofs(dudt);

    if(VisIt):
        LOGGER.info("Setting up VisIt visualization.");

        # Create the VisIt data collection.
        visit_dc_path   : str                       = os.path.join(os.path.join(os.path.dirname(__file__), "VisIt"), "Telegraphers-fom");
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
    # 9. Perform time-integration (looping over the time iterations, ti, with a time-step dt).

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
            LOGGER.debug("time step: " + str(ti) + ", time: " + str(numpy.round(t, 3)));

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

        # Update the Telegrapher's operator with the current solution.
        oper.SetParameters(u);



    # ---------------------------------------------------------------------------------------------
    # 10. Package everything up for returning.

    # Turn times, displacements, velocities lists into arrays.
    Times           = numpy.array(times_list,           dtype = numpy.float32);
    U               = numpy.array(displacements_list,   dtype = numpy.float32);
    DtU             = numpy.array(velocities_list,      dtype = numpy.float32);

    # Return everything
    return U, DtU, Positions, Times;



if __name__ == "__main__":
    Logging.Initialize_Logger(level = logging.DEBUG);
    U, DtU, X, T = Simulate();
