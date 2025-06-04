# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import      os;
import      sys;
import      logging;
from        os.path                 import  expanduser, join, dirname;

from        mfem.common.arg_parser  import  ArgParser;
import      mfem.par                as      mfem;
from        mfem.par                import  intArray, add_vector, Add;
from        mpi4py                  import  MPI 
import      numpy;
from        numpy                   import  sqrt, pi, cos, sin, hypot, arctan2;
from        scipy.special           import  erfc;

utils_path : str        = os.path.join(os.path.join(os.path.pardir, os.path.pardir), "Utilities");
sys.path.append(utils_path);
import      Logging;

# Setup logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Nonlinear Elasticity Classes
# -------------------------------------------------------------------------------------------------


class InitialVelocity(mfem.VectorPyCoefficient):
    """
    Defines the initial velocity field for the nonlinear elasticity problem. The velocity is zero 
    in all dimensions except the last, where it follows a sinusoidal pattern.
    """
    def EvalValue(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Evaluates the initial velocity at position x.

        

        -----------------------------------------------------------------------------------------------
        Arguments
        -----------------------------------------------------------------------------------------------
        x : numpy.ndarray
            Position vector at which to evaluate the velocity.

            
            
        -----------------------------------------------------------------------------------------------
        Returns
        -----------------------------------------------------------------------------------------------
        v : numpy.ndarray
            Initial velocity vector at position x. Has same dimension as x. Only the last component 
            is non-zero, given by: -(s/80.0)*sin(s*x[0]) where s is a global parameter controlling 
            the oscillation frequency.
        """
        dim : int = len(x)
        
        # Access global parameter s that controls oscillation frequency
        global s

        # Initialize velocity vector to zeros
        v : numpy.ndarray = numpy.zeros(len(x))
        
        # Set only the last component to be non-zero
        v[-1] = -(s/80.0) * sin(s * x[0])
        return v


class InitialDeformation(mfem.VectorPyCoefficient):
    """
    Defines the initial deformation field for the nonlinear elasticity problem. The initial 
    deformation is simply the identity map (no deformation).
    """
    def EvalValue(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Evaluates the initial deformation at position x.

        

        -----------------------------------------------------------------------------------------------
        Arguments
        -----------------------------------------------------------------------------------------------
        x : numpy.ndarray
            Position vector at which to evaluate the deformation.

            
        
        -----------------------------------------------------------------------------------------------
        Returns
        -----------------------------------------------------------------------------------------------
        y : numpy.ndarray
            Initial deformation vector at position x. Returns a copy of x, representing no initial 
            deformation.
        """
        from copy import deepcopy
        y : numpy.ndarray = deepcopy(x)
        return y


class ElasticEnergyCoefficient(mfem.PyCoefficient):
    """
    Defines the elastic energy density coefficient for the nonlinear elasticity problem. This 
    coefficient is used to compute the elastic energy at each point in the domain based on the 
    deformation gradient and a constitutive model.

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------
    model : mfem.PyNonlinearFormIntegrator
        The constitutive model that defines the strain energy density function.
    
    x : mfem.GridFunction
        The deformation field whose energy we want to compute.
    """
    def __init__(self, model: mfem.PyNonlinearFormIntegrator, x: mfem.GridFunction):
        self.x       : mfem.GridFunction                 = x;
        self.model   : mfem.PyNonlinearFormIntegrator    = model;
        self.J       : mfem.DenseMatrix                  = mfem.DenseMatrix();
        mfem.PyCoefficient.__init__(self);



    def Eval(self, T: mfem.ElementTransformation, ip: mfem.IntegrationPoint) -> float:
        """
        Evaluates the elastic energy density at a given point.

        

        -----------------------------------------------------------------------------------------------
        Arguments
        -----------------------------------------------------------------------------------------------
        T : mfem.ElementTransformation
            The transformation from reference to physical coordinates.
        
        ip : mfem.IntegrationPoint
            The integration point at which to evaluate the energy.

            

        -----------------------------------------------------------------------------------------------
        Returns
        -----------------------------------------------------------------------------------------------
        float
            The elastic energy density divided by the Jacobian determinant.
        """
        # Set the current transformation
        self.model.SetTransformation(T);
        
        # Get the deformation gradient
        self.x.GetVectorGradient(T, self.J);
        
        # Return elastic energy density divided by Jacobian determinant
        return self.model.EvalW(self.J)/(self.J.Det());



class HyperelasticOperator(mfem.PyTimeDependentOperator):
    def __init__(self, fespace: mfem.ParFiniteElementSpace, ess_tdof_list_: list[int], visc: float, mu: float, K: float) -> None:
        """
        Initialize the hyperelastic operator.

        Parameters
        ----------
        fespace : mfem.ParFiniteElementSpace
            The finite element space
        ess_tdof_list_ : list[int] 
            List of essential true degrees of freedom
        visc : float
            Viscosity coefficient
        mu : float
            Shear modulus
        K : float
            Bulk modulus
        """
        # Initialize parent class with size = 2*fespace.TrueVSize() and t=0.0
        mfem.PyTimeDependentOperator.__init__(self, 2*fespace.TrueVSize(), 0.0)

        # Set solver parameters
        rel_tol             : float = 1e-8;
        skip_zero_entries   : int   = 0;
        ref_density         : float = 1.0;

        # Store input parameters
        self.ess_tdof_list  : list[int] = ess_tdof_list_;
        self.z              : mfem.Vector = mfem.Vector(self.Height() // 2);
        self.z2             : mfem.Vector = mfem.Vector(self.Height() // 2);
        self.H_sp           : mfem.Vector = mfem.Vector(self.Height() // 2);
        self.dvxdt_sp       : mfem.Vector = mfem.Vector(self.Height() // 2);
        self.fespace        : mfem.ParFiniteElementSpace = fespace;
        self.viscosity      : float = visc;

        # Initialize bilinear and nonlinear forms
        M : mfem.ParBilinearForm    = mfem.ParBilinearForm(fespace);
        S : mfem.ParBilinearForm    = mfem.ParBilinearForm(fespace);
        H : mfem.ParNonlinearForm   = mfem.ParNonlinearForm(fespace);
        self.M : mfem.ParBilinearForm    = M;
        self.H : mfem.ParNonlinearForm    = H;
        self.S : mfem.ParBilinearForm     = S;

        # Setup mass matrix M
        rho: mfem.ConstantCoefficient = mfem.ConstantCoefficient(ref_density)
        M.AddDomainIntegrator(mfem.VectorMassIntegrator(rho))
        M.Assemble(skip_zero_entries)
        M.Finalize(skip_zero_entries)
        self.Mmat: mfem.HypreParMatrix = M.ParallelAssemble()
        self.Mmat.EliminateRowsCols(self.ess_tdof_list)

        # Setup mass matrix solver
        M_solver: mfem.CGSolver = mfem.CGSolver(fespace.GetComm())
        M_prec: mfem.HypreSmoother = mfem.HypreSmoother()
        M_solver.iterative_mode = False
        M_solver.SetRelTol(rel_tol)
        M_solver.SetAbsTol(0.0)
        M_solver.SetMaxIter(30)
        M_solver.SetPrintLevel(0)
        M_prec.SetType(mfem.HypreSmoother.Jacobi)
        M_solver.SetPreconditioner(M_prec)
        M_solver.SetOperator(self.Mmat)

        self.M_solver = M_solver
        self.M_prec = M_prec

        model = mfem.NeoHookeanModel(mu, K)
        H.AddDomainIntegrator(mfem.HyperelasticNLFIntegrator(model))
        H.SetEssentialTrueDofs(self.ess_tdof_list)
        self.model = model

        visc_coeff = mfem.ConstantCoefficient(visc)
        S.AddDomainIntegrator(mfem.VectorDiffusionIntegrator(visc_coeff))
        S.Assemble(skip_zero_entries)
        S.Finalize(skip_zero_entries)
        self.Smat = mfem.HypreParMatrix()
        S.FormSystemMatrix(self.ess_tdof_list, self.Smat)

    def Mult(self, vx, dvx_dt):
        sc = self.Height() // 2
        v = mfem.Vector(vx, 0,  sc)
        x = mfem.Vector(vx, sc,  sc)
        dv_dt = mfem.Vector(dvx_dt, 0, sc)
        dx_dt = mfem.Vector(dvx_dt, sc,  sc)

        self.H.Mult(x, self.z)
        self.H_sp.Assign(self.z)

        if (self.viscosity != 0.0):
            self.Smat.Mult(v, self.z2)
            self.z += self.z2
        
        self.z.Neg()
        self.M_solver.Mult(self.z, dv_dt)

        dx_dt.Assign(v) # this changes dvx_dt
        self.dvxdt_sp.Assign(dvx_dt)

    def ElasticEnergy(self, x):
        return self.H.GetEnergy(x)

    def KineticEnergy(self, v):
        from mpi4py import MPI
        local_energy = 0.5*self.M.InnerProduct(v, v)
        energy = MPI.COMM_WORLD.allreduce(local_energy, op=MPI.SUM)
        return energy

    def GetElasticEnergyDensity(self, x, w):
        w_coeff = ElasticEnergyCoefficient(self.model, x)
        w.ProjectCoefficient(w_coeff)




# -------------------------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------------------------

def Simulate(   meshfile_name   : str           = "beam-quad.mesh", 
                ser_ref_levels  : int           = 2,
                par_ref_levels  : int           = 0,
                order           : int           = 2,
                ode_solver_type : int           = 14,
                t_final         : float         = 100.0,
                time_step_size  : float         = 0.02,
                viscosity       : float         = 1e-2,
                shear_modulus   : float         = 0.25, 
                bulk_modulus    : float         = 5.0,
                num_positions   : int           = 1000,
                theta           : float         = 1.0,
                serialize_steps : int           = 10, 
                VisIt           : bool          = True) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    This examples solves a time dependent nonlinear elasticity problem of the form 

        (d/dt)v(X, t)   = H(d(X, t)) + S v(X, t), 
        (d/dt)d(X, t)   = v(X, t),
    
    where H is a hyperelastic model and S is a viscosity operator of Laplacian type. We also impose 
    with the following initial conditions:
        
        d((x, y), 0)         =  (x, y)
        v((x, y), 0)         =  (-theta*x^2, theta*x^2 (8.0 - x))
    
    where X[0] and X[-1] are the positions of the first and last nodes, respectively. Here, theta 
    is a parameter that the user can change. 
    
    See the c++ version of example 10 in the MFEM library for more detail.

    We solve this PDE, then return the solution at each time step. 

        

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
            2   - SDIRK2
            3   - SDIRK3
            11  - Forward Euler
            12  - RK2
            13  - RK3 SSP
            14  - RK4
            22  - ImplicitMidpointSolver
            23  - SDIRK23Solver
            24  - SDIRK34Solver
    
    t_final : float
        specifies the final time. We simulate the dynamics from the start time to the final time. 
        The start time is 0.

    time_step_size : float 
        specifies the time step size.

    viscosity : float
        specifies the viscosity coefficient.

    shear_modulus : float
        specifies the shear modulus in the Neo-Hookean hyperelastic model.

    bulk_modulus : float
        specifies the bulk modulus in the Neo-Hookean hyperelastic model.

    num_positions : int
        specifies the number of positions at which we will evaluate the solution.

    theta : float
        specifies the constant "theta" in the initial velocity.

    serialize_steps : int
        Specifies how frequently we serialize (save) and visualize the solution.

    VisIt : bool
        If True, will prompt the code to save the displacement and velocity GridFunctions every 
        time we serialize them. It will save the GridFunctions in a format that VisIt 
        (visit.llnl.gov) can understand/work with.
    
        
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    D, V, X, T. 
    
    D : numpy.ndarray, shape = (Nt, 2, num_positions)
        i, j, k element holds the j'th component of the displacement at the k'th position (i.e., 
        X[i, :]) at the i'th time step (i.e., T[i]).
    
    V : numpy.ndarray, shape = (Nt, 2, num_positions)
        i, j, k element holds the j'th component of the velocity at the k'th position (i.e., X[i]) 
        at the i'th time step (i.e., T[i]).

    X : numpy.ndarray, shape = (2, num_positions)
        i'th row holds the position of the i'th node at which we evaluate the solution.
    
    T : numpy.ndarray, shape = (Nt)
        i'th element holds the j'th time at which we evaluate the solution.
    """
    


    # ---------------------------------------------------------------------------------------------
    # 1. Setup 

    LOGGER.info("Setting up non-linear elasticity simulation with MFEM.");

    # Fetch thread information.
    comm                = MPI.COMM_WORLD
    myid        : int   = comm.Get_rank()
    num_procs   : int   = comm.Get_size()

    # Set variable aliases.
    dt          : float = time_step_size;
    visc        : float = viscosity;
    mu          : float = shear_modulus;
    K           : float = bulk_modulus;
    global s;
    s = theta;
    if(myid == 0): LOGGER.info("Simulating with theta = %f" % theta);

    # Select the ODE solver.
    LOGGER.debug("Selecting the ODE solver");
    if ode_solver_type == 1:
        ode_solver = mfem.BackwardEulerSolver();
    elif ode_solver_type == 2:
        ode_solver = mfem.SDIRK23Solver(2);
    elif ode_solver_type == 3:
        ode_solver = mfem.SDIRK33Solver();
    elif ode_solver_type == 11:
        ode_solver = mfem.ForwardEulerSolver();
    elif ode_solver_type == 12:
        ode_solver = mfem.RK2Solver(0.5);
    elif ode_solver_type == 13:
        ode_solver = mfem.RK3SSPSolver();
    elif ode_solver_type == 14:
        ode_solver = mfem.RK4Solver();
    elif ode_solver_type == 22:
        ode_solver = mfem.ImplicitMidpointSolver();
    elif ode_solver_type == 23:
        ode_solver = mfem.SDIRK23Solver();
    elif ode_solver_type == 24:
        ode_solver = mfem.SDIRK34Solver();
    else:
        print("Unknown ODE solver type: " + str(ode_solver_type));
        exit;
    

    # ---------------------------------------------------------------------------------------------
    # 2. Set up the mesh. 

    # Setup the mesh.
    if(myid == 0): LOGGER.debug("Lading the mesh and its properties");
    meshfile_path   : str   = expanduser(join(dirname(__file__), 'data', meshfile_name));
    mesh                    = mfem.Mesh(meshfile_path, 1, 1);
    dim             : int   = mesh.Dimension();
    if(myid == 0): LOGGER.debug("meshfile_path = %s" % meshfile_path);
    if(myid == 0): LOGGER.debug("dim = %d" % dim);

    # Refine the mesh 
    if(myid == 0): LOGGER.debug("Refining mesh");
    for lev in range(ser_ref_levels):
        mesh.UniformRefinement();
    
    # Now define a parallel mesh by a partitioning of the serial mesh. Refine this mesh further in 
    # parallel to increase the resolution. Once the parallel mesh is defined, the serial mesh can 
    # be deleted.
    pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    del mesh
    for x in range(par_ref_levels):
        pmesh.UniformRefinement();



    # ---------------------------------------------------------------------------------------------
    # 3. Define the vector finite element spaces representing the mesh
    #    deformation x, the velocity v, and the initial configuration, x_ref.
    #    Define also the elastic energy density, w, which is in a discontinuous
    #    higher-order space. Since x and v are integrated in time as a system,
    #    we group them together in block vector vx, with offsets given by the
    #    fe_offset array.

    if(myid == 0): LOGGER.info("Setting up the FEM space.");
    fe_coll             = mfem.H1_FECollection(order, dim);                 # Basis functions
    fespace             = mfem.ParFiniteElementSpace(pmesh, fe_coll, dim);  # FEM space (span of basis functions).
    glob_size   : int   = fespace.GlobalTrueVSize();
    if(myid == 0): LOGGER.info('Number of velocity/deformation unknowns: ' + str(glob_size));

    true_size   : int   = fespace.TrueVSize()
    true_offset         = intArray([0, true_size, 2*true_size]);

    # Setup the grid functions for displacement and velocity. Note that we use a BlockVector to 
    # hold the velocity and displacement because the ODE solvers expect a single vector.
    VD      = mfem.BlockVector(true_offset);
    V_gf    = mfem.ParGridFunction(fespace);
    D_gf    = mfem.ParGridFunction(fespace);
    V_gf.MakeTRef(fespace, VD, true_offset[0]);
    D_gf.MakeTRef(fespace, VD, true_offset[1]);
    
    # Get the nodes of the mesh.
    D_ref = mfem.ParGridFunction(fespace);
    pmesh.GetNodes(D_ref);
    
    # Elastic energy density.
    w_fec       : mfem.FiniteElementCollection  = mfem.L2_FECollection(order + 1, dim);
    w_fespace   : mfem.ParFiniteElementSpace    = mfem.ParFiniteElementSpace(pmesh, w_fec);
    w_gf        : mfem.ParGridFunction          = mfem.ParGridFunction(w_fespace);



    # ---------------------------------------------------------------------------------------------
    # 3. Set the initial conditions for v and x, and the boundary conditions on
    #    a beam-like mesh (see description above).

    if(myid == 0): LOGGER.info("Settng initial and boundary conditions");

    # Set up objects to hold the ICs
    if(myid == 0): LOGGER.debug("Setting up objects to hold the initial conditions;");
    velo        = InitialVelocity(dim);
    deform      = InitialDeformation(dim);
    
    # Project the initial velocity and deformation onto the FEM space.
    V_gf.ProjectCoefficient(velo);
    V_gf.SetTrueVector();

    # Project the initial deformation onto the FEM space.
    D_gf.ProjectCoefficient(deform);
    D_gf.SetTrueVector();

    # Set the grid functions to the true vectors (i.e., the vectors that hold the values of the 
    # velocity and deformation at the nodes of the mesh).
    V_gf.SetFromTrueVector();
    D_gf.SetFromTrueVector();

    # Get the true vectors (i.e., the vectors that hold the values of the velocity and deformation 
    # at the nodes of the mesh).
    V_gf.GetTrueDofs(VD.GetBlock(0));
    D_gf.GetTrueDofs(VD.GetBlock(1));

    # Impose boundary conditions.
    if(myid == 0): LOGGER.debug("Imposing Boundary Conditions");
    ess_bdr = intArray(fespace.GetMesh().bdr_attributes.Max());
    ess_bdr.Assign(0);
    ess_bdr[0] = 1;

    # Get the essential true dofs.
    ess_tdof_list = mfem.intArray()
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)



    # ---------------------------------------------------------------------------------------------
    # 4. Define HyperelasticOperator and initialize it the initial energies.
    
    if(myid == 0): LOGGER.info("Setting up Hyperelastic operator.");

    oper = HyperelasticOperator(fespace, ess_bdr, visc, mu, K);
    ee0 = oper.ElasticEnergy(D_gf);
    ke0 = oper.KineticEnergy(V_gf);

    if(myid == 0): LOGGER.info("initial elastic energy (EE) = " + str(ee0));
    if(myid == 0): LOGGER.info("initial kinetic energy (KE) = " + str(ke0));
    if(myid == 0): LOGGER.info("initial   total energy (TE) = " + str(ee0 + ke0));



    # ---------------------------------------------------------------------------------------------
    # 5. Extract the positions of the nodes.

    if(myid == 0): LOGGER.info("Sampling %d positions in the mesh" % num_positions);

    # Figure out the maximum/minimum x and y coordinates of the mesh.
    bb_min, bb_max = pmesh.GetBoundingBox();
    x_min   : float = bb_min[0];
    x_max   : float = bb_max[0];
    y_min   : float = bb_min[1];
    y_max   : float = bb_max[1];
    if(myid == 0): LOGGER.debug("x_min = %f, x_max = %f, y_min = %f, y_max = %f" % (x_min, x_max, y_min, y_max));

    # Now, sample num_positions points evenly spaced between x_min and x_max, and y_min and y_max.
    # If the mesh has an unusal shape, some of these points may lie outside the mesh. We sample 
    # too many positions to account for this. Any points that lie outside the mesh will be ignored.
    # We will sample new points if the number of points that lie inside the mesh is less than 
    # num_positions.
    Valid_Positions_List : list[numpy.ndarray] = [];
    elements_list        : list[int]           = [];
    ref_coords_list      : list[numpy.ndarray] = [];
    num_valid_positions  : int = 0;
    
    while(num_valid_positions < num_positions):
        if(myid == 0): LOGGER.debug("Sampling %d positions" % num_positions);

        # Sample random x,y coordinates
        x_positions : numpy.ndarray = numpy.random.uniform(x_min, x_max, num_positions);
        y_positions : numpy.ndarray = numpy.random.uniform(y_min, y_max, num_positions);

        # Create array of points in format expected by FindPoints
        points : numpy.ndarray = numpy.column_stack((x_positions, y_positions));

        # Find which points are in the mesh.
        count, elem_list, ref_coords = pmesh.FindPoints(points, warn = False, inv_trans = None);

        # Check which points are inside elements
        for i in range(num_positions):
            if elem_list[i] >= 0:  # -1 indicates point not found in any element
                Valid_Positions_List.append(points[i]);
                elements_list.append(elem_list[i]);
                ref_coords_list.append(ref_coords[i]);
                num_valid_positions += 1;

                # If we have enough valid positions, break.
                if num_valid_positions >= num_positions:
                    break;

        # If we have not enough valid positions, sample again.
        if(num_valid_positions < num_positions):
            if(myid == 0): LOGGER.debug("Not enough valid positions (current = %d, needed = %d), sampling again" % (num_valid_positions, num_positions));

    # Convert the list of valid positions to a numpy array.
    Positions : numpy.ndarray = numpy.array(Valid_Positions_List).T;
    Elements  : numpy.ndarray = numpy.array(elements_list);
    RefCoords : numpy.ndarray = numpy.array(ref_coords_list);
    if(myid == 0): LOGGER.debug("Positions has shape %s (dim = %d, num_positions = %d)" % (str(Positions.shape), dim, Positions.shape[1]));



    # ---------------------------------------------------------------------------------------------
    # 6. VisIt

    # Setup VisIt visualization (if we are doing that)
    if (VisIt == True):
        if(myid == 0): LOGGER.info("Setting up VisIt visualization.");

        dc_path : str   = os.path.join(os.path.join(os.path.dirname(__file__), "VisIt"), "nlelast-fom");
        dc              = mfem.VisItDataCollection(dc_path, pmesh);
        dc.SetPrecision(8);
        # // To save the mesh using MFEM's parallel mesh format:
        # // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
        dc.RegisterField("Disp",    D_gf);
        dc.RegisterField("Vel",     V_gf);
        dc.SetCycle(0);
        dc.SetTime(0.0);
        dc.Save();


    # ---------------------------------------------------------------------------------------------
    # 7. Setup lists to store the solution + evaluate the initial solution at the positions.

    LOGGER.info("Setting up lists to store the time, U, and DtU at each time step.");

    # Setup for time stepping.
    times_list          : list[float]           = [];    
    displacements_list  : list[numpy.ndarray]   = [];
    velocities_list     : list[numpy.ndarray]   = [];

    # Evaluate the initial solution at the positions.
    D_Positions_0               = numpy.zeros((2, num_positions));
    V_Positions_0               = numpy.zeros((2, num_positions));
    for i in range(num_positions):
        D_Positions_0[:, i]     = D_gf.GetValue(Elements[i], RefCoords[i], dim);
        V_Positions_0[:, i]     = V_gf.GetValue(Elements[i], RefCoords[i], dim);

    # Append the initial U, DtU, and time to their corresponding lists.
    times_list.append(0);
    displacements_list.append(  D_Positions_0);
    velocities_list.append(     V_Positions_0);



    # ---------------------------------------------------------------------------------------------
    # 7. Perform time-integration (looping over the time iterations, ti, with a time-step dt).
    
    if(myid == 0): LOGGER.info("Running time stepping from t = 0 to t = %f with dt %f" % (t_final, dt));

    # Time step!!!!!
    ode_solver.Init(oper);
    t           : float = 0.0;
    ti          : int   = 1;        # counter to keep track of when we should serialize solution.
    last_step   : bool  = False;

    while not last_step:
        # Check if we should stop time stepping (if this time step is within dt/2 of t_final.
        if (t + dt >= t_final - dt/2):
            last_step = True;

        t, dt = ode_solver.Step(VD, t, dt)
        ti = ti + 1;

        # Should we serialize?
        if (last_step or (ti % serialize_steps) == 0):
            # Find energy.
            ee = oper.ElasticEnergy(D_gf);
            ke = oper.KineticEnergy(V_gf);

            # Print the energy.
            if(myid == 0):
                text : str  = ( "step " + str(ti) + ", t = " + str(t) + ", EE = " +
                                str(ee) + ", KE = " + str(ke) +
                                ", dTE = " + str((ee + ke) - (ee0 + ke0)));
                LOGGER.info(text);

            # Update the solution to the grid functions
            V_gf.Assign(VD.GetBlock(0));
            D_gf.Assign(VD.GetBlock(1));

            # Evaluate the solution at the positions.
            D_Positions_t       = numpy.zeros((2, num_positions));
            V_Positions_t       = numpy.zeros((2, num_positions));
            for i in range(num_positions):
                D_Positions_t[:, i]     = D_gf.GetValue(Elements[i], RefCoords[i], dim);
                V_Positions_t[:, i]     = V_gf.GetValue(Elements[i], RefCoords[i], dim);

            # Append the current U, DtU, and time to their corresponding lists.
            times_list.append(t);
            displacements_list.append(  D_Positions_t);
            velocities_list.append(     V_Positions_t);
    
            # If visualizing, Save the solution to the VisIt object.
            if VisIt:
                # Set the mesh to the current displacement
                pmesh.SwapNodes(D_gf, 0);

                dc.SetCycle(ti);
                dc.SetTime(t);
                dc.Save();  
                
                # Now swap the deformed mesh back to reset everything.
                pmesh.SwapNodes(D_gf, 0);        



    # ---------------------------------------------------------------------------------------------
    # 7. Package everything up for returning.

    # Turn times, displacements, velocities lists into arrays.
    Times           = numpy.array(times_list,           dtype = numpy.float32);
    Displacements   = numpy.array(displacements_list,   dtype = numpy.float32);
    Velocities      = numpy.array(velocities_list,      dtype = numpy.float32);


    return Displacements, Velocities, Positions, Times;




if __name__ == "__main__":
    Logging.Initialize_Logger(level = logging.INFO);
    D, V, X, T = Simulate();

    print("D.shape = " + str(D.shape));
    print("V.shape = " + str(V.shape));
    print("X.shape = " + str(X.shape));
    print("T.shape = " + str(T.shape));
    print("D[:, 0, 0] = " + str(D[:, 0, 0]));
    print("V[:, 0, 0] = " + str(V[:, 0, 0]));
    print("X[:, 0] = " + str(X[:, 0]));
    exit();