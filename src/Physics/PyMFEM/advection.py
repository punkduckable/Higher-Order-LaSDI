# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

from        mfem                    import  path;
import      mfem.par                as      mfem;
from        mfem.par                import  intArray;
from        mpi4py                  import  MPI;
import      numpy;
from        numpy                   import  sqrt, pi, cos, sin, hypot, arctan2;
from        scipy.special           import  erfc;

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
# Advection classes
# -------------------------------------------------------------------------------------------------

class velocity_coeff(mfem.VectorPyCoefficient):
    def __init__(self, dim : int, bb_min : float, bb_max : float, problem : int):
        # Run the super class initializer
        mfem.VectorPyCoefficient.__init__(self, dim);

        # Now set problem specific attributes.
        self.bb_min     : float = bb_min;
        self.bb_max     : float = bb_max;
        self.problem    : int   = problem;


    def EvalValue(self, x):
        dim = len(x)

        center = (self.bb_min + self.bb_max)/2.0
        # map to the reference [-1,1] domain
        X = 2 * (x - center) / (self.bb_max - self.bb_min)
        if self.problem == 0:
            if dim == 1:
                v = [1.0, ]
            elif dim == 2:
                v = [sqrt(2./3.), sqrt(1./3)]
            elif dim == 3:
                v = [sqrt(3./6.), sqrt(2./6), sqrt(1./6.)]
        elif (self.problem == 1 or self.problem == 2):
            # Clockwise rotation in 2D around the origin
            w = pi/2
            if dim == 1:
                v = [1.0, ]
            elif dim == 2:
                v = [w*X[1],  - w*X[0]]
            elif dim == 3:
                v = [w*X[1],  - w*X[0],  0]
        elif (self.problem == 3):
            # Clockwise twisting rotation in 2D around the origin
            w = pi/2
            d = max((X[0]+1.)*(1.-X[0]), 0.) * max((X[1]+1.)*(1.-X[1]), 0.)
            d = d ** 2
            if dim == 1:
                v = [1.0, ]
            elif dim == 2:
                v = [d*w*X[1],  - d*w*X[0]]
            elif dim == 3:
                v = [d*w*X[1],  - d*w*X[0],  0]
        return v



class u0_coeff(mfem.PyCoefficient):
    def __init__(self, bb_min : float, bb_max : float, problem : int):
        # Run the super class initializer
        mfem.PyCoefficient.__init__(self);

        # Now set problem specific attributes.
        self.bb_min : float = bb_min;
        self.bb_max : float = bb_max;
        self.problem : int   = problem;


    def EvalValue(self, x):
        dim = len(x)

        center = (self.bb_min + self.bb_max)/2.0
        # map to the reference [-1,1] domain
        X = 2 * (x - center) / (self.bb_max - self.bb_min)
        if (self.problem == 0 or self.problem == 1):
            if dim == 1:
                return numpy.exp(-40. * (X[0]-0.5)**2)
            elif (dim == 2 or dim == 3):
                rx  : float = 0.45
                ry  : float = 0.25
                cx  : float = 0.
                cy  : float = -0.2
                w   : float = 10.
                if dim == 3:
                    s = (1. + 0.25*cos(2 * pi * x[2]))
                    rx = rx * s
                    ry = ry * s
                return (erfc(w * (X[0]-cx-rx)) * erfc(-w*(X[0]-cx+rx)) *
                        erfc(w * (X[1]-cy-ry)) * erfc(-w*(X[1]-cy+ry)))/16

        elif self.problem == 2:
            rho = hypot(x[0], x[1])
            phi = arctan2(x[1], x[0])
            return (sin(pi * rho) ** 2) * sin(3*phi)
        elif self.problem == 3:
            return sin(pi * X[0]) * sin(pi * X[1])

        return 0.0



# Inflow boundary condition (zero for the problems considered in this example)
class inflow_coeff(mfem.PyCoefficient):
    def EvalValue(self, x):
        return 0



class FE_Evolution(mfem.PyTimeDependentOperator):
    def __init__(self, M, K, b):
        mfem.PyTimeDependentOperator.__init__(self, M.Height())

        self.M_prec = mfem.HypreSmoother()
        self.M_solver = mfem.CGSolver(M.GetComm())
        self.z = mfem.Vector(M.Height())

        self.K = K
        self.M = M
        self.b = b
        self.M_prec.SetType(mfem.HypreSmoother.Jacobi)
        self.M_solver.SetPreconditioner(self.M_prec)
        self.M_solver.SetOperator(M)
        self.M_solver.iterative_mode = False
        self.M_solver.SetRelTol(1e-9)
        self.M_solver.SetAbsTol(0.0)
        self.M_solver.SetMaxIter(100)
        self.M_solver.SetPrintLevel(0)


#    def EvalMult(self, x):
#        if you want to impolement Mult in using python objects,
#        such as numpy.. this needs to be implemented and don't
#        overwrite Mult


    def Mult(self, x, y):
        self.K.Mult(x, self.z)
        self.z += self.b
        self.M_solver.Mult(self.z, y)



# -------------------------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------------------------


def Simulate(   meshfile_name       : str       = "periodic-square.mesh", 
                ser_ref_levels      : int       = 2,
                par_ref_levels      : int       = 0,
                order               : int       = 3,
                ode_solver_type     : int       = 4,
                t_final             : float     = 10.0,
                time_step_size      : float     = 0.01,
                serialization_steps : int       = 10, 
                num_positions       : int       = 100,
                VisIt               : bool      = True) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
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
            2   - RK2
            3   - RK3
            4   - RK4
            6   - RK6
    
    t_final : float
        specifies the final time. We simulate the dynamics from the start time to the final time. 
        The start time is 0.

    time_step_size : float 
        specifies the time step size.


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

    Sol, X, T. 
    
    Sol : numpy.ndarray, shape = (Nt, 1, num_positions)
        i, j, k element holds the j'th component of the solution at the k'th position (i.e., 
        X[i, :]) at the i'th time step (i.e., T[i]).

    X : numpy.ndarray, shape = (2, num_positions)
        i'th row holds the position of the i'th position at which we evaluate the solution.
    
    T : numpy.ndarray, shape = (Nt)
        i'th element holds the j'th time at which we evaluate the solution.
    """

    # ---------------------------------------------------------------------------------------------
    # 1. Setup 

    LOGGER.info("Setting up advection simulation with MFEM.");

    # Fetch thread information.
    comm                        = MPI.COMM_WORLD;
    myid                : int   = comm.Get_rank();
    num_procs           : int   = comm.Get_size();
    verbose             : bool  = (myid == 0);

    # Set variables.
    problem             : int   = 0;
    dt                  : float = time_step_size;

    # Set the device.
    device : mfem.Device = mfem.Device('cpu');
    if myid == 0:
        device.Print();

    # Define the ODE solver used for time integration. Several explicit Runge-Kutta methods are 
    # available.
    LOGGER.debug("Selecting the ODE solver");
    ode_solver = None;  
    if ode_solver_type == 1:
        ode_solver = mfem.ForwardEulerSolver();
    elif ode_solver_type == 2:
        ode_solver = mfem.RK2Solver(1.0);
    elif ode_solver_type == 3:
        ode_solver = mfem.RK3SSolver();
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
    for k in range(par_ref_levels):
        pmesh.UniformRefinement();



    # ---------------------------------------------------------------------------------------------
    # 3. Define the discontinuous DG finite element space of the given
    #    polynomial order on the refined mesh.

    if(myid == 0): LOGGER.info("Setting up the FEM space.");
    fec : mfem.DG_FECollection          = mfem.DG_FECollection(order, dim, mfem.BasisType.GaussLobatto);
    fes : mfem.ParFiniteElementSpace    = mfem.ParFiniteElementSpace(pmesh, fec);

    global_vSize : int = fes.GlobalTrueVSize();
    if(myid == 0): LOGGER.info("Number of unknowns: " + str(global_vSize));


    # ---------------------------------------------------------------------------------------------
    # 4. Define the coefficient objects.
    
    if(myid == 0): LOGGER.info("Setting up the coefficient objects.");
    velocity    = velocity_coeff(dim, bb_min, bb_max, problem);
    inflow      = inflow_coeff();
    u0          = u0_coeff(bb_min, bb_max, problem); 



    # ---------------------------------------------------------------------------------------------
    # 5. Set up and assemble the bilinear and linear forms corresponding to the DG discretization. 
    # The DGTraceIntegrator involves integrals over mesh interior faces.

    # Setup the bilinear forms.
    LOGGER.debug("Setting up the bilinear forms.");
    m : mfem.ParBilinearForm = mfem.ParBilinearForm(fes);
    m.AddDomainIntegrator(mfem.MassIntegrator());
    k : mfem.ParBilinearForm = mfem.ParBilinearForm(fes);
    k.AddDomainIntegrator(mfem.ConvectionIntegrator(velocity, -1.0))
    k.AddInteriorFaceIntegrator(
        mfem.TransposeIntegrator(mfem.DGTraceIntegrator(velocity, 1.0, -0.5)))
    k.AddBdrFaceIntegrator(
        mfem.TransposeIntegrator(mfem.DGTraceIntegrator(velocity, 1.0, -0.5)))

    # Setup the linear forms.
    LOGGER.debug("Setting up the linear forms.");
    b : mfem.ParLinearForm = mfem.ParLinearForm(fes);
    b.AddBdrFaceIntegrator(
        mfem.BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5))

    # Assemble the forms.
    LOGGER.debug("Assembling the bilinear forms.");
    m.Assemble()
    m.Finalize()
    skip_zeros = 0
    k.Assemble(skip_zeros)
    k.Finalize(skip_zeros)
    b.Assemble()

    # Parallel assemble the forms.
    LOGGER.debug("Parallel assembling the bilinear forms.");
    M : mfem._par.hypre.HypreParMatrix = m.ParallelAssemble();
    K : mfem._par.hypre.HypreParMatrix = k.ParallelAssemble();
    B : mfem._par.hypre.HypreParMatrix = b.ParallelAssemble();



    # ---------------------------------------------------------------------------------------------
    # 6. Setup the a grid function to hold the solution.

    if(myid == 0): LOGGER.debug("Setting up the grid function to hold the initial condition.");
    u_gf    : mfem.ParGridFunction                  = mfem.ParGridFunction(fes);
    u_gf.ProjectCoefficient(u0);
    U       : mfem._par.hypre.HypreParVector        = u_gf.GetTrueDofs();

    # Save the initial solution.
    if(myid == 0): LOGGER.debug("Saving the initial solution.");
    smyid       : str = '{:0>6d}'.format(myid);
    mesh_name   : str = "ex9-mesh." + smyid;
    sol_name    : str = "ex9-init." + smyid;
    pmesh.Print(mesh_name, 8);
    u_gf.Save(sol_name, 8);



    # ---------------------------------------------------------------------------------------------
    # 7. Set up positions at which we will evaluate the solution.

    if(myid == 0): LOGGER.info("Setting up Positions");

    # Figure out the maximum/minimum x and y coordinates of the mesh.
    bb_min, bb_max = pmesh.GetBoundingBox();
    x_min   : float = bb_min[0];
    x_max   : float = bb_max[0];
    y_min   : float = bb_min[1];
    y_max   : float = bb_max[1];
    if(myid == 0): LOGGER.debug("x_min = %f, x_max = %f, y_min = %f, y_max = %f" % (x_min, x_max, y_min, y_max));

    # Now, sample num_positions points evenly spaced between x_min and x_max, and y_min and y_max.
    x_positions     : numpy.ndarray = numpy.random.uniform(x_min, x_max, num_positions);
    y_positions     : numpy.ndarray = numpy.random.uniform(y_min, y_max, num_positions);
    Positions       : numpy.ndarray = numpy.row_stack((x_positions, y_positions));
    Num_Positions   : int           = Positions.shape[1];
    if(myid == 0): LOGGER.debug("Positions has shape %s (dim = %d, num_positions = %d)" % (str(Positions.shape), dim, num_positions));


    # ---------------------------------------------------------------------------------------------
    # 8. VisIt

    # Setup VisIt visualization (if we are doing that)
    if (VisIt == True):
        if(myid == 0): LOGGER.info("Setting up VisIt visualization.");

        dc_path : str   = os.path.join(os.path.join(os.path.dirname(__file__), "VisIt"), "nlelast-fom");
        dc              = mfem.VisItDataCollection(dc_path, pmesh);
        dc.SetPrecision(8);
        # // To save the mesh using MFEM's parallel mesh format:
        # // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
        dc.RegisterField("u",    u_gf);
        dc.SetCycle(0);
        dc.SetTime(0.0);
        dc.Save();


    # --------------------------------------------------------------------------------------------- 
    # 9.  Perform time-integration 

    # Setup the ODE solver.
    adv : FE_Evolution = FE_Evolution(M, K, B);

    # Setup the lists to hold the solution at each time step.
    times_list          : list[float]           = [];    
    u_list              : list[numpy.ndarray]   = [];

    # Append the initial time
    times_list.append(0);

    # Evaluate the initial solution at the positions.
    element_nums        = pmesh.FindPoints(Positions.T);
    u_Positions_0       = numpy.zeros((1, Num_Positions));

    for i in range(Num_Positions):
        # Get the element number of the i'th point
        element_num : int   = element_nums[1][i];
        point               = mfem.IntegrationPoint();
        x,y                 = numpy.float64(Positions[:, i].tolist());
        point.Set2(x, y);
        u_Positions_0[0, i]= u_gf.GetValue(element_num, point, dim);

    u_list.append( u_Positions_0);

    # Initialize the ODE solver.
    ode_solver.Init(adv);

    # Run the time stepping loop.
    t   : float = 0.0;
    ti  : int   = 0;
    while True:
        # Check if we should stop time stepping (if this time step is within dt/2 of t_final.
        if t > t_final - dt/2:
            break;
        
        # Step the ODE solver.
        t, dt = ode_solver.Step(U, t, dt);
        u_gf.Assign(U);

        # Should we serialize?
        if ti % serialization_steps == 0:
            if(myid == 0): LOGGER.info("time step: " + str(ti) + ", time: " + str(numpy.round(t, 3)));

            # Serialize the current displacement, velocity, and time.
            times_list.append(t);

            # Evaluate the solution at the positions.
            element_nums        = mesh.FindPoints(Positions.T);
            u_Positions_t       = numpy.zeros((1, Num_Positions));

            for i in range(Num_Positions):
                # Get the element number of the i'th point
                element_num : int   = element_nums[1][i];
                point               = mfem.IntegrationPoint();
                x,y                 = numpy.float64(Positions[:, i].tolist());
                point.Set2(x, y)
                u_Positions_t[0, i] = u_gf.GetValue(element_num, point, dim)

            # Append the current solution to the list.
            u_list.append(u_Positions_t);

            # If visualizing, Save the solution to the VisIt object.
            if(VisIt):
                # Save the mesh, solution, and time.
                dc.SetCycle(ti);
                dc.SetTime(t);
                dc.Save();
            
        # Increment the time step counter.
        ti = ti + 1;


    # ---------------------------------------------------------------------------------------------
    # 7. Package everything up for returning.

    # Save the final solution.
    u_gf.Assign(U);
    sol_name = "ex9-final." + smyid;
    u_gf.Save(sol_name, 8);

    # Turn times, displacements, velocities lists into arrays.
    Times       : numpy.ndarray = numpy.array(times_list,           dtype = numpy.float32);
    Trajectory  : numpy.ndarray = numpy.array(u_list,               dtype = numpy.float32);


    return Trajectory, Positions, Times;


if __name__ == "__main__":
    Logging.Initialize_Logger(level = logging.DEBUG);
    Sol, X, T = Simulate();