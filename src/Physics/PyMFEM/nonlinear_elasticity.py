# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import      os;
import      sys;
import      logging;
from        os.path                 import  expanduser, join, dirname;

from        mfem.common.arg_parser  import  ArgParser;
import      mfem.ser                as      mfem;
from        mfem.ser                import  intArray, add_vector, Add;
import      numpy;
from        numpy                   import  sqrt, pi, cos, sin, hypot, arctan2;
from        scipy.special           import  erfc;

utils_path : str        = os.path.join(os.path.join(os.path.pardir, os.path.pardir), "Utilities");
sys.path.append(utils_path);
import      Logging;

# Setup logger
LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Classes
# -------------------------------------------------------------------------------------------------



class InitialVelocity(mfem.VectorPyCoefficient):
    def EvalValue(self, x):
        dim = len(x)

        global s;

        v = numpy.zeros(len(x))
        v[-1] = s*x[0]**2*(8.0-x[0])
        v[0] = -s*x[0]**2
        return v



class InitialDeformation(mfem.VectorPyCoefficient):
    def EvalValue(self, x):
        return x.copy()



class ElasticEnergyCoefficient(mfem.PyCoefficient):
    def __init__(self, model, x):
        self.x = x
        self.model = model
        self.J = mfem.DenseMatrix()
        mfem.PyCoefficient.__init__(self)

    def Eval(self, T, ip):
        self.model.SetTransformation(T)
        self.x.GetVectorGradient(T, self.J)
        # T.Jacobian().Print()
        # print self.x.GetDataArray()
        # self.J.Print()
        return self.model.EvalW(self.J)/(self.J.Det())



class ReducedSystemOperator(mfem.PyOperator):
    def __init__(self, M, S, H):
        mfem.PyOperator.__init__(self, M.Height())
        self.M = M
        self.S = S
        self.H = H
        self.Jacobian = None
        h = M.Height()
        self.w = mfem.Vector(h)
        self.z = mfem.Vector(h)
        self.dt = 0.0
        self.v = None
        self.x = None

    def SetParameters(self, dt, v, x):
        self.dt = dt
        self.v = v
        self.x = x

    def Mult(self, k, y):
        add_vector(self.v, self.dt, k, self.w)
        add_vector(self.x, self.dt, self.w, self.z)
        self.H.Mult(self.z, y)
        self.M.AddMult(k, y)
        self.S.AddMult(self.w, y)

    def GetGradient(self, k):
        Jacobian = Add(1.0, self.M.SpMat(), self.dt, self.S.SpMat())
        self.Jacobian = Jacobian
        add_vector(self.v, self.dt, k, self.w)
        add_vector(self.x, self.dt, self.w, self.z)
        grad_H = self.H.GetGradientMatrix(self.z)

        Jacobian.Add(self.dt**2, grad_H)
        return Jacobian



class HyperelasticOperator(mfem.PyTimeDependentOperator):
    def __init__(self, fespace, ess_bdr, visc, mu, K):
        mfem.PyTimeDependentOperator.__init__(self, 2*fespace.GetVSize(), 0.0)

        rel_tol = 1e-8
        skip_zero_entries = 0
        ref_density = 1.0
        self.z = mfem.Vector(self.Height()//2)
        self.fespace = fespace
        self.viscosity = visc

        M = mfem.BilinearForm(fespace)
        S = mfem.BilinearForm(fespace)
        H = mfem.NonlinearForm(fespace)
        self.M = M
        self.H = H
        self.S = S

        rho = mfem.ConstantCoefficient(ref_density)
        M.AddDomainIntegrator(mfem.VectorMassIntegrator(rho))
        M.Assemble(skip_zero_entries)
        M.EliminateEssentialBC(ess_bdr)
        M.Finalize(skip_zero_entries)

        M_solver = mfem.CGSolver()
        M_prec = mfem.DSmoother()
        M_solver.iterative_mode = False
        M_solver.SetRelTol(rel_tol)
        M_solver.SetAbsTol(0.0)
        M_solver.SetMaxIter(30)
        M_solver.SetPrintLevel(0)
        M_solver.SetPreconditioner(M_prec)
        M_solver.SetOperator(M.SpMat())

        self.M_solver = M_solver
        self.M_prec = M_prec

        model = mfem.NeoHookeanModel(mu, K)
        H.AddDomainIntegrator(mfem.HyperelasticNLFIntegrator(model))
        H.SetEssentialBC(ess_bdr)
        self.model = model

        visc_coeff = mfem.ConstantCoefficient(visc)
        S.AddDomainIntegrator(mfem.VectorDiffusionIntegrator(visc_coeff))
        S.Assemble(skip_zero_entries)
        S.EliminateEssentialBC(ess_bdr)
        S.Finalize(skip_zero_entries)

        self.reduced_oper = ReducedSystemOperator(M, S, H)

        J_prec = mfem.DSmoother(1)
        J_minres = mfem.MINRESSolver()
        J_minres.SetRelTol(rel_tol)
        J_minres.SetAbsTol(0.0)
        J_minres.SetMaxIter(300)
        J_minres.SetPrintLevel(-1)
        J_minres.SetPreconditioner(J_prec)

        self.J_solver = J_minres
        self.J_prec = J_prec

        newton_solver = mfem.NewtonSolver()
        newton_solver.iterative_mode = False
        newton_solver.SetSolver(self.J_solver)
        newton_solver.SetOperator(self.reduced_oper)
        newton_solver.SetPrintLevel(1)  # print Newton iterations
        newton_solver.SetRelTol(rel_tol)
        newton_solver.SetAbsTol(0.0)
        newton_solver.SetMaxIter(10)
        self.newton_solver = newton_solver

    def Mult(self, vx, dvx_dt):
        sc = self.Height()//2
        v = mfem.Vector(vx, 0,  sc)
        x = mfem.Vector(vx, sc,  sc)
        z = self.z;
        dv_dt = mfem.Vector(dvx_dt, 0, sc)
        dx_dt = mfem.Vector(dvx_dt, sc,  sc)
        self.H.Mult(x, z)
        if (self.viscosity != 0.0):
            self.S.AddMult(v, z)
        z.Neg()
        self.M_solver.Mult(z, dv_dt)
        dx_dt = v
#        Print(vx.Size())

    def ImplicitSolve(self, dt, vx, dvx_dt):
        sc = self.Height()//2
        v = mfem.Vector(vx, 0,  sc)
        x = mfem.Vector(vx, sc,  sc)
        dv_dt = mfem.Vector(dvx_dt, 0, sc)
        dx_dt = mfem.Vector(dvx_dt, sc,  sc)

        # By eliminating kx from the coupled system:
        # kv = -M^{-1}*[H(x + dt*kx) + S*(v + dt*kv)]
        # kx = v + dt*kv
        # we reduce it to a nonlinear equation for kv, represented by the
        # backward_euler_oper. This equation is solved with the newton_solver
        # object (using J_solver and J_prec internally).
        self.reduced_oper.SetParameters(dt, v, x)
        zero = mfem.Vector()  # empty vector is interpreted as
        # zero r.h.s. by NewtonSolver
        self.newton_solver.Mult(zero, dv_dt)
        add_vector(v, dt, dv_dt, dx_dt)

    def ElasticEnergy(self, x):
        return self.H.GetEnergy(x)

    def KineticEnergy(self, v):
        return 0.5*self.M.InnerProduct(v, v)

    def GetElasticEnergyDensity(self, x, w):
        w_coeff = ElasticEnergyCoefficient(self.model, x)
        w.ProjectCoefficient(w_coeff)





# -------------------------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------------------------

def Simulate(   meshfile_name   : str           = "beam-quad.mesh", 
                ref_levels      : int           = 2,
                order           : int           = 3,
                ode_solver_type : int           = 3,
                t_final         : float         = 300.0,
                time_step_size  : float         = 3.0,
                viscosity       : float         = 1e-2,
                shear_modulus   : float         = 0.25, 
                bulk_modulus    : float         = 5.0,
                theta           : float         = 0.1/64.,
                serialize_steps : int           = 1, 
                VisIt           : bool          = True) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    This examples solves a time dependent nonlinear elasticity problem of the form 

        (d/dt)v(X, t)   = H(d(X, t)) + S v(X, t), 
        (d/dt)d(X, t)   = v(X, t),
    
    where H is a hyperelastic model and S is a viscosity operator of Laplacian type. We also impose 
    with the following initial conditions:
        
        d((x, y), 0)         =  (x, y)
        v((x, y), 0)         =  (-theta*x^2, theta*x^2 (8.0 - x))
    
    where X[0] and X[-1] are the positions of the first and lash nodes, respectively. Here, theta 
    is a parameter that the user can change. 
    
    See the c++ version of example 10 in the MFEM library for more detail.

    We solve this PDE, then return the solution at each time step. 

        

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    meshfile_name : str
        specifies the mesh file to use. This should specify a file in the Physics/PyMFEM/data 
        subdirectory.

    ref_levels : int   
        specifies the number of times to refine the mesh uniformly.

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

    theta : float
        specifies the constant "theta" in the initial velocity.

    serialize_steps : int
        Specifies how frequently we serialize (save) the solution.

    VisIt : bool
        If True, will prompt the code to save the displacement and velocity GridFunctions every 
        time we serialize them. It will save the GridFunctions in a format that VisIt 
        (visit.llnl.gov) can understand/work with.
    
        
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    D, V, X, T. 
    
    D : numpy.ndarray, shape = (Nt, 2, N_Nodes)
        i, j, k element holds the j'th component of the displacement at the k'th position (i.e., 
        X[i, :]) at the i'th time step (i.e., T[i]).
    
    V : numpy.ndarray, shape = (Nt, 2, N_Nodes)
        i, j, k element holds the j'th component of the velocity at the k'th position (i.e., X[i]) 
        at the i'th time step (i.e., T[i]).

    X : numpy.ndarray, shape = (N_Nodes, 2)
        i'th row holds the position of the i'th node at which we evaluate the solution.
    
    T : numpy.ndarray, shape = (Nt)
        i'th element holds the j'th time at which we evaluate the solution.
    """
    


    # ---------------------------------------------------------------------------------------------
    # 1. Setup 

    LOGGER.info("Setting up non-linear elasticity simulation with MFEM.");

    # Set variable aliases.
    dt      : float = time_step_size;
    visc    : float = viscosity;
    mu      : float = shear_modulus;
    K       : float = bulk_modulus;
    global s;
    s = theta;
    LOGGER.info("Simulating with theta = %f" % theta);

    # Setup the mesh.
    LOGGER.debug("Lading the mesh and its properties");
    meshfile_path   : str   = expanduser(join(dirname(__file__), 'data', meshfile_name));
    mesh                    = mfem.Mesh(meshfile_path, 1, 1);
    dim             : int   = mesh.Dimension();
    LOGGER.debug("meshfile_path = %s" % meshfile_path);
    LOGGER.debug("dim = %d" % dim);

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

    # Refine the mesh 
    LOGGER.debug("Refining mesh");
    for lev in range(ref_levels):
        mesh.UniformRefinement();
    

    # ---------------------------------------------------------------------------------------------
    # 2. Define the vector finite element spaces representing the mesh
    #    deformation x, the velocity v, and the initial configuration, x_ref.
    #    Define also the elastic energy density, w, which is in a discontinuous
    #    higher-order space. Since x and v are integrated in time as a system,
    #    we group them together in block vector vx, with offsets given by the
    #    fe_offset array.

    LOGGER.info("Setting up the FEM space.");
    fec                 = mfem.H1_FECollection(order, dim);         # Basis functions
    fespace             = mfem.FiniteElementSpace(mesh, fec, dim);  # FEM space (span of basis functions).

    
    fe_size     : int   = fespace.GetVSize();
    LOGGER.info("Number of velocity/deformation unknowns: " + str(fe_size));
   
    fe_offset           = intArray([0, fe_size, 2*fe_size]);

    # Setup the grid functions for displacement and velocity.
    VD      = mfem.BlockVector(fe_offset);
    D_gf    = mfem.GridFunction();
    V_gf    = mfem.GridFunction();
    V_gf.MakeRef(fespace, VD.GetBlock(0), 0);
    D_gf.MakeRef(fespace, VD.GetBlock(1), 0);
    
    # ???
    D_ref = mfem.GridFunction(fespace);
    mesh.GetNodes(D_ref);
    
    # Elastic energy density.
    w_fec       = mfem.L2_FECollection(order + 1, dim);
    w_fespace   = mfem.FiniteElementSpace(mesh, w_fec);
    w           = mfem.GridFunction(w_fespace);



    # ---------------------------------------------------------------------------------------------
    # 3. Set the initial conditions for v and x, and the boundary conditions on
    #    a beam-like mesh (see description above).

    LOGGER.info("Settng initial and boundary conditions");

    # Set up objects to hold the ICs
    LOGGER.debug("Setting up objects to hold the initial conditions;");
    velo        = InitialVelocity(dim);
    deform      = InitialDeformation(dim);
    V_gf.ProjectCoefficient(velo);
    D_gf.ProjectCoefficient(deform);

    # Impose boundary conditions.
    LOGGER.debug("Imposing Boundary Conditions");
    ess_bdr = intArray(fespace.GetMesh().bdr_attributes.Max());
    ess_bdr.Assign(0);
    ess_bdr[0] = 1;



    # ---------------------------------------------------------------------------------------------
    # 4. Define HyperelasticOperator and initialize it the initial energies.
    
    LOGGER.info("Setting up Hyperelastic operator.");

    oper = HyperelasticOperator(fespace, ess_bdr, visc, mu, K);
    ee0 = oper.ElasticEnergy(D_gf);
    ke0 = oper.KineticEnergy(V_gf);

    LOGGER.info("initial elastic energy (EE) = " + str(ee0));
    LOGGER.info("initial kinetic energy (KE) = " + str(ke0));
    LOGGER.info("initial   total energy (TE) = " + str(ee0 + ke0));



    # ---------------------------------------------------------------------------------------------
    # 5. Extract the positions of the nodes.

    LOGGER.info("Extracting node positions");

    # Fetch the nodes + number of them
    Nodes_GridFun   : mfem.GridFunction = mfem.GridFunction(fespace);
    mesh.GetNodes(Nodes_GridFun);                                               # Get GridFunction that holds the nodes
    Num_Nodes       : int               = Nodes_GridFun.FESpace().GetNDofs();   # Get the number of nodes
    LOGGER.debug("There are %d nodes" % Num_Nodes);

    # Now extra the data stored at the nodes. This will look like the a list holding the first
    # coordinate of each node concatenated with a list holding the second coordinate of every node
    # and so on. For example, if dim = 2, this is the array (x1, ... , xN, y1, ... , yN).
    nodes_data      : numpy.ndarray     = Nodes_GridFun.GetDataArray();         
    
    # Reshape to be an array whose i'th row holds the position of the i'th node
    Positions       : numpy.ndarray     = numpy.reshape(nodes_data, (dim, Num_Nodes)).T; 
    LOGGER.debug("Positions has shape %s (Num_Nodes = %d, dim = %d)" % (str(Positions.shape), Num_Nodes, dim));



    # ---------------------------------------------------------------------------------------------
    # 7. VisIt

    # Setup VisIt visualization (if we are doing that)
    if (VisIt):
        LOGGER.info("Setting up VisIt visualization.");

        dc_path : str   = os.path.join(os.path.join(os.path.curdir, "VisIt"), "nlelast-fom");
        dc              = mfem.VisItDataCollection(dc_path, mesh);
        dc.SetPrecision(8);
        # // To save the mesh using MFEM's parallel mesh format:
        # // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
        dc.RegisterField("Disp",    D_gf);
        dc.RegisterField("Vel",     V_gf);
        dc.SetCycle(0);
        dc.SetTime(0.0);
        dc.Save();



    # ---------------------------------------------------------------------------------------------
    # 6. Perform time-integration (looping over the time iterations, ti, with a time-step dt).
    
    LOGGER.info("Running time stepping from t = 0 to t = %f with dt = %d" % (t_final, dt));

    # Setup for time stepping.
    ode_solver.Init(oper);
    times_list          : list[float]           = [];    
    displacements_list  : list[numpy.ndarray]   = [];
    velocities_list     : list[numpy.ndarray]   = [];

    # Append the ICs.
    times_list.append(0);
    displacements_list.append(  numpy.reshape(D_gf.GetDataArray().copy(), (dim, Num_Nodes)));
    velocities_list.append(     numpy.reshape(V_gf.GetDataArray().copy(), (dim, Num_Nodes)));

    # Time step!!!!!
    t           : float = 0.0;
    ti          : int   = 1;        # counter to keep track of when we should serialize solution.
    last_step   : bool  = False;
    while not last_step:
        # Check if we should stop time stepping (if this time step is within dt/2 of t_final.
        if (t + dt >= t_final - dt/2):
            last_step = True;

        t, dt = ode_solver.Step(VD, t, dt)

        # Should we serialize?
        if (last_step or (ti % serialize_steps) == 0):
            # Find energy.
            ee = oper.ElasticEnergy(D_gf);
            ke = oper.KineticEnergy(V_gf);

            text : str  = ( "step " + str(ti) + ", t = " + str(t) + ", EE = " +
                            str(ee) + ", KE = " + str(ke) +
                            ", dTE = " + str((ee + ke) - (ee0 + ke0)));
            LOGGER.info(text);


            # Serialize the current displacement, velocity, and time.
            times_list.append(t);
            displacements_list.append(  numpy.reshape(D_gf.GetDataArray().copy(),  (dim, Num_Nodes)));
            velocities_list.append(     numpy.reshape(V_gf.GetDataArray().copy(),  (dim, Num_Nodes)));


            # If visualizing, Save the GridFunctions to the VisIt object.
            if(VisIt):
                # Set the mesh to the current displacement
                mesh.SwapNodes(D_gf, 0);

                # Save the mesh, displacement, and velocity
                dc.SetCycle(ti);
                dc.SetTime(t);
                dc.Save();
        
                # Now swap the deformed mesh back to reset everything.
                mesh.SwapNodes(D_gf, 0);

        ti = ti + 1;
        



    # ---------------------------------------------------------------------------------------------
    # 7. Package everything up for returning.

    # Turn times, displacements, velocities lists into arrays.
    Times           = numpy.array(times_list, dtype = numpy.float32);
    Displacements   = numpy.array(displacements_list, dtype = numpy.float32);
    Velocities      = numpy.array(velocities_list, dtype = numpy.float32);


    return Displacements, Velocities, Positions, Times;

    nodes = x
    owns_nodes = 0
    nodes, owns_nodes = mesh.SwapNodes(nodes, owns_nodes)
    mesh.Print('deformed.mesh', 8)
    mesh.SwapNodes(nodes, owns_nodes)
    v.Save('velocity.sol', 8)
    oper.GetElasticEnergyDensity(x, w)
    w.Save('elastic_energy.sol',  8)



if __name__ == "__main__":
    Logging.Initialize_Logger(level = logging.INFO);
    D, V, X, T = Simulate();
