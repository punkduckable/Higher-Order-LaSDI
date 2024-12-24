Dependencies (version number)
- Python        (3.10)j
- numpy         (1.26.4)
- pytorch       (2.5.1)
- scikit-learn  (1.5.2)
- pyyaml        (6.0.2)
- jupyter       (1.0.0)
- scipy         (1.14.1)
- matplotlib    (3.9.2)
- seaborn       (0.13.2)

What things to I need to modify if I want to use my own [object]?
- [LatentDynamics]
    - Just the Latent Dynamics class. Make a new file in LatentDynamics to define it. Subclass the 
    LatentDynamics class. 
    - Make sure to import your new class in Initialize and add the new class to the ld dictionary. 
    - You need to implement calibrate and simulate for your new class. Make sure the function 
    signatures match those of the base class.
- [Model]
    - Model subclass in Model.py. It must have an Encode and Decode method. You also need to write
    the latent_initial_conditions method. 
    - get_FOM_max_std in Simulate.py. You need to specify how to get an STD from your model. 
    - train method in GPLaSDI. You need to specify how to train your model.
    - Initialize_Model in Initiaize.py. You need to specify how to set up your mode. Chances are 
    the method for autoencoders will work... just double check.
    - model_dict and model_load_dict in Initialize.py.