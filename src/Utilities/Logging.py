import logging;



# -------------------------------------------------------------------------------------------------
# Initialize logger
# -------------------------------------------------------------------------------------------------

def Initialize_Logger(level):

    # Initialize the logger, set the level.
    logger = logging.getLogger();
    logger.setLevel(level);

    # Set up a handler to pass logged info to the console
    sh = logging.StreamHandler();
    sh.setLevel(level);
    
    # Setup a formatter for the handler. 
    LOG_FMT : str = '%(asctime)s - %(name)s:%(funcName)s:%(lineno)s - %(levelname)s - %(message)s';
    sh.setFormatter(logging.Formatter(LOG_FMT));
    logger.addHandler(sh);



# -------------------------------------------------------------------------------------------------
# Helper functions. 
# -------------------------------------------------------------------------------------------------

def Log_Dictionary(LOGGER : logging.Logger, D : dict, level : int = logging.DEBUG, indent : int = 0) -> None:
    indent_str : str = '   '*indent;

    # Determine which level we're using to report the dictionary. Can either be debug or info.
    if(  level == logging.DEBUG):
        Report = LOGGER.debug;
    elif(level == logging.INFO):
        Report = LOGGER.info;
    else:
        LOGGER.warning("Invalid dictionary log level. Valid options are logging.DEBUG = %d and logging.INFO = %d. Got %d" % (logging.DEBUG, logging.INFO, level));
        LOGGER.warning("Returning without reporting dictionary...");
        return; 

    # Report the dictionary
    for k,v in D.items():
        if(isinstance(v, dict)):
            Report("%s[%s] ==>" % (indent_str, str(k)));
            Log_Dictionary(LOGGER = LOGGER, D = v, level = level, indent = indent + 1);   
        else:
            Report("%s[%s] ==> [%s]" % (indent_str, str(k), str(v)));



def Print_Dictionary(_ : dict, indent : int = 0) -> None:
    assert isinstance(_, dict);
    assert isinstance(indent, int);

    istr : str = '   '.join(['' for _ in range(indent)]);
    for k,v in _.items():
        if isinstance(v, dict):
            print(f'{istr}[{k}] ==>');
            Print_Dictionary(v, indent + 2);
        else:
            print(f'{istr}[{k}] ==> [{v}]');

