import argparse

def parse():
    """Parse inputs for no_cars (number of cars),
                        no_routes (number of routes)

    Args:
        raw_args (list): list of strings describing the inputs e.g. ["-N 3", "-R 3"]

    Returns:
        dict: A dictionary of the inputs as values, and the name of variables as the keys
    """
    # optimizer_choices = ["ADAM", "CG", "COBYLA", "L_BFGS_B",\
    #                     "NELDER_MEAD", "NFT", "POWELL",\
    #                      "SLSQP", "SPSA", "TNC", "P_BFGS", "BOBYQA"]
    parser = argparse.ArgumentParser()
    required_named = parser.add_argument_group('Required arguments')
    required_named.add_argument("--no_cars", "-N",
                                required = False,
                                help="Set the number of cars",
                                type = int
                                )
    required_named.add_argument("--no_routes", "-R",
                                required = False,
                                help="Set the number of routes for each car",
                                type = int
                                )
    required_named.add_argument("--penalty_multiplier", "-P",
                                required = False,
                                help="Set the penalty multiplier for QUBO constraint",
                                type = float
                                )
    required_named.add_argument("--p_max", "-M",
                                required = False,
                                help = "Set maximum number of layers for QAOA",
                                type = int
                                )
    required_named.add_argument("--no_restarts", "-T", required = False,
                                help = "Set number of restarts for QAOA",
                                type = int
                                )
    required_named.add_argument("--no_samples", "-S", required = False,
                                help = "Set number of samples/qubos with given no_cars no_routes",
                                type=int
                                )
    required_named.add_argument("--method", "-O", required=False, help = "Set optimizer method from NLOPT library", type = str)
    parser.add_argument("--fourier", "-F", default = False, help = "Set whether to use FOURIER parametrisation or not", action = "store_true")
    parser.add_argument("--interp", "-INT", default = False, help = "Set whether to use INTERP layer-based point initialization or not", action = "store_true")
    parser.add_argument("--bias", "-B", default = False, help = "Set whether to use biased correlations or not", action = "store_true")
    parser.add_argument("--symmetrise", "-Y", default = False, help = "Set whether to symmetrise Hamiltonian initially or not", action = "store_true")
    parser.add_argument("--customise", "-C", default = False, help = "Set whether to use Custom QAOA circuit or Regular", action = "store_true")
    parser.add_argument("--visual",
                        "-V", default = False,
                        help="Activate routes visualisation with '-V' ",
                        action="store_true"
                        )
    parser.add_argument("--simulator", "-I", required=False, default=None, help = "Set simulation method from Qiskit Aer library", type = str)
    parser.add_argument("--noisy", "-X", required=False, default=False, help = "Set whether simulation is noisy", action="store_true")
    parser.add_argument("--multiplier", "-U", required=False, default=1.0, help = "Set error rate multiplier", type=float)
    args = parser.parse_args()
    args = vars(args)
    return args