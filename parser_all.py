import argparse

def parse(raw_args):
    """Parse inputs for no_cars (number of cars),
                        no_routes (number of routes)

    Args:
        raw_args (list): list of strings describing the inputs e.g. ["-N 3", "-R 3"]

    Returns:
        dict: A dictionary of the inputs as values, and the name of variables as the keys
    """
    optimizer_choices = ["ADAM", "CG", "COBYLA", "L_BFGS_B",\
                        "NELDER_MEAD", "NFT", "POWELL",\
                         "SLSQP", "SPSA", "TNC", "P_BFGS", "BOBYQA"]
    parser = argparse.ArgumentParser()
    required_named = parser.add_argument_group('Required arguments')
    required_named.add_argument("--no_cars", "-N",
                                required = True,
                                help="Set the number of cars",
                                type = int
                                )
    required_named.add_argument("--no_routes", "-R",
                                required = True,
                                help="Set the number of routes for each car",
                                type = int
                                )
    required_named.add_argument("--penalty_multiplier", "-P",
                                required = True,
                                help="Set the penalty multiplier for QUBO constraint",
                                type = float
                                )
    required_named.add_argument("--optimizer", "-O",
                                required = True,
                                help = "Choose from Qiskit's optimizers",
                                type = str,
                                choices=optimizer_choices
                                )
    required_named.add_argument("--p_max", "-M",
                                required = True,
                                help = "Set maximum number of layers for QAOA",
                                type = int
                                )
    required_named.add_argument("--no_restarts", "-T", required = True,
                                help = "Set number of restarts for QAOA",
                                type = int
                                )
    required_named.add_argument("--no_samples", "-S", required = True,
                                help = "Set number of samples/qubos with given no_cars no_routes",
                                type=int
                                )
    parser.add_argument("--visual",
                        "-V", default = False,
                        help="Activate routes visualisation with '-V' ",
                        action="store_true"
                        )
    args = parser.parse_args(raw_args)
    args = vars(args)
    return args