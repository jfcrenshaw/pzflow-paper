import paths

rule train_two_moons_flow:
    output:
        paths.data / "two_moons_flow.pzflow.pkl"
    cache:
        True
    script:
        paths.scripts / "forward_model" / "train_two_moons_flow.py"

rule train_main_galaxy_flow:
    output:
        directory(paths.data / "main_galaxy_flow")
    cache:
        True
    script:
        paths.scripts / "forward_model" / "train_main_galaxy_flow.py"

rule train_conditional_galaxy_flow:
    output:
        directory(paths.data / "conditional_galaxy_flow")
    cache:
        True
    script:
        paths.scripts / "forward_model" / "train_conditional_galaxy_flow.py"

rule simulate_pzflow_catalog:
    output:
        paths.data / "pzflow_catalog.pkl"
    cache:
        True
    script:
        paths.scripts / "forward_model" / "simulate_pzflow_catalog.py"
