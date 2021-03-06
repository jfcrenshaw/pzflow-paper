rule train_two_moons_flow:
    output:
        "src/data/two_moons_flow.pzflow.pkl"
    cache:
        True
    script:
        "src/scripts/intro/train_two_moons_flow.py"

rule train_main_galaxy_flow:
    input:
        "src/data/cosmoDC2_subset.pkl"
    output:
        directory("src/data/main_galaxy_flow")
    cache:
        True
    script:
        "src/scripts/forward_model/train_main_galaxy_flow.py"

rule train_conditional_galaxy_flow:
    input:
        "src/data/cosmoDC2_subset.pkl"
    output:
        directory("src/data/conditional_galaxy_flow")
    cache:
        True
    script:
        "src/scripts/forward_model/train_conditional_galaxy_flow.py"

rule simulate_pzflow_catalog:
    input:
        directory("src/data/main_galaxy_flow"),
        directory("src/data/conditional_galaxy_flow")
    output:
        "src/data/pzflow_catalog.pkl"
    cache:
        True
    script:
        "src/scripts/forward_model/simulate_pzflow_catalog.py"
