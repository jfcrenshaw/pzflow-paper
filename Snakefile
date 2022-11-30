rule train_two_moons_flow:
    output:
        "src/data/two_moons_flow.pzflow.pkl",
    cache: True
    script:
        "src/scripts/intro/train_two_moons_flow.py"


rule train_main_galaxy_flow:
    input:
        "src/data/cosmoDC2_subset.pkl",
    output:
        directory("src/data/main_galaxy_flow"),
    cache: True
    script:
        "src/scripts/forward_model/train_main_galaxy_flow.py"


rule train_conditional_galaxy_flow:
    input:
        "src/data/cosmoDC2_subset.pkl",
    output:
        directory("src/data/conditional_galaxy_flow"),
    cache: True
    script:
        "src/scripts/forward_model/train_conditional_galaxy_flow.py"


rule simulate_pzflow_catalog:
    input:
        directory("src/data/main_galaxy_flow"),
        directory("src/data/conditional_galaxy_flow"),
    output:
        "src/data/pzflow_catalog.pkl",
    cache: True
    script:
        "src/scripts/forward_model/simulate_pzflow_catalog.py"


rule train_pz_ensemble:
    input:
        "src/data/pzflow_catalog.pkl",
    output:
        directory("src/data/pz_ensemble/"),
    cache: True
    script:
        "src/scripts/photo-z/train_pz_ensemble.py"


rule calculate_posteriors:
    input:
        "src/data/pzflow_catalog.pkl",
        "src/data/pz_ensemble/",
    output:
        "src/data/redshift_posteriors.npz",
    cache: True
    script:
        "src/scripts/photo-z/calculate_posteriors.py"


rule calculate_posteriors_without_u:
    input:
        "src/data/pzflow_catalog.pkl",
        "src/data/pz_ensemble/",
    output:
        "src/data/redshift_posteriors_without_u.npz",
    cache: True
    script:
        "src/scripts/photo-z/calculate_posteriors_without_u.py"
