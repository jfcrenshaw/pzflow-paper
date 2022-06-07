"""Train the flow for the two moons example."""
from pzflow import Flow
from pzflow.examples import get_twomoons_data
from showyourwork.paths import user as Paths

# instantiate the paths
paths = Paths()

# load the two moons example data
data = get_twomoons_data()

# create and train a flow to model the two moons data
flow = Flow(data.columns)
_ = flow.train(data)

# save the trained flow
flow.save(paths.data / "two_moons_flow.pzflow.pkl")
