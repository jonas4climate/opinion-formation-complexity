"""
PSEUDOCODE TO OBTAIN POWER LAW OUT OF THE CA
"""


# Step 1 - Set system parameters (except temperature)
## Same as with other scripts



# Step 2 - Find the critical temperature of the system
## Empirically (like shown in critical_temperature.py) or
## Analitically, using the approximations referenced in the CA paper



# Step 3 - Set the system to the critical temperature
critical_temperature = 100


# Step 4 - Simulate system at this critical temperature
temperature = critical_temperature
timesteps = 100

possible_cluster_sizes = R # As many elements as possible cluster radius: 1,2... up to R
cluster_sizes = np.zeros(R)

## For all timesteps
for timestep in range(timesteps):
    ### Compute timestep
    ### Update all clusters sizes (not only the one in the center!)
    ### Save that information


# Step 4 - Plot distribution of cluster sizes
# Verify if they follow a power law with the package powerlaw



# Step 5 - Plot distribution of cluster durations
# Verify if they follow a power law with the package powerlaw
