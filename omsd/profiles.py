"""Main experiment defaults plus the OMIGA research branch's profile."""

PROFILES = {
    "mpe": dict(env_id="simple_spread", backend="mpe", dataset_format="numpy", episode_length=25,
                terminal_policy="fixed_horizon", critic_input="obs"),
    "mamujoco210": dict(env_id="2ant", backend="mamujoco210", dataset_format="ogmarl",
                        episode_length=1000, terminal_policy="provided", critic_input="obs"),
    "omiga": dict(env_id="HalfCheetah-v2", backend="omiga", dataset_format="omiga", episode_length=1000,
                  terminal_policy="provided", critic_input="obs"),
}
