"""Dataset/environment defaults for the main MPE and OG-MARL experiments."""

PROFILES = {
    "mpe": dict(env_id="simple_spread", backend="mpe", dataset_format="numpy", episode_length=25,
                terminal_policy="fixed_horizon", critic_input="obs"),
    "mamujoco210": dict(env_id="2ant", backend="mamujoco210", dataset_format="ogmarl",
                        episode_length=1000, terminal_policy="provided", critic_input="obs"),
}
