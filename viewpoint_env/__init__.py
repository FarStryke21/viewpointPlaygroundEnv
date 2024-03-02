from gymnasium.envs.registration import register

register(
    id="CoverageEnv-v0",
    entry_point="viewpoint_env.viewpointWorld:CoverageEnv",
)