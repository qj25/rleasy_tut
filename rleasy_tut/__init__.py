from gym.envs.registration import (
    register,
)

register(
    id="PandaInsert-v0",
    entry_point="rleasy_tut.envs:PandaInsertEnv",
    max_episode_steps=1000,
)