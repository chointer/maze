from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pygame

from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.utils.play import MissingKeysToAction, PlayableGame, display_arr

def play_discrete(
    env: Env,
    transpose: Optional[bool] = True,
    fps: Optional[int] = None,
    zoom: Optional[float] = None,
    callback: Optional[Callable] = None,
    keys_to_action: Optional[Dict[Union[Tuple[Union[str, int]], str], ActType]] = None,
    seed: Optional[int] = None,
    noop: ActType = 0,
):
    """Allows one to play the game using keyboard. 
    This function is a modified version of the play function provided by the gymnasium.utils.play.
    Modifications were made for discrete games; responds only to the last pressed key.

    Args:
        env: Environment to use for playing.
        transpose: If this is ``True``, the output of observation is transposed. Defaults to ``True``.
        fps: Maximum number of steps of the environment executed every second. If ``None`` (the default),
            ``env.metadata["render_fps""]`` (or 30, if the environment does not specify "render_fps") is used.
        zoom: Zoom the observation in, ``zoom`` amount, should be positive float
        callback: If a callback is provided, it will be executed after every step. It takes the following input:
                obs_t: observation before performing action
                obs_tp1: observation after performing action
                action: action that was executed
                rew: reward that was received
                terminated: whether the environment is terminated or not
                truncated: whether the environment is truncated or not
                info: debug info
        keys_to_action:  Mapping from keys pressed to action performed.
            Different formats are supported: Key combinations can either be expressed as a tuple of unicode code
            points of the keys, as a tuple of characters, or as a string where each character of the string represents
            one key.
        seed: Random seed used when resetting the environment. If None, no seed is used.
        noop: The action used when no key input has been entered, or the entered key combination is unknown.
    """

    key_for_action = []
    env.reset(seed=seed)


    if keys_to_action is None:
        if hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        elif hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        else:
            assert env.spec is not None
            raise MissingKeysToAction(
                f"{env.spec.id} does not have explicit key to action mapping, "
                "please specify one manually"
            )
    assert keys_to_action is not None

    key_code_to_action = {}
    for key_combination, action in keys_to_action.items():
        key_code = tuple(
            sorted(ord(key) if isinstance(key, str) else key for key in key_combination)
        )
        key_code_to_action[key_code] = action

    game = PlayableGame(env, key_code_to_action, zoom)

    if fps is None:
        fps = env.metadata.get("render_fps", 30)

    done, obs = True, None
    clock = pygame.time.Clock()

    while game.running:
        if done:
            done = False
            obs = env.reset(seed=seed)
        else:
            action = key_code_to_action.get(tuple(sorted(key_for_action)), noop)        # modified; from game.pressed_keys to key_for_aciton
            key_for_action = []                                                         # reset

            prev_obs = obs
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if callback is not None:
                callback(prev_obs, obs, action, rew, terminated, truncated, info)
        if obs is not None:
            rendered = env.render()
            if isinstance(rendered, List):
                rendered = rendered[-1]
            assert rendered is not None and isinstance(rendered, np.ndarray)
            display_arr(
                game.screen, rendered, transpose=transpose, video_size=game.video_size
            )

        # process pygame events
        for event in pygame.event.get():
            game.process_event(event)
            if event.type == pygame.KEYDOWN and event.key in game.relevant_keys:
                key_for_action.append(event.key)
            
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()