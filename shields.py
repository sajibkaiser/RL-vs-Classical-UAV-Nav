"""
Safety shields: they sit between the pilot and the drone,
vetoing actions that look dangerous.
"""
import numpy as np

ACTIONS = [0, 1, 2, 3, 4]   # hover, up, down, left, right


def _forces(action, thrust):
    """Turn an action number into a push (same mapping as drone_env.py)."""
    if action == 1: return 0.0, -thrust
    if action == 2: return 0.0,  thrust
    if action == 3: return -thrust, 0.0
    if action == 4: return  thrust, 0.0
    return 0.0, 0.0


def _is_unsafe(x, y, env):
    """Would the drone be crashed at this position?"""
    if x < 0 or x > 8.0 or y < 0 or y > 6.0:
        return True
    for wall_x, gap_y in [(env.wall1_x, env.wall1_gap_y),
                          (env.wall2_x, env.wall2_gap_y)]:
        if wall_x <= x <= wall_x + env.wall_width:
            if not (gap_y <= y <= gap_y + env.gap_height):
                return True
    return False


def _predict(env, obs, action, n_steps, brake_after_first=False, worst_case_wind=0.0):
    """
    Simulates forward from the state in obs (obs[0]=x, obs[1]=y, obs[2]=vx,
    obs[3]=vy) - the pilot's (possibly noisy) reading of its own position and
    velocity - not from env's true state. Fixed parameters (mass, thrust,
    drag, time_step, walls) still come from env, since those describe the
    world rather than a noisy sensor reading.

    Also imagines the worst possible gust at every step (in whichever
    direction is currently more dangerous), so the shield isn't judging
    safety as if the air were perfectly calm.
    """
    for gust_sign in (-1.0, 1.0):
        x, y = obs[0], obs[1]
        vx, vy = obs[2], obs[3]
        dt = env.time_step
        safe = True

        for step in range(n_steps):
            if brake_after_first and step > 0:
                fx, fy = -np.sign(vx) * env.max_lateral_thrust, 0.0
                if abs(vy) > abs(vx):
                    fx, fy = 0.0, -np.sign(vy) * env.max_lateral_thrust
            else:
                fx, fy = _forces(action, env.max_lateral_thrust)

            fy += gust_sign * worst_case_wind   # assume the worst gust, not calm air

            ax = (fx - env.linear_drag * vx) / env.mass
            ay = (fy - env.linear_drag * vy) / env.mass
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt

            if _is_unsafe(x, y, env):
                safe = False
                break

        if not safe:
            return False   # unsafe in at least one worst-case direction
    return True

def naive_shield(env, obs, proposed_action):
    """
    THE STANDARD APPROACH: check only ONE step ahead.
    If the proposed action looks fine one step from now, allow it.

    Predicts forward from obs's position/velocity reading (which may be
    noisy), not the environment's true state.
    """
    if _predict(env, obs, proposed_action, n_steps=1):
        return proposed_action, False          # allowed, no intervention

    # Not safe - try the other actions, pick the first safe one
    for alt in ACTIONS:
        if alt != proposed_action and _predict(env, obs, alt, n_steps=1):
            return alt, True                   # overridden
    return 0, True                             # nothing safe - hover


def lookahead_shield(env, obs, proposed_action, horizon=5):
    """
    Same idea as naive_shield, but checks several steps ahead instead of
    just one. Does not assume the drone brakes, and does not model wind
    (n_steps=horizon with brake_after_first and worst_case_wind left at
    their defaults).

    Predicts forward from obs's position/velocity reading, same as naive_shield.
    """
    if _predict(env, obs, proposed_action, n_steps=horizon):
        return proposed_action, False          # allowed, no intervention

    # Not safe - try the other actions, pick the first safe one
    for alt in ACTIONS:
        if alt != proposed_action and _predict(env, obs, alt, n_steps=horizon):
            return alt, True                   # overridden
    return 0, True                             # nothing safe - hover


def momentum_aware_shield(env, obs, proposed_action, horizon=3, wind_margin=0.6):
    """
    wind_margin: how much of the max gust to assume. Full strength (1.0) is
    unrealistically pessimistic because real gusts flip direction and cancel out.

    Predicts forward from obs's position/velocity reading, same as naive_shield.
    """
    assumed_wind = env.wind_strength * wind_margin

    if _predict(env, obs, proposed_action, n_steps=horizon, brake_after_first=True,
                worst_case_wind=assumed_wind):
        return proposed_action, False

    # Try alternatives
    for alt in ACTIONS:
        if alt != proposed_action and _predict(env, obs, alt, n_steps=horizon,
                                               brake_after_first=True,
                                               worst_case_wind=assumed_wind):
            return alt, True

    # NOTHING is safe - don't hover (that's helpless). Keep the pilot's choice,
    # since the pilot at least aims somewhere sensible.
    return proposed_action, False

def no_shield(env, obs, proposed_action):
    """No shield at all - the pilot does whatever it wants."""
    return proposed_action, False