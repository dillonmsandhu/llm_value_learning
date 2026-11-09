# helpers.py
# This file contains technical helpers used for the RL loop, including GAE and trace computation, PPO loss, and environment initialization.
from core.imports import *
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper
from envs.log_wrapper import LogWrapper
from envs.wrappers import (NormalizeObservationWrapper, NormalizeRewardWrapper, 
AddChannelWrapper, ClipAction, NormalizeRewardEnvState, NormalizeObsEnvState, 
TerminalInfoWrapper, MountainCarNormalizeWrapper, MountainCarSparseRewardWrapper)
from envs.boyan_chain import MatrixMockEnv, BoyanParams
from envs.whirlpool import WhirlpoolExactValue
from envs.whirlpool_env import Whirlpool
from gymnax.environments import spaces
from flax.core import unfreeze, freeze

def create_evaluator(config, env=None, env_params=None):
    from envs.fourrooms import (
        FourRoomsExactValue,
        FourRoomsDenseExactValue,
        FourRoomsMinesExactValue,
        FourRoomsMinesDenseExactValue,
    )
    from envs.fourrooms_continuing import ContinuingFourRooms
    from envs.eightrooms import (
        EightRoomsExactValue,
        ContinuingEightRooms,
        EightRoomsDenseExactValue,
        ContinuingEightRoomsDense,
    )
    from envs.boyan_chain import ContinuingBoyanRing
    from envs.whirlpool import WhirlpoolExactValue, ContinuingWhirlpool
    from envs.mountaincar_exact import MountainCarExactValue

    env_name = config['ENV_NAME'].lower()

    if env_name in ['fourrooms', 'fourrooms-misc']:
        return FourRoomsExactValue(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (11, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['fourrooms-dense', 'fourrooms_dense', 'fourroomsdense', 'fourrooms-misc-dense']:
        return FourRoomsDenseExactValue(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (11, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
    elif env_name in ['fourrooms-mines', 'fourrooms_mines', 'fourroom-mines', 'fourrooms-misc-mines']:
        return FourRoomsMinesExactValue(
            start_pos=getattr(env, 'pos_fixed', config.get('START_POS', (3, 1))),
            goal_pos=getattr(env, 'goal_fixed', config.get('GOAL_POS', (11, 11))),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
            mine_locations=config.get('MINE_LOCATIONS', None),
            mine_reward=config.get('MINE_REWARD', 0.0),
        )
    elif env_name in ['fourrooms-mines-dense', 'fourrooms_mines_dense', 'fourroom-mines-dense']:
        return FourRoomsMinesDenseExactValue(
            start_pos=getattr(env, 'pos_fixed', config.get('START_POS', (3, 1))),
            goal_pos=getattr(env, 'goal_fixed', config.get('GOAL_POS', (11, 11))),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
            mine_locations=config.get('MINE_LOCATIONS', None),
            mine_reward=config.get('MINE_REWARD', 0.0),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
    elif env_name in ['continuous-fourrooms', 'continuous_fourrooms', 'continuousfourrooms', 'continuous-fourrooms-misc', 'fourrooms-continuous']:
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        return FourRoomsExactValue(
            start_pos=(3, 1),
            goal_pos=(11, 11),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.0)),
            gamma=config['GAMMA'],
            use_visual_obs=use_visual,
        )
    elif env_name in ['continuous-fourrooms-dense', 'continuous_fourrooms_dense', 'continuousfourroomsdense', 'continuous-fourrooms-misc-dense', 'fourrooms-continuous-dense']:
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        return FourRoomsDenseExactValue(
            start_pos=(3, 1),
            goal_pos=(11, 11),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.0)),
            gamma=config['GAMMA'],
            use_visual_obs=use_visual,
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
    elif env_name == 'fourrooms-cont':
        return ContinuingFourRooms(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (11, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['eightrooms', 'eightrooms-misc']:
        return EightRoomsExactValue(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (23, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['eightrooms-cont']:
        return ContinuingEightRooms(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (23, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['eightrooms-dense', 'eightrooms_dense', 'eightroomsdense', 'eightrooms-misc-dense']:
        return EightRoomsDenseExactValue(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (23, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
    elif env_name in ['eightrooms-dense-cont', 'eightrooms_dense_cont', 'eightroomsdense-cont']:
        return ContinuingEightRoomsDense(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (23, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
    elif env_name in ['continuous-eightrooms', 'continuous_eightrooms', 'continuouseightrooms', 'continuous-eightrooms-misc', 'eightrooms-continuous']:
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        return EightRoomsExactValue(
            start_pos=(3, 1),
            goal_pos=(23, 11),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.0)),
            gamma=config['GAMMA'],
            use_visual_obs=use_visual,
        )
    elif env_name in ['continuous-eightrooms-dense', 'continuous_eightrooms_dense', 'continuouseightroomsdense', 'continuous-eightrooms-misc-dense', 'eightrooms-continuous-dense']:
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        return EightRoomsDenseExactValue(
            start_pos=(3, 1),
            goal_pos=(23, 11),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.0)),
            gamma=config['GAMMA'],
            use_visual_obs=use_visual,
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
    elif env_name == 'boyan':
        return ContinuingBoyanRing(
            size=config.get('ENV_SIZE', 21),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['whirlpool', 'whirlpool-misc']:
        return WhirlpoolExactValue(
            size=config.get('ENV_SIZE', 21),
            gamma=config['GAMMA'],
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.5)),
            start_pos=getattr(env, 'pos_fixed', None),
            goal_pos=getattr(env, 'goal_fixed', None),
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['whirlpool-cont']:
        return ContinuingWhirlpool(
            size=config.get('ENV_SIZE', 20),
            gamma=config['GAMMA'],
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.5)),
            start_pos=getattr(env, 'pos_fixed', None),
            goal_pos=getattr(env, 'goal_fixed', None),
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['spaceinvadersexactvalue', 'spaceinvaders', 'spaceinvaders-exact']:
        from envs.space_invaders import SpaceInvadersExactValue
        return SpaceInvadersExactValue(
            width=config.get('SPACE_INVADERS_WIDTH', 7),
            height=config.get('SPACE_INVADERS_HEIGHT', 6),
            num_aliens=config.get('SPACE_INVADERS_NUM_ALIENS', 3),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
            endless=config.get('SPACE_INVADERS_ENDLESS', True),
        )
    elif env_name in ['mountaincar', 'mountaincar-v0']:
        from envs.mountaincar_exact import MountainCarExactValue
        return MountainCarExactValue(
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', False),
            goal_reward=config.get('GOAL_REWARD', 1.0),
        )
    elif env_name in ['mountaincar-dense', 'mountaincardense']:
        from envs.mountaincar_exact import MountainCarDenseExactValue
        return MountainCarDenseExactValue(
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', False),
            potential_scale=config.get('POTENTIAL_SCALE', 20.0/100.0),
            goal_reward=config.get('GOAL_REWARD', 1.0),
        )
    return None

def initialize_evaluator(config, env, env_params):
    if not config.get("CALC_TRUE_VALUES", False):
        return None
    if hasattr(env, "evaluator") and env.evaluator is not None:
        return env.evaluator
    return create_evaluator(config, env, env_params)

def make_env(config):
    env_name = config['ENV_NAME'].lower()
    use_tabular = config.get("USE_TABULAR_SIMULATOR", True)

    tabular_env_names = [
        'whirlpool', 'whirlpool-misc', 'whirlpool-cont',
        'fourrooms', 'fourrooms-misc', 'fourrooms-cont', 'fourrooms-dense',
        'fourrooms-mines', 'fourrooms_mines', 'fourroom-mines', 'fourrooms-misc-mines',
        'fourrooms-mines-dense', 'fourrooms_mines_dense', 'fourroom-mines-dense',
        'eightrooms', 'eightrooms-misc', 'eightrooms-cont',
        'eightrooms-dense', 'eightrooms_dense', 'eightroomsdense', 'eightrooms-misc-dense',
        'eightrooms-dense-cont', 'eightrooms_dense_cont',
        'boyan',
        'spaceinvadersexactvalue', 'spaceinvaders', 'spaceinvaders-exact',
        'mountaincar', 'mountaincar-v0', 'mountaincar-dense', 'mountaincardense',
    ]

    if use_tabular and env_name in tabular_env_names:
        from envs.tabular_matrix_env import TabularMatrixEnv, TabularParams
        evaluator = create_evaluator(config)
        env = TabularMatrixEnv(evaluator, name=config['ENV_NAME'])
        env_params = TabularParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1e6)),
            fail_prob=getattr(evaluator, 'fail_prob', 0.0),
        )
        env = TerminalInfoWrapper(env)
        if '-cont' in env_name:
            from envs.wrappers import ContinuingWrapper
            env = ContinuingWrapper(env)

    elif env_name in ['mountaincar-v0', 'mountaincar', 'mountaincar-dense']:
        env, env_params = gymnax.make('MountainCar-v0')
        env_params = env_params.replace(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1e6))
        )
        env = TerminalInfoWrapper(env)
        env = MountainCarNormalizeWrapper(env)
        if 'dense' in env_name:
            from envs.wrappers import MountainCarDenseRewardWrapper
            env = MountainCarDenseRewardWrapper(
                env,
                potential_scale=config.get('POTENTIAL_SCALE', 20.0),
                goal_reward=config.get('GOAL_REWARD', 100.0),
                gamma=config['GAMMA'],
            )
        else:
            env = MountainCarSparseRewardWrapper(env, goal_reward=config.get('GOAL_REWARD', 100.0))

    elif config['ENV_NAME'] == 'FourRooms-misc':
        env, env_params = gymnax.make(config["ENV_NAME"], use_visual_obs=True, goal_fixed=(11,11), pos_fixed = (3,1))
        env_params = env_params.replace(
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE'], 
            fail_prob=config.get('FAIL_PROB', 0.1)
        )
        env = TerminalInfoWrapper(env)
        
    elif config['ENV_NAME'] == 'FourRooms-cont':
        from envs.wrappers import ContinuingWrapper
        env, env_params = gymnax.make('FourRooms-misc', use_visual_obs=True, goal_fixed=(11,11), pos_fixed = (3,1))
        env_params = env_params.replace(
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE'], 
            fail_prob=config.get('FAIL_PROB', 0.1)
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    elif config['ENV_NAME'].lower() in ['eightrooms', 'eightrooms-misc']:
        from envs.eightrooms import EightRooms, EightRoomsParams
        env = EightRooms(use_visual_obs=True, goal_fixed=(23, 11), pos_fixed=(3, 1))
        env_params = EightRoomsParams(
            max_steps_in_episode=config.get('MAX_STEPS_IN_EPISODE', 1e6),
            fail_prob=config.get('FAIL_PROB', 0.01)
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower() in ['eightrooms-cont']:
        from envs.eightrooms import EightRooms, EightRoomsParams
        from envs.wrappers import ContinuingWrapper
        env = EightRooms(use_visual_obs=True, goal_fixed=(23, 11), pos_fixed=(3, 1))
        env_params = EightRoomsParams(
            max_steps_in_episode=config.get('MAX_STEPS_IN_EPISODE', 1e6),
            fail_prob=config.get('FAIL_PROB', 0.01)
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    elif config['ENV_NAME'].lower() in ['eightrooms-dense', 'eightrooms_dense', 'eightroomsdense', 'eightrooms-misc-dense']:
        from envs.eightrooms import EightRoomsDense, EightRoomsDenseParams
        env = EightRoomsDense(
            use_visual_obs=True,
            goal_fixed=(23, 11),
            pos_fixed=(3, 1),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 1.0),
        )
        env_params = EightRoomsDenseParams(
            max_steps_in_episode=config.get('MAX_STEPS_IN_EPISODE', 1e6),
            fail_prob=config.get('FAIL_PROB', 0.01),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 1.0),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower() in ['eightrooms-dense-cont', 'eightrooms_dense_cont', 'eightroomsdense-cont']:
        from envs.eightrooms import EightRoomsDense, EightRoomsDenseParams
        from envs.wrappers import ContinuingWrapper
        env = EightRoomsDense(
            use_visual_obs=True,
            goal_fixed=(23, 11),
            pos_fixed=(3, 1),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 1.0),
        )
        env_params = EightRoomsDenseParams(
            max_steps_in_episode=config.get('MAX_STEPS_IN_EPISODE', 1e6),
            fail_prob=config.get('FAIL_PROB', 0.01),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 1.0),
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    elif config['ENV_NAME'].lower().replace('_', '').replace('-', '') in [
        'continuouseightrooms',
        'continuouseightroomsmisc',
        'eightroomscontinuous',
        'eightroomscontcontrol',
    ]:
        from envs.continuous_eightrooms import ContinuousEightRooms, ContinuousEightRoomsParams
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        if use_visual:
            config['NETWORK_TYPE'] = 'cnn'
        else:
            config['NETWORK_TYPE'] = 'mlp'
        env = ContinuousEightRooms(
            use_visual_obs=use_visual,
            include_vel_in_obs=config.get('INCLUDE_VEL_IN_OBS', False),
            goal_fixed=config.get('GOAL_POS', (23.5, 11.5)),
            pos_fixed=config.get('START_POS', (3.5, 1.5)),
            substeps=int(config.get('SUBSTEPS', 4)),
        )
        env_params = ContinuousEightRoomsParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1000)),
            step_size=float(config.get('STEP_SIZE', 0.5)),
            agent_radius=float(config.get('AGENT_RADIUS', 0.2)),
            goal_radius=float(config.get('GOAL_RADIUS', 0.6)),
            action_noise=float(config.get('ACTION_NOISE', 0.0)),
            fail_prob=float(config.get('FAIL_PROB', 0.0)),
            control_mode=int(config.get('CONTROL_MODE', 0)),
            damping=float(config.get('DAMPING', 0.2)),
            dt=float(config.get('DT', 0.1)),
            max_vel=float(config.get('MAX_VEL', 1.0)),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower().replace('_', '').replace('-', '') in [
        'continuouseightroomscont',
        'eightroomscontinuouscont',
    ]:
        from envs.continuous_eightrooms import ContinuousEightRooms, ContinuousEightRoomsParams
        from envs.wrappers import ContinuingWrapper
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        if use_visual:
            config['NETWORK_TYPE'] = 'cnn'
        else:
            config['NETWORK_TYPE'] = 'mlp'
        env = ContinuousEightRooms(
            use_visual_obs=use_visual,
            include_vel_in_obs=config.get('INCLUDE_VEL_IN_OBS', False),
            goal_fixed=config.get('GOAL_POS', (23.5, 11.5)),
            pos_fixed=config.get('START_POS', (3.5, 1.5)),
            substeps=int(config.get('SUBSTEPS', 4)),
        )
        env_params = ContinuousEightRoomsParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1000)),
            step_size=float(config.get('STEP_SIZE', 0.5)),
            agent_radius=float(config.get('AGENT_RADIUS', 0.2)),
            goal_radius=float(config.get('GOAL_RADIUS', 0.6)),
            action_noise=float(config.get('ACTION_NOISE', 0.0)),
            fail_prob=float(config.get('FAIL_PROB', 0.0)),
            control_mode=int(config.get('CONTROL_MODE', 0)),
            damping=float(config.get('DAMPING', 0.2)),
            dt=float(config.get('DT', 0.1)),
            max_vel=float(config.get('MAX_VEL', 1.0)),
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    elif config['ENV_NAME'].lower().replace('_', '').replace('-', '') in [
        'continuouseightroomsdense',
        'continuouseightroomsmiscdense',
        'eightroomscontinuousdense',
    ]:
        from envs.continuous_eightrooms import ContinuousEightRoomsDense, ContinuousEightRoomsDenseParams
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        if use_visual:
            config['NETWORK_TYPE'] = 'cnn'
        else:
            config['NETWORK_TYPE'] = 'mlp'
        env = ContinuousEightRoomsDense(
            use_visual_obs=use_visual,
            include_vel_in_obs=config.get('INCLUDE_VEL_IN_OBS', False),
            goal_fixed=config.get('GOAL_POS', (23.5, 11.5)),
            pos_fixed=config.get('START_POS', (3.5, 1.5)),
            substeps=int(config.get('SUBSTEPS', 4)),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
        env_params = ContinuousEightRoomsDenseParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1000)),
            step_size=float(config.get('STEP_SIZE', 0.5)),
            agent_radius=float(config.get('AGENT_RADIUS', 0.2)),
            goal_radius=float(config.get('GOAL_RADIUS', 0.6)),
            action_noise=float(config.get('ACTION_NOISE', 0.0)),
            fail_prob=float(config.get('FAIL_PROB', 0.0)),
            control_mode=int(config.get('CONTROL_MODE', 0)),
            damping=float(config.get('DAMPING', 0.2)),
            dt=float(config.get('DT', 0.1)),
            max_vel=float(config.get('MAX_VEL', 1.0)),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower().replace('_', '').replace('-', '') in [
        'continuouseightroomsdensecont',
        'eightroomscontinuousdensecont',
    ]:
        from envs.continuous_eightrooms import ContinuousEightRoomsDense, ContinuousEightRoomsDenseParams
        from envs.wrappers import ContinuingWrapper
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        if use_visual:
            config['NETWORK_TYPE'] = 'cnn'
        else:
            config['NETWORK_TYPE'] = 'mlp'
        env = ContinuousEightRoomsDense(
            use_visual_obs=use_visual,
            include_vel_in_obs=config.get('INCLUDE_VEL_IN_OBS', False),
            goal_fixed=config.get('GOAL_POS', (23.5, 11.5)),
            pos_fixed=config.get('START_POS', (3.5, 1.5)),
            substeps=int(config.get('SUBSTEPS', 4)),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
        env_params = ContinuousEightRoomsDenseParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1000)),
            step_size=float(config.get('STEP_SIZE', 0.5)),
            agent_radius=float(config.get('AGENT_RADIUS', 0.2)),
            goal_radius=float(config.get('GOAL_RADIUS', 0.6)),
            action_noise=float(config.get('ACTION_NOISE', 0.0)),
            fail_prob=float(config.get('FAIL_PROB', 0.0)),
            control_mode=int(config.get('CONTROL_MODE', 0)),
            damping=float(config.get('DAMPING', 0.2)),
            dt=float(config.get('DT', 0.1)),
            max_vel=float(config.get('MAX_VEL', 1.0)),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    elif config['ENV_NAME'].lower().replace('_', '').replace('-', '') in [
        'continuousfourrooms',
        'continuousfourroomsmisc',
        'fourroomscontinuous',
        'fourroomscontcontrol',
    ]:
        from envs.continuous_fourrooms import ContinuousFourRooms, ContinuousFourRoomsParams
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        if use_visual:
            config['NETWORK_TYPE'] = 'cnn'
        else:
            config['NETWORK_TYPE'] = 'mlp'
        env = ContinuousFourRooms(
            use_visual_obs=use_visual,
            include_vel_in_obs=config.get('INCLUDE_VEL_IN_OBS', False),
            goal_fixed=config.get('GOAL_POS', (11.5, 11.5)),
            pos_fixed=config.get('START_POS', (3.5, 1.5)),
            substeps=int(config.get('SUBSTEPS', 4)),
        )
        env_params = ContinuousFourRoomsParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1000)),
            step_size=float(config.get('STEP_SIZE', 0.5)),
            agent_radius=float(config.get('AGENT_RADIUS', 0.2)),
            goal_radius=float(config.get('GOAL_RADIUS', 0.6)),
            action_noise=float(config.get('ACTION_NOISE', 0.0)),
            fail_prob=float(config.get('FAIL_PROB', 0.0)),
            control_mode=int(config.get('CONTROL_MODE', 0)),
            damping=float(config.get('DAMPING', 0.2)),
            dt=float(config.get('DT', 0.1)),
            max_vel=float(config.get('MAX_VEL', 1.0)),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower().replace('_', '').replace('-', '') in [
        'continuousfourroomsdense',
        'continuousfourroomsmiscdense',
        'fourroomscontinuousdense',
    ]:
        from envs.continuous_fourrooms import ContinuousFourRoomsDense, ContinuousFourRoomsDenseParams
        use_visual = config.get('USE_VISUAL_OBS', config.get('NETWORK_TYPE') == 'cnn')
        if use_visual:
            config['NETWORK_TYPE'] = 'cnn'
        else:
            config['NETWORK_TYPE'] = 'mlp'
        env = ContinuousFourRoomsDense(
            use_visual_obs=use_visual,
            include_vel_in_obs=config.get('INCLUDE_VEL_IN_OBS', False),
            goal_fixed=config.get('GOAL_POS', (11.5, 11.5)),
            pos_fixed=config.get('START_POS', (3.5, 1.5)),
            substeps=int(config.get('SUBSTEPS', 4)),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
        env_params = ContinuousFourRoomsDenseParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1000)),
            step_size=float(config.get('STEP_SIZE', 0.5)),
            agent_radius=float(config.get('AGENT_RADIUS', 0.2)),
            goal_radius=float(config.get('GOAL_RADIUS', 0.6)),
            action_noise=float(config.get('ACTION_NOISE', 0.0)),
            fail_prob=float(config.get('FAIL_PROB', 0.0)),
            control_mode=int(config.get('CONTROL_MODE', 0)),
            damping=float(config.get('DAMPING', 0.2)),
            dt=float(config.get('DT', 0.1)),
            max_vel=float(config.get('MAX_VEL', 1.0)),
            gamma=config.get('GAMMA', 0.99),
            potential_scale=config.get('POTENTIAL_SCALE', 0.03125),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'] == 'boyan':
        # Create our lightweight mock primitives right here
        env = MatrixMockEnv(size=20, use_visual_obs=config.get("USE_VISUAL_OBS", True))
        env_params = BoyanParams(
            fail_prob=0.0, 
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE']
        )
    elif config['ENV_NAME'].lower() in ['whirlpool', 'whirlpool-misc']:
        from envs.whirlpool_env import Whirlpool, EnvParams
        env = Whirlpool(size=config.get('ENV_SIZE', 21), use_visual_obs=True)
        env_params = EnvParams(
            fail_prob=config.get('FAIL_PROB', 0.5),
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1e6)),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower() in ['whirlpool-cont']:
        from envs.whirlpool_env import Whirlpool, EnvParams
        from envs.wrappers import ContinuingWrapper
        env = Whirlpool(size=config.get('ENV_SIZE', 21), use_visual_obs=True)
        env_params = EnvParams(
            fail_prob=config.get('FAIL_PROB', 0.5),
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1e6)),
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
    
    print('Env:', config['ENV_NAME'])
    print('Default Obs Shape:', env.observation_space(env_params).shape)
    
    env = LogWrapper(env)
    
    if isinstance(env.action_space(env_params), spaces.Box):
        env = ClipAction(env)
    
    if config.get("NETWORK_TYPE") == "mlp":
        if len(env.observation_space(env_params).shape) > 1:
            env = FlattenObservationWrapper(env)
    if config.get("NETWORK_TYPE") == "cnn":
        if len(env.observation_space(env_params).shape) == 2:
            env = AddChannelWrapper(env)
    if config.get("NORMALIZE_OBS", False):
        env = NormalizeObservationWrapper(env) 
    
    print('Obs Shape:', env.observation_space(env_params).shape)
    print('Action Shape:', env.action_space(env_params).shape)
    return env, env_params
    
def _loss_fn(params, network, traj_batch, gae, targets, config):
    # Critic loss
    value_loss = v_loss_fn(params, network, traj_batch, gae, targets, config)

    # Actor loss
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        config.get('POLICY_COEFF', 1.0) * loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)

def _loss_fn_no_w(params, network, traj_batch, gae, targets, config):
    # Critic loss
    value_loss = no_w_v_loss_fn(params, network, traj_batch, gae, targets, config)

    # Actor loss
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)    

def post_process_advantage(advantages, config, weights=None):
    """
    Standardizes and clips advantages for PPO policy optimization.
    
    If weights are provided (e.g. w = mu[:-1, None] * old_pi for exact methods),
    computes the weighted mean and weighted standard deviation over state-action visitation.
    Otherwise (e.g. sampled rollouts in standard PPO and hybrid scripts),
    computes the unweighted sample mean and standard deviation.
    
    Supports soft floor (ADV_STD_FLOOR) to prevent noise explosion near initialization,
    and outlier clipping (ADV_CLIP).
    """
    std_floor = config.get("ADV_STD_FLOOR", 0.1)
    adv_clip = config.get("ADV_CLIP", 3.0)

    if weights is not None:
        if weights.ndim == 1 and advantages.ndim == 2:
            weights = weights[:, None]
        w = weights / jnp.sum(weights)
        mean_adv = jnp.sum(w * advantages)
        var_adv = jnp.sum(w * (advantages - mean_adv) ** 2)
        std_adv = jnp.sqrt(var_adv + 1e-8)
    else:
        mean_adv = jnp.mean(advantages)
        var_adv = jnp.var(advantages)
        std_adv = jnp.sqrt(var_adv + 1e-8)

    if std_floor is not None and std_floor > 0.0:
        denom = jnp.maximum(std_adv, std_floor)
    else:
        denom = std_adv + 1e-8

    adv_norm = (advantages - mean_adv) / denom

    if adv_clip is not None and adv_clip > 0.0:
        adv_norm = jnp.clip(adv_norm, -adv_clip, adv_clip)

    return jax.lax.stop_gradient(adv_norm)


def compute_exact_advantage(P, R, P_pi, R_pi, v, γ, λ):
    """Computes exact GAE advantages across all non-terminal states and actions."""
    # δ_gae = (I - γ * λ * P_pi)^(-1) δ is the discounted sum of TD errors from step 1 onward.
    # At step 0, action a has immediate TD error δ(s, a) = R(s, a) + γ * v(s') - v(s).
    # Future TD errors from step 1 onward are discounted by γ * λ:
    # A^GAE(s, a) = δ + γ * λ * E_{s'}[δ_gae(s')]
    #             = R(s, a) + γ * E_{s'}[v(s') + λ * δ_gae(s')] - v(s)
    I = jnp.eye(P_pi.shape[0])
    L_pi = jnp.linalg.inv(I - γ * λ * P_pi)
    δ = R_pi + γ * (P_pi @ v) - v
    δ_gae = L_pi @ δ

    R_sa = jnp.einsum("sam,sam->sa", P[:-1], R[:-1])
    Q_sa = R_sa + γ * jnp.einsum("sam,m->sa", P[:-1], v + λ * δ_gae)
    return Q_sa - v[:-1, None]


def pi_loss_fn(params, network, traj_batch, gae, config):
    pi = network.apply(params, traj_batch.obs, method=network.policy)
    log_prob = pi.log_prob(traj_batch.action)

    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = post_process_advantage(gae, config)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - config["CLIP_EPS"],
            1.0 + config["CLIP_EPS"],
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    return loss_actor, entropy


def ppo_clipped_v_loss(traj_batch, value_pred, targets, config):
    e = config.get("VF_CLIP", None)
    if e is None or e is False or (isinstance(e, (int, float)) and e <= 0):
        return 0.5 * jnp.mean(jnp.square(value_pred - targets))
    value_pred_clipped = traj_batch.value + (
        value_pred - traj_batch.value
    ).clip(-e, e)
    value_losses = jnp.square(value_pred - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    
def v_loss_fn(params, network, traj_batch, gae, targets, config):
    # VALUE LOSS
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    value_loss = ppo_clipped_v_loss(traj_batch, value_pred, targets, config)
    total_loss = config["VF_COEF"] * value_loss
    return total_loss

def v_loss_fn_laplacian_smoothing(params, network, traj_batch, targets, config):
    gamma = config["GAMMA"]
    c = config["VF_CLIP"]
    # 1. Current State Predictions & Errors (e_i)
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    value_pred_clipped = traj_batch.value + (
        value_pred - traj_batch.value).clip(-c,c)
    e_i = targets - value_pred_clipped 
    base_ve_loss = 0.5 * jnp.mean(e_i ** 2)
    # 2. Next State Predictions & Errors (e_j)
    # Requires traj_batch to contain the adjacent (s, G) pairs
    next_value_pred = network.apply(params, traj_batch.next_obs, method=network.value)
    next_v_fixed = jax.lax.stop_gradient(next_value_pred)
    next_value_pred_clipped = next_v_fixed + (
        next_value_pred - next_v_fixed).clip(-c,c)
    e_j = traj_batch.next_target - next_value_pred_clipped
    valid_mask = 1.0 - traj_batch.done
    n_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)
    laplacian_loss = 0.5 * jnp.sum(valid_mask * (e_i - e_j) ** 2) / n_valid
    # Combine using the exact Dirichlet expansion weights
    weight_laplacian = gamma * config['LAPLACE_SMOOTHING_COEFF']
    dirichlet_value_loss = (1 - weight_laplacian) * base_ve_loss + weight_laplacian * laplacian_loss
    total_loss = config["VF_COEF"] * dirichlet_value_loss
    return total_loss, {"base_ve_loss": base_ve_loss, "laplacian_loss": laplacian_loss, "total_loss": total_loss}

def no_w_v_loss_fn(params, network, traj_batch, gae, targets, config):
    # ---------------------------------------------------------
    # Firewalled Parameters
    # ---------------------------------------------------------
    def freeze_w_map(path, val):
        is_w = any(getattr(p, 'key', None) in ('w_layer', 'critic_head') or 
                   'w_layer' in str(p) or 'critic_head' in str(p) for p in path)
        return jax.lax.stop_gradient(val) if is_w else val
    
    params_w_frozen = jax.tree_util.tree_map_with_path(freeze_w_map, params)
    value_for_phi = network.apply(params_w_frozen, traj_batch.obs, method = network.value)
    loss_phi = ppo_clipped_v_loss(traj_batch, value_for_phi, targets, config)
    return config["VF_COEF"] * loss_phi

def v_loss_fn_no_grad(params, network, traj_batch, gae, targets, config):
    "No update to phi."
    # 1. Forward pass through the CNN to get the features
    phi = network.apply(params, traj_batch.obs, method=network.value_features)
    
    # 2. SEVER THE GRAPH: Gradients from the value loss cannot pass this point.
    # The CNN weights will receive zero gradient from this loss function.
    phi_freeze = jax.lax.stop_gradient(phi)
    
    # 3. Forward pass through ONLY the linear head using the frozen features
    value_pred = network.apply(params, phi_freeze, method=network.value_from_features)
    value_loss = ppo_clipped_v_loss(traj_batch, value_pred, targets, config)
    total_loss = config["VF_COEF"] * value_loss
    return total_loss

def shuffle_and_batch(rng, transitions, n_minibatches):
    def preprocess_transition(x, rng):
        x = x.reshape(-1, *x.shape[2:])  # num_steps*num_envs (batch_size), ...
        x = jax.random.permutation(rng, x)  # shuffle the transitions
        x = x.reshape(n_minibatches, -1, *x.shape[1:])  # num_mini_updates, batch_size/num_mini_updates, ...
        return x
    minibatches = jax.tree.map(lambda x: preprocess_transition(x, rng), transitions)  # num_actors*num_envs (batch_size), ...
    return minibatches


def add_values_to_metric(config, metric, evaluator, network, train_state, traj_batch, compute_true_vals = True):
    """Uses evaluator to compute the per-state quantities and append them to metric."""
    if evaluator:
        pi, v_pred = network.apply(train_state.params, evaluator.obs_stack)
        pi = jnp.vstack([pi, jnp.zeros((1, pi.shape[-1]))]) # assumes terminal state.
        
        Φ = network.apply(train_state.params, evaluator.obs_stack, method=network.value_features)
        Φ = jnp.vstack([Φ, jnp.zeros((1, Φ.shape[-1]))])  # assumes terminal state.
        
        v_pred = network.apply(train_state.params, Φ, method=network.value_from_features)
    
    # True value
    if compute_true_vals:
        # The evaluator dictates the exact ground truth shapes here
        v = evaluator.compute_true_values(pi)

    # 4. Visitation Logic
    obs = jnp.asarray(traj_batch.obs)
    next_obs = jnp.asarray(traj_batch.next_obs)
    env_name = config.get("ENV_NAME", "")
    
    if env_name in {"FourRooms-misc", "FourRoomsCustom-v0"} or "SparseMaze" in env_name:
        if obs.ndim >= 5:
            metric['visitation_count'] = next_obs[..., 1].sum(axis=(0, 1))
        elif obs.ndim >= 3 and obs.shape[-1] >= 2:
            size = traj_batch.reward.shape[0] 
            pos = next_obs[..., :2].astype(jnp.int32)
            y = jnp.clip(pos[..., 0], 0, size - 1).reshape(-1)
            x = jnp.clip(pos[..., 1], 0, size - 1).reshape(-1)
            counts = jnp.zeros((size, size), dtype=jnp.float32)
            metric['visitation_count'] = counts.at[y, x].add(1.0)
    
    # 5. Error Metrics (Perfect shape alignment guaranteed by the evaluator)    
    metric.update({
        "v": v,
        "v_pred": v_pred,
        "pi": pi,
        "Empirical MSVE": jnp.mean((v - v_pred)**2) ,
    })
    
    return metric


# def calculate_gae(traj_batch, γ, λ,):

#     def _get_advantages(gae, transition):
#         done = transition.done

#         delta = transition.reward + γ * transition.next_value * (1 - done) - transition.value
#         gae = delta + (γ * λ * (1 - done) * gae)
        
#         return gae, gae

#     initial_accs = jnp.zeros_like(traj_batch.value[0])
#     _, advantages = jax.lax.scan(
#         _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
#     )
    
#     return (advantages, advantages + traj_batch.value)

def calculate_gae(traj_batch, γ, λ):
    def _get_advantages(gae, transition):
        done = transition.done
        is_timeout = transition.info["is_timeout"]

        # MASK 1: Value Bootstrapping
        true_terminal = done & ~is_timeout
        bootstrap_mask = 1.0 - true_terminal

        # MASK 2: GAE Accumulation (Trajectory Boundary)
        # Sever the GAE chain if the environment reset for ANY reason (terminal or timeout).
        # The 'gae' variable coming from the future belongs to a different episode.
        boundary_mask = 1.0 - done

        # 1. Compute TD Error (Safely bootstraps through timeouts)
        delta = transition.reward + γ * transition.next_value * bootstrap_mask - transition.value
        
        # 2. Accumulate GAE (Safely breaks at episode resets)
        gae = delta + (γ * λ * boundary_mask * gae)
        
        return gae, gae

    initial_accs = jnp.zeros_like(traj_batch.value[0])
    _, advantages = jax.lax.scan(
        _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
    )
    
    return (advantages, advantages + traj_batch.value)


def find_closest_divisor(total, requested):
    for n in range(requested, 0, -1):
        if total % n == 0:
            return n
    return 1

def inject_weights(train_state, w):
    """Overwrites the critic_head weights, preserving the original PyTree type."""
    # 1. Slice the weights (last dim is bias)
    kernel_weights = jnp.expand_dims(w[:-1], axis=-1)
    bias_weight = w[-1:]
    
    # 2. Define the new layer dictionary
    new_head = {
        'kernel': kernel_weights,
        'bias': bias_weight
    }
    
    # 3. Inject it while preserving the container type (dict vs frozendict)
    params = train_state.params
    if not isinstance(params, dict):
        params = unfreeze(params)
    
    new_params = dict(params)
    new_params['params'] = dict(new_params['params'])
    if 'critic_head' in new_params['params']:
        new_params['params']['critic_head'] = new_head
    if 'w_layer' in new_params['params']:
        new_params['params']['w_layer'] = new_head

    if isinstance(train_state.params, dict):
        return train_state.replace(params=new_params)
    else:
        return train_state.replace(params=freeze(new_params))

def get_evaluation_policies(base_config, evaluator):
    """Gets a target policy for evaluation. Returns (policy_fn, policy_matrix):
        - policy_fn: a function from obs to action distribution
        - policy_matrix: a matrix of size |S| x |A| of the policy
        This function constructs these functions for either an epsilon-optimal policy or a trained actor network.
    """
    if base_config.get("USE_GREEDY_POLICY", False):
        import core.bellman_error as bellman_error
        import distrax
        if hasattr(evaluator, "get_optimal_value_function"):
            V_star = evaluator.get_optimal_value_function()
        else:
            V_star = jnp.zeros(evaluator.num_total_states)
        
        greedy_actions = bellman_error.compute_greedy_policy(evaluator.P, evaluator.R, evaluator.gamma, V_star)
        pi_greedy = jax.nn.one_hot(greedy_actions, evaluator.num_actions)
        epsilon = base_config.get("POLICY_EPSILON", 0.0)
        pi_eps = (1 - epsilon) * pi_greedy + (epsilon / evaluator.num_actions) * jnp.ones_like(pi_greedy)
        
        def policy_fn(obs):
            obs_flat = obs.reshape((obs.shape[0], -1)) if obs.ndim > 1 else obs.flatten()[None, :]
            stack_flat = evaluator.obs_stack.reshape((evaluator.obs_stack.shape[0], -1))
            diffs = jnp.sum((obs_flat[:, None, :] - stack_flat[None, :, :])**2, axis=-1)
            state_indices = jnp.argmin(diffs, axis=-1)
            probs = pi_eps[state_indices]
            return distrax.Categorical(probs=probs)
            
        # pi_eps already has shape (num_total_states, A), we just need to ensure the terminal state is uniform
        pi_eps = pi_eps.at[-1, :].set(jnp.ones(evaluator.num_actions) / evaluator.num_actions)

        is_continuous_env = base_config.get("ENV_NAME", "").lower().replace('_', '').replace('-', '') in [
            'continuouseightrooms',
            'continuouseightroomsdense',
            'continuouseightroomsmisc',
            'continuouseightroomsmiscdense',
            'eightroomscontinuous',
            'eightroomscontinuousdense',
            'continuouseightroomscont',
            'eightroomscontinuouscont',
        ]
        if is_continuous_env:
            from flax import struct
            action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)

            @struct.dataclass
            class CardinalContinuousPolicy:
                cat_dist: Any
                basis: jax.Array

                def sample(self, seed=None):
                    idx = self.cat_dist.sample(seed=seed)
                    return self.basis[idx]

                def log_prob(self, value):
                    diffs = jnp.sum((value[..., None, :] - self.basis)**2, axis=-1)
                    idx = jnp.argmin(diffs, axis=-1)
                    return self.cat_dist.log_prob(idx)

                def mode(self):
                    return self.basis[self.cat_dist.mode()]

            orig_policy_fn = policy_fn
            policy_fn = lambda obs: CardinalContinuousPolicy(cat_dist=orig_policy_fn(obs), basis=action_basis)

        return policy_fn, pi_eps
    else:
        import core.utils as utils
        from scripts.sweep_pipeline import resolve_model_load_dir
        resolved_dir = resolve_model_load_dir(base_config.get('MODEL_LOAD_DIR'), base_config['ENV_NAME'], 'results')
        model_dir = 'ppo/' + resolved_dir if not resolved_dir.startswith('ppo/') else resolved_dir
        print(model_dir)
        _, out = utils.load_run_data(model_dir, base_config['ENV_NAME'], 'results') 
        policy_train_state = out['runner_state'][0]
        policy_params = jax.tree_util.tree_map(lambda x: x[0], policy_train_state.params)
        
        def policy_fn(obs):
            pi, _ = policy_train_state.apply_fn(policy_params, obs)
            # handle cases where apply_fn returns a tuple (pi, value) or just pi
            if isinstance(pi, tuple):
                pi = pi[0]
            return pi
            
        # build the matrix
        pi_dist = policy_fn(evaluator.obs_stack)
        if hasattr(pi_dist, "probs"):
            pi_probs = pi_dist.probs
        else:
            action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)
            log_p = jax.vmap(lambda a: pi_dist.log_prob(a), in_axes=0, out_axes=-1)(action_basis)
            pi_probs = jax.nn.softmax(log_p, axis=-1)
        terminal_policy = jnp.ones([1, evaluator.num_actions], dtype=pi_probs.dtype) / evaluator.num_actions
        policy_matrix = jnp.vstack([pi_probs, terminal_policy])
        
        return policy_fn, policy_matrix


# ==============================================================================
# Sampled E and E(lambda) Critic Losses & Helpers
# ==============================================================================

def e_lambda_fixed_loss_fn(params, network, traj_batch, gae, targets, config):
    """
    Unclipped critic MSE loss + clipped PPO actor loss for E_lambda_fixed.
    Targets are precomputed symmetrized E(lambda) targets.
    """
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    value_loss = 0.5 * jnp.mean(jnp.square(value_pred - targets))
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        config.get("POLICY_COEFF", 1.0) * loss_actor
        + config.get("VF_COEF", 0.5) * value_loss
        - config.get("ENT_COEF", 0.01) * entropy
    )
    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "actor_loss": loss_actor,
        "entropy": entropy,
    }
    return total_loss, losses


def e_critic_loss(
    v_i,
    targets_i,
    v_j,
    targets_j,
    true_terminal,
    gamma,
    variant="oct8",
    is_start=None,
    is_timeout=None,
    nu=None,
    v_aux_i=None,
    v_aux_j=None,
    rewards=None,
    aux_coef=1.0,
    aux_targets_i=None,
    w_mc=None,
    w_dir=None,
    return_details=False,
):
    """
    Computes sampled E-loss across original and debiased v_aux formulations:
    - 'sep14' / 'uncorrected': pure positive semi-definite (magnitude + Dirichlet smoothness)
    - 'corrected' / 'oct5': adds difference of second moments 0.5 * gamma * (mean(e_i^2) - mean(e_j^2))
    - 'oct8' / 'start_anchored': adds positive start-state anchor 0.5 * gamma * E_{s0}[e(s0)^2]
    - 'forward_e' / 'forward_only': semi-gradient form dropping grad v_j branch to eliminate double-sampling without aux net
    - 'bilinear_v_aux': direct unbiased bilinear form <e, delta>_D using v_aux for the adjoint error
    - 'surgical_v_aux': Split-branch Dirichlet form: real returns on v_i branch (1-step TD reward, zero return noise), v_aux on v_j branch (smooth Dirichlet Laplacian, zero return noise & zero double-sampling bias)
    - 'dirichlet_v_aux': Oct 8 Dirichlet form using v_aux in the entire Laplacian term and G_t in anchors

    Accepts relative term weights (w_mc, w_dir) as input:
    - Default (w_mc=None, w_dir=None): w_mc = 1.0 - gamma, w_dir = 0.5 * gamma (standard E(0))
    - BSMC: w_mc = 0.5 * (1 - gamma) / (1 - gamma * lambda), w_dir = 0.5 * gamma * (1 - lambda) / (1 - gamma * lambda)
    """
    sg = jax.lax.stop_gradient

    # Empirical MC errors (from trajectory returns G_t, G_{t+1})
    e_i = targets_i - v_i
    e_j = jnp.where(true_terminal, 0.0, targets_j - v_j)

    # Auxiliary head loss (trains v_aux_i to predict aux_targets_i, e.g. TD(lambda=0.9), or targets_i = G_t)
    if v_aux_i is not None:
        targets_aux = aux_targets_i if aux_targets_i is not None else targets_i
        aux_mc_loss = aux_coef * jnp.mean((sg(targets_aux) - v_aux_i) ** 2)
        e_tilde_i = sg(v_aux_i) - v_i
        e_tilde_j = jnp.where(true_terminal, 0.0, sg(v_aux_j) - v_j)
    else:
        aux_mc_loss = 0.0
        e_tilde_i = None
        e_tilde_j = None

    weight_mc = (1.0 - gamma) if w_mc is None else w_mc
    weight_dir = (0.5 * gamma) if w_dir is None else w_dir

    # --- VARIANT: Bilinear <e, delta>_D (Direct sample, zero boundary terms needed) ---
    if variant == "bilinear_v_aux":
        if v_aux_i is None:
            raise ValueError("variant='bilinear_v_aux' requires v_aux_i and v_aux_j.")
        r_t = rewards if rewards is not None else sg(targets_i - gamma * jnp.where(true_terminal, 0.0, targets_j))
        v_j_masked = jnp.where(true_terminal, 0.0, v_j)
        delta_t = r_t + gamma * v_j_masked - v_i

        # Forward TD(0) term + Adjoint E term (matches exact gradient of E = <e, delta>_D)
        bilinear_surrogate = sg(delta_t) * (-v_i) + sg(e_tilde_i) * delta_t
        value_loss = jnp.mean(bilinear_surrogate) + aux_mc_loss

        magnitude_loss = weight_mc * jnp.mean(e_i ** 2)
        laplacian_loss = weight_dir * jnp.mean((e_tilde_i - e_tilde_j) ** 2)
        if return_details:
            return value_loss, {
                "value_loss": value_loss,
                "magnitude_loss": magnitude_loss,
                "dirichlet_loss": laplacian_loss,
                "start_loss": 0.0,
                "timeout_loss": 0.0,
                "aux_loss": aux_mc_loss,
            }
        return value_loss, magnitude_loss, laplacian_loss

    magnitude_loss = weight_mc * jnp.mean(e_i ** 2)

    if variant in ["surgical_v_aux", "surgical", "aux", "v_aux"]:
        if v_aux_i is None:
            raise ValueError(f"variant='{variant}' requires v_aux_i and v_aux_j.")
        # Terms 1 & 2 (-e_i grad v_i + e_j grad v_i): real returns cancel post-(t+1) noise, yielding exact 1-step TD(0) reward
        branch_v_i = (e_i - sg(e_j)) ** 2
        # Terms 3 & 4 (-e_tilde_j grad v_j + e_tilde_i grad v_j): v_aux predictions eliminate all post-(t+1) return noise & double-sampling bias
        branch_v_j = jnp.where(true_terminal, 0.0, (sg(e_tilde_i) - e_tilde_j) ** 2)
        laplacian_loss = weight_dir * jnp.mean(branch_v_i + branch_v_j)
    elif variant in ["forward_e", "forward_only", "semi_gradient_e", "forward", "fwd"]:
        # Terms 1 & 2 only (-e_i grad v_i + e_j grad v_i): drops grad v_j branch to eliminate double-sampling without auxiliary net
        branch_v_i = (e_i - sg(e_j)) ** 2
        laplacian_loss = weight_dir * jnp.mean(branch_v_i)
    elif variant in ["dirichlet_v_aux", "dirichlet"]:
        if v_aux_i is None:
            raise ValueError("variant='dirichlet_v_aux' requires v_aux_i and v_aux_j.")
        laplacian_loss = weight_dir * jnp.mean((e_tilde_i - e_tilde_j) ** 2)
    else:
        laplacian_loss = weight_dir * jnp.mean((e_i - e_j) ** 2)

    if variant in ["sep14", "uncorrected"]:
        value_loss = magnitude_loss + laplacian_loss
        start_anchor_loss = 0.0
        timeout_corr_loss = 0.0
    elif variant in ["corrected", "corr", "oct5"]:
        corr_loss = weight_dir * (jnp.mean(e_i ** 2) - jnp.mean(e_j ** 2))
        value_loss = magnitude_loss + laplacian_loss + corr_loss
        start_anchor_loss = 0.0
        timeout_corr_loss = 0.0
    elif variant in [
        "oct8", "start_anchored", "surgical_v_aux", "surgical",
        "dirichlet_v_aux", "forward_e", "forward_only", "semi_gradient_e",
        "forward", "fwd", "aux", "v_aux",
    ]:
        start_anchor_loss = (
            weight_dir * jnp.mean(jnp.where(is_start, e_i ** 2, 0.0))
            if is_start is not None
            else 0.0
        )
        if is_timeout is not None:
            timeout_corr_loss = -weight_dir * jnp.mean(jnp.where(is_timeout, e_j ** 2, 0.0))
        else:
            timeout_corr_loss = 0.0
        value_loss = magnitude_loss + laplacian_loss + start_anchor_loss + timeout_corr_loss + aux_mc_loss
    else:
        raise ValueError(f"Unknown E variant: {variant}")

    if return_details:
        losses = {
            "value_loss": value_loss,
            "magnitude_loss": magnitude_loss,
            "dirichlet_loss": laplacian_loss,
            "start_loss": start_anchor_loss,
            "timeout_loss": timeout_corr_loss,
            "aux_loss": aux_mc_loss,
        }
        return value_loss, losses
    return value_loss, magnitude_loss, laplacian_loss


def ppo_actor_loss(params, network, obs_mb, action_mb, log_prob_mb, advantages_mb, config):
    """
    Standard PPO actor clipped surrogate objective with entropy bonus.
    """
    pi = network.apply(params, obs_mb, method=network.policy)
    log_prob = pi.log_prob(action_mb)
    entropy = pi.entropy().mean()
    ratio = jnp.exp(log_prob - log_prob_mb)

    adv_norm = post_process_advantage(advantages_mb, config)

    surr1 = ratio * adv_norm
    surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    total_loss = actor_loss - entropy * config.get("ENT_COEF", 0.01)
    return total_loss, {
        "actor_loss": actor_loss,
        "entropy": entropy,
    }


def critic_td_lambda_loss(params, network, obs_mb, targets_mb, value_mb, config):
    """
    TD(lambda) critic loss with optional value function clipping.
    Requires precomputed lambda targets (e.g. from helpers.calculate_gae).
    """
    value_pred = network.apply(params, obs_mb, method=network.value)
    if config.get("VF_CLIP", 0.0) > 0:
        e = config["VF_CLIP"]
        value_pred_clipped = value_mb + (value_pred - value_mb).clip(-e, e)
        value_losses = jnp.square(value_pred - targets_mb)
        value_losses_clipped = jnp.square(value_pred_clipped - targets_mb)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    else:
        value_loss = 0.5 * jnp.mean((value_pred - targets_mb) ** 2)
    return value_loss, {"value_loss": value_loss}


def critic_td_zero_loss(params, network, obs_mb, next_obs_mb, reward_mb, mask_mb, value_mb, config):
    """
    TD(0) critic loss computed from scratch each minibatch using inference on v(s') and stop_gradient.
    Requires next_obs, reward, and bootstrap_mask (1 - (done & ~is_timeout)).
    """
    value_pred = network.apply(params, obs_mb, method=network.value)
    next_value_pred = network.apply(params, next_obs_mb, method=network.value)
    td_target = reward_mb + config["GAMMA"] * mask_mb * next_value_pred
    td_target = jax.lax.stop_gradient(td_target)

    if config.get("VF_CLIP", 0.0) > 0:
        e = config["VF_CLIP"]
        value_pred_clipped = value_mb + (value_pred - value_mb).clip(-e, e)
        value_losses = jnp.square(value_pred - td_target)
        value_losses_clipped = jnp.square(value_pred_clipped - td_target)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    else:
        value_loss = 0.5 * jnp.mean((value_pred - td_target) ** 2)
    return value_loss, {"value_loss": value_loss}


def critic_sampled_e_loss(params, network, obs_mb, next_obs_mb, targets_mb, next_target_mb, true_terminal_mb, config, is_start_mb=None, nu=None):
    """
    Sampled E-loss for critic (magnitude anchor + Dirichlet/Laplacian smoothness + boundary correction).
    Requires targets and aligned next_targets.
    """
    gamma = config["GAMMA"]
    variant = config.get("E_VARIANT", "oct8")
    v_i = network.apply(params, obs_mb, method=network.value)
    v_j = network.apply(params, next_obs_mb, method=network.value)
    value_loss, magnitude_loss, laplacian_loss = e_critic_loss(
        v_i, targets_mb, v_j, next_target_mb, true_terminal_mb, gamma, variant=variant, is_start=is_start_mb, nu=nu
    )
    return value_loss, {
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "laplacian_loss": laplacian_loss,
    }


def e_loss_fn(
    params,
    network,
    obs,
    action,
    log_prob_old,
    next_obs,
    true_terminal,
    next_target,
    advantages,
    targets,
    config,
    is_start=None,
):
    """
    Combined loss for PPO with Sampled E critic.
    """
    # 1. Actor Loss (PPO clipped surrogate)
    pi = network.apply(params, obs, method=network.policy)
    log_prob = pi.log_prob(action)
    entropy = pi.entropy().mean()
    ratio = jnp.exp(log_prob - log_prob_old)

    adv_norm = post_process_advantage(advantages, config)
    surr1 = ratio * adv_norm
    surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    # 2. Critic Loss (Sampled E-loss)
    gamma = config["GAMMA"]
    variant = config.get("E_VARIANT", "oct8")
    v_i = network.apply(params, obs, method=network.value)
    v_j = network.apply(params, next_obs, method=network.value)
    # Terminal absorbing state has value 0
    v_j = jnp.where(true_terminal, 0.0, v_j)

    value_loss, magnitude_loss, laplacian_loss = e_critic_loss(
        v_i, targets, v_j, next_target, true_terminal, gamma, variant=variant, is_start=is_start
    )

    total_loss = (
        config["POLICY_COEFF"] * actor_loss
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "laplacian_loss": laplacian_loss,
        "actor_loss": actor_loss,
        "entropy": entropy,
    }
    return total_loss, losses


def calculate_e_lambda_targets(
    traj_batch,
    gamma: float,
    lmbda: float,
    return_lambda: float = 1.0,
    next_value_T: jnp.ndarray = None,
):
    """
    Computes scalar value regression targets for the symmetrized E(lambda) objective
    using forward and backward error traces:
        y_t = G_t - sg[ 0.5 * gamma * (1 - lambda) * (e_{>t} + e_{<t}) ]
    """
    # 1. Baseline returns G_t and errors e_t = G_t - v_t
    _, returns = calculate_gae(traj_batch, gamma, return_lambda)
    errors = returns - traj_batch.value
    gl = gamma * lmbda

    is_timeout = traj_batch.info.get("is_timeout", jnp.zeros_like(traj_batch.done, dtype=bool))
    true_terminal = traj_batch.done & ~is_timeout
    done_f = traj_batch.done.astype(jnp.float32)
    true_term_f = true_terminal.astype(jnp.float32)

    # Continuation error at rollout buffer boundary step T
    if next_value_T is None:
        next_value_T = traj_batch.next_value[-1]
    e_T = jnp.where(true_terminal[-1], 0.0, traj_batch.next_value[-1] - next_value_T)
    errors_ext = jnp.concatenate([errors, e_T[None]], axis=0)
    e_next = errors_ext[1:]

    # 2. Forward Trace: Accumulate future errors backwards in time
    def _backward_pass(forward_trace, transition):
        done_t, true_term_t, e_nxt = transition
        valid_future = 1.0 - done_t
        trace_t = jnp.where(
            true_term_t > 0.5,
            0.0,
            e_nxt + gl * valid_future * forward_trace,
        )
        return trace_t, trace_t

    init_forward = jnp.zeros_like(errors[0])
    _, forward_traces = jax.lax.scan(
        _backward_pass,
        init_forward,
        (done_f, true_term_f, e_next),
        reverse=True,
    )

    # 3. Backward Trace: Accumulate past errors forwards in time
    prev_done = jnp.roll(done_f, shift=1, axis=0).at[0].set(1.0)
    e_prev = jnp.roll(errors, shift=1, axis=0).at[0].set(0.0)

    def _forward_pass(backward_trace, transition):
        p_done, e_prv = transition
        trace_t = (1.0 - p_done) * (e_prv + gl * backward_trace)
        return trace_t, trace_t

    init_backward = jnp.zeros_like(errors[0])
    _, backward_traces = jax.lax.scan(
        _forward_pass,
        init_backward,
        (prev_done, e_prev),
        reverse=False,
    )

    # 4. Construct Symmetrized E(lambda) scalar regression targets
    coeff = 0.5 * gamma * (1.0 - lmbda)
    smoothing_correction = coeff * (forward_traces + backward_traces)
    targets = returns - jax.lax.stop_gradient(smoothing_correction)

    diagnostics = {
        "returns": returns,
        "errors": errors,
        "forward_traces": forward_traces,
        "backward_traces": backward_traces,
        "correction": smoothing_correction,
    }
    return targets, diagnostics


def e_lambda_differentiable_critic_loss(
    values,
    targets,
    dones,
    gamma: float,
    lmbda: float,
    true_terminals=None,
    next_value_T=None,
    next_target_T=None,
):
    """
    Computes differentiable E(lambda) loss using backward moment traces (Method 2).
    Backpropagates through both values v(s_t) and future values v(s_{t+k+1}).
    """
    if true_terminals is None:
        true_terminals = dones
    if next_value_T is None:
        next_value_T = jnp.zeros_like(values[0])
    if next_target_T is None:
        next_target_T = jnp.zeros_like(targets[0])

    errors = targets - values
    gl = gamma * lmbda
    done_f = dones.astype(jnp.float32)
    true_term_f = true_terminals.astype(jnp.float32)

    # Continuation error at rollout buffer boundary step T
    e_T = jnp.where(true_terminals[-1], 0.0, next_target_T - next_value_T)
    errors_ext = jnp.concatenate([errors, e_T[None]], axis=0)
    e_next = errors_ext[1:]

    def _moment_step(traces, transition):
        w0_future, w1_future, w2_future = traces
        done_t, true_term_t, e_curr, e_nxt = transition

        valid_future = 1.0 - done_t

        w0_t = jnp.where(true_term_t > 0.5, 1.0, 1.0 + gl * valid_future * w0_future)
        w1_t = jnp.where(true_term_t > 0.5, 0.0, e_nxt + gl * valid_future * w1_future)
        w2_t = jnp.where(true_term_t > 0.5, 0.0, (e_nxt ** 2) + gl * valid_future * w2_future)

        # Dirichlet quadratic expansion: w0 * e_t^2 - 2 * w1 * e_t + w2
        dirichlet_t = w0_t * (e_curr ** 2) - 2.0 * w1_t * e_curr + w2_t

        # Traces passed backward to step t-1 are severed if episode ended (done_t == 1)
        valid_to_prev = 1.0 - done_t
        w0_to_prev = valid_to_prev * w0_t
        w1_to_prev = valid_to_prev * w1_t
        w2_to_prev = valid_to_prev * w2_t

        return (w0_to_prev, w1_to_prev, w2_to_prev), dirichlet_t

    init_traces = (
        jnp.zeros_like(errors[0]),
        jnp.zeros_like(errors[0]),
        jnp.zeros_like(errors[0]),
    )
    _, dirichlet_terms = jax.lax.scan(
        _moment_step,
        init_traces,
        (done_f, true_term_f, errors, e_next),
        reverse=True,
    )

    magnitude_weight = (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * gamma * (1.0 - lmbda)

    magnitude_loss = magnitude_weight * jnp.mean(errors ** 2)
    dirichlet_loss = dirichlet_weight * jnp.mean(dirichlet_terms)
    value_loss = magnitude_loss + dirichlet_loss

    return value_loss, magnitude_loss, dirichlet_loss


def shuffle_and_batch_envs(rng, batch, n_minibatches):
    """
    Shuffles environments (columns) and splits into minibatches,
    preserving full trajectory sequences along the time axis (axis 0).
    """
    sample_leaf = jax.tree.leaves(batch)[0]
    num_envs = sample_leaf.shape[1]
    n_minibatches = max(1, min(n_minibatches, num_envs))
    env_per_mb = num_envs // n_minibatches

    perm = jax.random.permutation(rng, num_envs)

    def _split_leaf(x):
        x_shuffled = jnp.take(x, perm, axis=1)
        x_trimmed = x_shuffled[:, : env_per_mb * n_minibatches]
        reshaped = x_trimmed.reshape(x.shape[0], n_minibatches, env_per_mb, *x.shape[2:])
        return jnp.swapaxes(reshaped, 0, 1)

    return jax.tree.map(_split_leaf, batch)


def e_lambda_differentiable_loss_fn(
    params,
    network,
    traj_batch,
    advantages,
    returns,
    config,
):
    """
    Combined PPO loss with Method 2 differentiable E(lambda) critic loss.
    """
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, advantages, config)

    values = network.apply(params, traj_batch.obs, method=network.value)
    gamma = config["GAMMA"]
    e_lambda = config["VALUE_LAMBDA"]

    # Boundary continuation at step T
    next_value_T = network.apply(params, traj_batch.next_obs[-1], method=network.value)
    next_target_T = traj_batch.next_value[-1]

    # Terminals vs Timeouts
    is_timeout = traj_batch.info.get("is_timeout", jnp.zeros_like(traj_batch.done, dtype=bool))
    true_terminal = traj_batch.done & ~is_timeout

    value_loss, magnitude_loss, dirichlet_loss = e_lambda_differentiable_critic_loss(
        values=values,
        targets=returns,
        dones=traj_batch.done,
        gamma=gamma,
        lmbda=e_lambda,
        true_terminals=true_terminal,
        next_value_T=next_value_T,
        next_target_T=next_target_T,
    )

    total_loss = (
        config["POLICY_COEFF"] * loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )

    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "actor_loss": loss_actor,
        "entropy": entropy,
    }
    return total_loss, losses


def e_lambda_geometric_critic_loss(
    values: jax.Array,
    targets: jax.Array,
    dones: jax.Array,
    gamma: float,
    lmbda: float,
    rng: jax.Array,
    aux_values: jax.Array = None,
    true_terminals: jax.Array = None,
    next_value_T: jax.Array = None,
    next_aux_value_T: jax.Array = None,
    next_target_T: jax.Array = None,
    num_start_states: int = None,
    init_steps_since_start: jax.Array = None,
    aux_loss_coef: float = 1.0,
    return_details: bool = False,
):
    """
    Computes sampled E(lambda) critic loss by sampling lookahead horizon skips
    K ~ Geometric(1 - gamma * lambda) with auxiliary critic v_aux,
    decaying start-state reset grounding, and absorbing boundary conditions.
    """
    if true_terminals is None:
        true_terminals = dones
    # Note: aux_values=values sets e_aux=0, dropping Term 1 (Forward-Only ablation)
    if aux_values is None: 
        aux_values = values
    if next_value_T is None:
        next_value_T = jnp.zeros_like(values[0])
    if next_aux_value_T is None:
        next_aux_value_T = jnp.zeros_like(aux_values[0])
    if next_target_T is None:
        next_target_T = jnp.zeros_like(targets[0])

    T, B = targets.shape[:2]
    T_start = min(T, max(1, T - 16)) if num_start_states is None else min(num_start_states, T)

    gl = gamma * lmbda
    sg = jax.lax.stop_gradient

    # 1. Error vectors & boundary extensions
    errors = sg(targets) - values
    e_aux = sg(aux_values) - values

    e_T = jnp.where(true_terminals[-1], 0.0, sg(next_target_T) - next_value_T)
    e_aux_T = jnp.where(true_terminals[-1], 0.0, sg(next_aux_value_T) - next_value_T)

    errors_ext = jnp.concatenate([errors, e_T[None]], axis=0)       # (T + 1, B)
    e_aux_ext = jnp.concatenate([e_aux, e_aux_T[None]], axis=0)     # (T + 1, B)

    # 2. Geometric jump sampling: m = K + 1 where K ~ Geom(1 - gl)
    u = jax.random.uniform(rng, shape=(T_start, B))
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T_start, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    m = jnp.maximum(jumps, 0) + 1
    t_arr = jnp.arange(T_start)[:, None]
    j = jnp.minimum(t_arr + m, T)

    # 3. Check transitions in [t, j - 1] for terminal absorption or timeouts
    is_timeout = dones & (~true_terminals)
    term_cumsum_excl = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(true_terminals.astype(jnp.int32), axis=0)],
        axis=0,
    )
    timeout_cumsum_excl = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(is_timeout.astype(jnp.int32), axis=0)],
        axis=0,
    )

    hit_true_terminal = (jnp.take_along_axis(term_cumsum_excl, j, axis=0) - term_cumsum_excl[:T_start]) > 0
    hit_timeout = (jnp.take_along_axis(timeout_cumsum_excl, j, axis=0) - timeout_cumsum[:T_start]) > 0
    hit_horizon = (t_arr + m >= T)

    e_start = errors[:T_start]
    e_aux_start = e_aux[:T_start]
    e_jump = jnp.take_along_axis(errors_ext, j, axis=0)
    e_aux_jump = jnp.take_along_axis(e_aux_ext, j, axis=0)

    diff_sq = jnp.where(
        hit_true_terminal | (hit_horizon & ~hit_timeout),
        e_start ** 2,  # Absorbs at e_term = 0 or falls back to pure MC on horizon exceedance
        jnp.where(
            hit_timeout,
            0.0,
            (e_start - sg(e_jump)) ** 2 + (sg(e_aux_start) - e_aux_jump) ** 2,
        ),
    )

    # 4. Geometrically decaying start-state reset grounding (scanning over true_terminals & timeouts)
    def scan_steps(k, step_flags):
        term, timeout = step_flags
        k_next = jnp.where(term, 0, jnp.where(timeout, 10_000, k + 1))
        return k_next, k

    init_k = jnp.full((B,), 10_000, dtype=jnp.int32) if init_steps_since_start is None else init_steps_since_start
    final_k, steps_since_start = jax.lax.scan(scan_steps, init_k, (true_terminals, is_timeout))

    start_trace = jnp.where(
        gl < 1e-6,
        jnp.where(steps_since_start[:T_start] == 0, 1.0, 0.0),
        safe_gl ** steps_since_start[:T_start],
    )

    # 5. Exact theoretical prefactors
    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    magnitude_weight = 0.5 * (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * tilde_gamma
    start_weight = 0.25 * tilde_gamma  # Double 1/2 factor: outer 1/2 * inner 1/2

    magnitude_loss = magnitude_weight * jnp.mean(e_start ** 2)
    dirichlet_loss = dirichlet_weight * jnp.mean(diff_sq)
    start_loss = start_weight * jnp.mean(start_trace * (e_start ** 2))
    value_loss = magnitude_loss + dirichlet_loss + start_loss

    aux_loss = 0.5 * jnp.mean((sg(targets) - aux_values) ** 2)
    total_loss = value_loss + aux_loss_coef * aux_loss

    losses = {
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "start_loss": start_loss,
        "aux_loss": aux_loss,
        "total_loss": total_loss,
    }

    if return_details:
        return total_loss, losses, final_k
    return value_loss, magnitude_loss, dirichlet_loss


def e_lambda_geometric_loss_fn(
    params,
    network,
    traj_batch,
    advantages,
    returns,
    config,
    rng,
    aux_params=None,
    aux_network=None,
    init_steps_since_start=None,
):
    """
    Combined PPO loss with Method 3 sampled geometric E(lambda) critic loss.
    """
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, advantages, config)

    values = network.apply(params, traj_batch.obs, method=network.value)
    gamma = config["GAMMA"]
    e_lambda = config["VALUE_LAMBDA"]

    # Boundary continuation at step T
    next_value_T = network.apply(params, traj_batch.next_obs[-1], method=network.value)
    next_target_T = traj_batch.next_value[-1]

    # Auxiliary critic evaluations if auxiliary network is provided
    if aux_network is not None and aux_params is not None:
        aux_values = aux_network.apply(aux_params, traj_batch.obs)
        next_aux_value_T = aux_network.apply(aux_params, traj_batch.next_obs[-1])
    else:
        aux_values = None
        next_aux_value_T = None

    # Terminals vs Timeouts
    is_timeout = traj_batch.info.get("is_timeout", jnp.zeros_like(traj_batch.done, dtype=bool))
    true_terminal = traj_batch.done & ~is_timeout

    _, loss_details, final_k = e_lambda_geometric_critic_loss(
        values=values,
        targets=returns,
        dones=traj_batch.done,
        gamma=gamma,
        lmbda=e_lambda,
        rng=rng,
        aux_values=aux_values,
        true_terminals=true_terminal,
        next_value_T=next_value_T,
        next_aux_value_T=next_aux_value_T,
        next_target_T=next_target_T,
        init_steps_since_start=init_steps_since_start,
        return_details=True,
    )

    value_loss = loss_details["value_loss"]
    total_loss = (
        config["POLICY_COEFF"] * loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )

    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": loss_details["magnitude_loss"],
        "dirichlet_loss": loss_details["dirichlet_loss"],
        "start_loss": loss_details["start_loss"],
        "aux_loss": loss_details.get("aux_loss", 0.0),
        "actor_loss": loss_actor,
        "entropy": entropy,
        "final_k": final_k,
    }
    return total_loss, losses


def compute_adaptive_runway_length(
    gamma: float,
    lmbda: float,
    total_steps: int = 256,
) -> int:
    """
    Computes an adaptive runway length based on gamma and lambda:
    runway = min(m_95, upper_bound), bounded below by 8.
    Here m_95 is the 95th percentile of the geometric jump length Geom(1 - gamma * lambda):
        P(m <= m_95) >= 0.95 <=> (gamma * lambda)^m_95 <= 0.05
        m_95 = ceil(log(0.05) / log(gamma * lambda))
    upper_bound is total_steps // 8 (e.g. 32 for T=256), preserving >= 87.5% of rollout steps as sources.
    """
    import math
    gl = max(1e-6, min(0.9999, float(gamma * lmbda)))
    m_95 = int(math.ceil(math.log(0.05) / math.log(gl)))
    upper_bound = max(8, int(total_steps) // 8)
    return max(8, min(m_95, upper_bound))


class ELambdaJumpTransition(NamedTuple):
    """
    Transition container pairing source states with geometric jump destination states.
    Shape before shuffle_and_batch: (T, B, ...)
    Shape inside minibatch: (minibatch_size, ...)
    """
    obs: jnp.ndarray
    dest_obs: jnp.ndarray
    target: jnp.ndarray
    dest_target: jnp.ndarray
    hit_true_terminal: jnp.ndarray
    hit_timeout: jnp.ndarray
    start_weight: jnp.ndarray
    hit_horizon: Optional[jnp.ndarray] = None
    aux_target: Optional[jnp.ndarray] = None
    action: Optional[jnp.ndarray] = None
    log_prob: Optional[jnp.ndarray] = None
    advantage: Optional[jnp.ndarray] = None
    critic_mask: Optional[jnp.ndarray] = None


def calculate_self_bootstrapped_aux_returns(
    aux_network,
    aux_params,
    traj_batch,
    gamma: float,
    return_lambda: float = 1.0,
) -> jax.Array:
    """
    Computes empirical returns for the auxiliary critic v_aux bootstrapped strictly from v_aux itself at the horizon.
    Only the final next-state value v_aux(s_T) is needed; intermediate values telescope out for return_lambda = 1.0.
    """
    final_next_aux_val = aux_network.apply(aux_params, traj_batch.next_obs[-1])
    traj_batch_aux = traj_batch._replace(next_value=traj_batch.next_value.at[-1].set(final_next_aux_val))
    _, aux_returns = calculate_gae(traj_batch_aux, gamma, return_lambda)
    return aux_returns


def prepare_e_lambda_jump_pairs(
    traj_batch,
    returns: jax.Array,
    gamma: float,
    lmbda: float,
    rng: jax.Array,
    steps_since_start: jax.Array,
    advantages: Optional[jax.Array] = None,
    aux_returns: Optional[jax.Array] = None,
    jump_mode: str = "full_mc_invalid_jump",
):
    """
    Pairs each source state s_t with its geometric jump destination s_j = s_{min(t+m, T)},
    evaluates boundary conditions in O(1), and packages an ELambdaJumpTransition of shape (T, B, ...).
    Runs ONCE per rollout batch outside the gradient descent loop.
    Returns (jump_batch, final_steps_since_start).

    Supported jump_modes:
    - 'full_mc_invalid_jump' / 'mc_full_weight' (default): Actor and critic use all T data (critic_mask = 1.0). If a sampled jump
      overshoots the buffer (t + m > T), it trains purely on the MC return target with FULL weight (1.0).
    - 'mc_invalid_jump': Actor and critic use all T data (critic_mask = 1.0). If a sampled jump
      overshoots the buffer (t + m > T), its Dirichlet smoothing term is zeroed out and it trains purely on the MC return target with weight (1 - tilde_gamma).
    - 'drop_invalid_jump': Actor uses all T data. If a sampled jump overshoots the buffer (t + m > T)
      without hitting an absorbing terminal, that transition is excluded from the critic (critic_mask = 0).
    """
    T, B = returns.shape[:2]
    gl = gamma * lmbda
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)

    # 1. Sample geometric jump lengths: m = K + 1 where K ~ Geom(1 - gl) for all T steps
    u = jax.random.uniform(rng, shape=(T, B))
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    m = jnp.maximum(jumps, 0) + 1
    t_arr = jnp.arange(T)[:, None]
    j = jnp.minimum(t_arr + m, T)

    # 2. Episode boundary queries in [t, j - 1]
    is_timeout = traj_batch.info.get("is_timeout", jnp.zeros_like(traj_batch.done, dtype=bool))
    true_terminals = traj_batch.done & ~is_timeout

    term_cumsum = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(true_terminals.astype(jnp.int32), axis=0)],
        axis=0,
    )
    timeout_cumsum = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(is_timeout.astype(jnp.int32), axis=0)],
        axis=0,
    )

    hit_true_terminal = (jnp.take_along_axis(term_cumsum, j, axis=0) - term_cumsum[:T]) > 0
    hit_timeout = (jnp.take_along_axis(timeout_cumsum, j, axis=0) - timeout_cumsum[:T]) > 0
    hit_horizon = (t_arr + m > T)

    # A jump is valid if it lands on an observed state (<= T) or reaches an absorbing terminal
    is_valid_jump = (t_arr + m <= T) | hit_true_terminal
    if jump_mode == "drop_invalid_jump":
        critic_mask = is_valid_jump.astype(jnp.float32)
    else:  # "mc_invalid_jump", "full_mc_invalid_jump", "mc_full_weight"
        critic_mask = jnp.ones((T, B), dtype=jnp.float32)

    # 3. Track steps since start (clearing to 10,000 on timeout)
    def scan_steps(k, step_flags):
        term, timeout = step_flags
        k_next = jnp.where(term, 0, jnp.where(timeout, 10_000, k + 1))
        return k_next, k

    init_k = jnp.full((B,), 10_000, dtype=jnp.int32) if steps_since_start is None else steps_since_start
    final_steps_since_start, all_steps = jax.lax.scan(scan_steps, init_k, (true_terminals, is_timeout))

    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    start_trace = jnp.where(
        gl < 1e-6,
        jnp.where(all_steps == 0, 1.0, 0.0),
        safe_gl ** all_steps,
    )
    start_weight = 0.25 * tilde_gamma * start_trace

    # 4. Form extended arrays (length T + 1) for destination lookups at step T
    obs_ext = jnp.concatenate([traj_batch.obs, traj_batch.next_obs[-1][None]], axis=0)
    target_ext = jnp.concatenate([returns, traj_batch.next_value[-1][None]], axis=0)

    # Gather destination states and targets along time axis (axis 0)
    j_expanded = j.reshape(j.shape + (1,) * (obs_ext.ndim - 2))
    j_expanded = jnp.broadcast_to(j_expanded, (T, B) + obs_ext.shape[2:])
    dest_obs = jnp.take_along_axis(obs_ext, j_expanded, axis=0)
    dest_target = jnp.take_along_axis(target_ext, j, axis=0)

    action = traj_batch.action if hasattr(traj_batch, "action") else None
    log_prob = traj_batch.log_prob if hasattr(traj_batch, "log_prob") else None
    adv = advantages if advantages is not None else None
    aux_target = returns if aux_returns is None else aux_returns

    jump_batch = ELambdaJumpTransition(
        obs=traj_batch.obs,
        dest_obs=dest_obs,
        target=returns,
        dest_target=dest_target,
        hit_true_terminal=hit_true_terminal,
        hit_timeout=hit_timeout,
        start_weight=start_weight,
        hit_horizon=hit_horizon,
        aux_target=aux_target,
        action=action,
        log_prob=log_prob,
        advantage=adv,
        critic_mask=critic_mask,
    )
    return jump_batch, final_steps_since_start


def e_lambda_pointwise_critic_loss(
    v_start: jax.Array,
    v_jump: jax.Array,
    v_aux_start: jax.Array,
    v_aux_jump: jax.Array,
    mb: ELambdaJumpTransition,
    gamma: float,
    lmbda: float,
    variant: str = "aux",
    jump_mode: str = "full_mc_invalid_jump",
    full_weight_mc: bool = False,
):
    """
    Evaluates the exact Dirichlet, magnitude, and start losses on a 1D minibatch of jump pairs.
    Supported variants:
    - 'aux' / 'v_aux': Debiased surrogate using auxiliary critic v_aux for the backward branch
    - 'naive' / 'oct8': Naive jump sampling (Oct 8 analog) differentiating both source and destination
    - 'fwd' / 'forward': Forward-only sampling dropping the backward gradient branch

    Supported jump_modes for invalid jumps (hit_timeout or hit_horizon t + m > T):
    - 'full_mc_invalid_jump' / 'mc_full_weight' (default, or full_weight_mc=True): sets diff_sq = e_start ** 2 so that
      (magnitude_weight + dirichlet_weight) * e_start ** 2 = 0.5 * e_start ** 2 (FULL unit-weight MC target).
    - 'mc_invalid_jump': zeroes Dirichlet smoothing (0.0), keeping standard magnitude weight (1 - tilde_gamma).
    - 'drop_invalid_jump': excluded via critic_mask.
    """
    sg = jax.lax.stop_gradient
    gl = gamma * lmbda

    e_start = sg(mb.target) - v_start
    e_jump = jnp.where(mb.hit_true_terminal, 0.0, sg(mb.dest_target) - v_jump)

    e_aux_start = sg(v_aux_start) - v_start
    e_aux_jump = jnp.where(mb.hit_true_terminal, 0.0, sg(v_aux_jump) - v_jump)

    if variant in ["naive", "oct8"]:
        inner_diff_sq = (e_start - e_jump) ** 2
    elif variant in ["fwd", "forward", "forward_e"]:
        inner_diff_sq = (e_start - sg(e_jump)) ** 2
    else:
        inner_diff_sq = (e_start - sg(e_jump)) ** 2 + (sg(e_aux_start) - e_aux_jump) ** 2

    is_horizon = mb.hit_horizon if mb.hit_horizon is not None else False
    is_invalid = mb.hit_timeout | is_horizon
    is_full_mc = full_weight_mc or (jump_mode in ["full_mc_invalid_jump", "mc_full_weight", "mc_full", "full_weight_mc", "mc_invalid_jump_full"])

    # If hit true absorbing terminal: (e_start - 0)^2 = e_start^2 (full-weight MC target).
    # If hit timeout or horizon overshoot (t + m >= T):
    #   - If is_full_mc: set diff_sq = e_start ** 2 so (magnitude + dirichlet) = 0.5 * e_start ** 2 (full-weight MC target).
    #   - Otherwise: zero out smoothing term (0.0), keeping standard magnitude weight (1 - tilde_gamma).
    invalid_diff_sq = jnp.where(is_full_mc, e_start ** 2, 0.0)
    diff_sq = jnp.where(
        mb.hit_true_terminal,
        e_start ** 2,
        jnp.where(is_invalid, invalid_diff_sq, inner_diff_sq),
    )

    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    magnitude_weight = 0.5 * (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * tilde_gamma

    if mb.critic_mask is not None:
        c_mask = mb.critic_mask.astype(jnp.float32)
        n_crit = jnp.maximum(jnp.sum(c_mask), 1.0)
        magnitude_loss = magnitude_weight * (jnp.sum((e_start ** 2) * c_mask) / n_crit)
        dirichlet_loss = dirichlet_weight * (jnp.sum(diff_sq * c_mask) / n_crit)
        start_loss = jnp.sum(mb.start_weight * (e_start ** 2) * c_mask) / n_crit
    else:
        magnitude_loss = magnitude_weight * jnp.mean(e_start ** 2)
        dirichlet_loss = dirichlet_weight * jnp.mean(diff_sq)
        start_loss = jnp.mean(mb.start_weight * (e_start ** 2))
    value_loss = magnitude_loss + dirichlet_loss + start_loss

    losses = {
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "start_loss": start_loss,
    }
    return value_loss, losses


def e_lambda_ppo_minibatch_loss(
    params,
    network,
    mb: ELambdaJumpTransition,
    config,
    aux_params=None,
    aux_network=None,
):
    """
    Combined PPO loss on a flat minibatch of jump pairs.
    """
    # 1. Actor policy loss & entropy
    pi = network.apply(params, mb.obs, method=network.policy)
    log_prob = pi.log_prob(mb.action)
    ratio = jnp.exp(log_prob - mb.log_prob)
    adv = post_process_advantage(mb.advantage, config)
    loss_actor1 = ratio * adv
    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
    entropy = pi.entropy().mean()

    # 2. Critic value predictions on (obs, dest_obs)
    v_start = network.apply(params, mb.obs, method=network.value)
    v_jump = network.apply(params, mb.dest_obs, method=network.value)

    if aux_network is not None and aux_params is not None:
        v_aux_start = aux_network.apply(aux_params, mb.obs)
        v_aux_jump = aux_network.apply(aux_params, mb.dest_obs)
    else:
        v_aux_start = v_start
        v_aux_jump = v_jump

    variant = config.get("E_LAMBDA_VARIANT", config.get("E_VARIANT", "aux"))
    jump_mode = config.get("E_LAMBDA_JUMP_MODE", "full_mc_invalid_jump")
    full_mc = config.get("E_LAMBDA_FULL_WEIGHT_MC", config.get("FULL_WEIGHT_MC_INVALID", False))
    value_loss, critic_losses = e_lambda_pointwise_critic_loss(
        v_start=v_start,
        v_jump=v_jump,
        v_aux_start=v_aux_start,
        v_aux_jump=v_aux_jump,
        mb=mb,
        gamma=config["GAMMA"],
        lmbda=config["VALUE_LAMBDA"],
        variant=variant,
        jump_mode=jump_mode,
        full_weight_mc=full_mc,
    )

    total_loss = (
        config["POLICY_COEFF"] * loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )

    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": critic_losses["magnitude_loss"],
        "dirichlet_loss": critic_losses["dirichlet_loss"],
        "start_loss": critic_losses["start_loss"],
        "actor_loss": loss_actor,
        "entropy": entropy,
    }
    return total_loss, losses


def e_lambda_fixed_critic_minibatch_loss(
    params,
    network,
    mb: ELambdaJumpTransition,
    config,
    aux_params=None,
    aux_network=None,
):
    """
    Critic-only loss on a flat minibatch of jump pairs for fixed policy evaluation.
    """
    v_start = network.apply(params, mb.obs)
    v_jump = network.apply(params, mb.dest_obs)

    if aux_network is not None and aux_params is not None:
        v_aux_start = aux_network.apply(aux_params, mb.obs)
        v_aux_jump = aux_network.apply(aux_params, mb.dest_obs)
    else:
        v_aux_start = v_start
        v_aux_jump = v_jump

    variant = config.get("E_LAMBDA_VARIANT", config.get("E_VARIANT", "aux"))
    jump_mode = config.get("E_LAMBDA_JUMP_MODE", "full_mc_invalid_jump")
    full_mc = config.get("E_LAMBDA_FULL_WEIGHT_MC", config.get("FULL_WEIGHT_MC_INVALID", False))
    value_loss, critic_losses = e_lambda_pointwise_critic_loss(
        v_start=v_start,
        v_jump=v_jump,
        v_aux_start=v_aux_start,
        v_aux_jump=v_aux_jump,
        mb=mb,
        gamma=config["GAMMA"],
        lmbda=config["VALUE_LAMBDA"],
        variant=variant,
        jump_mode=jump_mode,
        full_weight_mc=full_mc,
    )

    total_loss = config.get("VF_COEF", 1.0) * value_loss
    critic_losses["total_loss"] = total_loss
    return total_loss, critic_losses


def compute_bsmc_weights(gamma: float, lmbda: float) -> tuple[float, float, float]:
    """
    Computes exact weights (w_mc, w_dir, alpha) for the Bellman-Smoothed Monte Carlo objective:
        L_BSMC(alpha) = w_mc * L_MC + w_dir * L_smooth^(1-step)
    where:
        w_mc  = 0.5 * (1 - gamma) / (1 - gamma * lambda)
        w_dir = 0.5 * gamma * (1 - lambda) / (1 - gamma * lambda)
        Total weight W = w_mc + w_dir = 0.5
        Relative MC share: (1 - alpha) = w_mc / W = (1 - gamma) / (1 - gamma * lambda)
        Relative Smoothing share: alpha = w_dir / W = gamma * (1 - lambda) / (1 - gamma * lambda)
    """
    gl = gamma * lmbda
    denom = jnp.maximum(1.0 - gl, 1e-8)
    w_mc = 0.5 * (1.0 - gamma) / denom
    w_dir = 0.5 * (gamma * (1.0 - lmbda)) / denom
    alpha = (gamma * (1.0 - lmbda)) / denom
    return w_mc, w_dir, alpha


class BSMCTransition(NamedTuple):
    """
    Transition container for 1-step Bellman-Smoothed Monte Carlo (BSMC).
    Directly packages rollout transitions (s_t, s_{t+1}, G_t, G_{t+1}) with no runway cutoff.
    Shape before shuffle_and_batch: (T, B, ...)
    Shape inside minibatch: (minibatch_size, ...)
    """
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    target: jnp.ndarray
    next_target: jnp.ndarray
    true_terminal: jnp.ndarray
    is_timeout: jnp.ndarray
    is_start: jnp.ndarray
    aux_target: Optional[jnp.ndarray] = None
    action: Optional[jnp.ndarray] = None
    log_prob: Optional[jnp.ndarray] = None
    advantage: Optional[jnp.ndarray] = None


def bsmc_critic_loss(
    v_i: jax.Array,
    targets_i: jax.Array,
    v_j: jax.Array,
    targets_j: jax.Array,
    true_terminal: jax.Array,
    is_timeout: Optional[jax.Array] = None,
    is_start: Optional[jax.Array] = None,
    w_mc: Optional[float] = None,
    w_dir: Optional[float] = None,
    v_aux_i: Optional[jax.Array] = None,
    v_aux_j: Optional[jax.Array] = None,
    variant: str = "aux",
    gamma: float = 0.99,
):
    """
    Computes 1-step Bellman-Smoothed Monte Carlo (BSMC) critic loss based directly on
    the 1-step E(0) loss (e_critic_loss), with relative term weighting (w_mc, w_dir) as input:
        L_BSMC = w_mc * L_MC + w_dir * L_smooth^(1-step) + start_loss + timeout_loss
    where:
        w_mc  = 0.5 * (1 - gamma) / (1 - gamma * lambda)
        w_dir = 0.5 * gamma * (1 - lambda) / (1 - gamma * lambda)
    When lambda = 0: w_mc = 0.5 * (1 - gamma), w_dir = 0.5 * gamma (exact E(0) loss).
    When lambda = 1: w_mc = 0.5, w_dir = 0.0 (exact pure MC loss).
    """
    if variant in ["aux", "v_aux", "surgical", "surgical_v_aux"]:
        e_variant = "surgical_v_aux"
    elif variant in ["forward", "forward_e", "forward_only", "fwd"]:
        e_variant = "forward_e"
    elif variant in ["dirichlet", "dirichlet_v_aux"]:
        e_variant = "dirichlet_v_aux"
    else:
        e_variant = "oct8"

    return e_critic_loss(
        v_i=v_i,
        targets_i=targets_i,
        v_j=v_j,
        targets_j=targets_j,
        true_terminal=true_terminal,
        gamma=gamma,
        variant=e_variant,
        is_start=is_start,
        is_timeout=is_timeout,
        v_aux_i=v_aux_i,
        v_aux_j=v_aux_j,
        w_mc=w_mc,
        w_dir=w_dir,
        return_details=True,
    )


def bsmc_ppo_minibatch_loss(
    params,
    network,
    mb: BSMCTransition,
    config,
    aux_params=None,
    aux_network=None,
):
    """
    Combined PPO loss on a flat minibatch of 1-step BSMC transitions.
    """
    # 1. Actor policy loss & entropy
    pi = network.apply(params, mb.obs, method=network.policy)
    log_prob = pi.log_prob(mb.action)
    ratio = jnp.exp(log_prob - mb.log_prob)
    adv = post_process_advantage(mb.advantage, config)
    loss_actor1 = ratio * adv
    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
    entropy = pi.entropy().mean()

    # 2. Critic value predictions on (obs, next_obs)
    v_i = network.apply(params, mb.obs, method=network.value)
    v_j = network.apply(params, mb.next_obs, method=network.value)

    if aux_network is not None and aux_params is not None:
        v_aux_i = aux_network.apply(aux_params, mb.obs)
        v_aux_j = aux_network.apply(aux_params, mb.next_obs)
    else:
        v_aux_i = None
        v_aux_j = None

    if "BSMC_WEIGHTS" in config:
        w_mc, w_dir = config["BSMC_WEIGHTS"]
        total_w = w_mc + w_dir
        alpha = w_dir / jnp.maximum(total_w, 1e-8)
    elif "BSMC_ALPHA" in config:
        alpha = config["BSMC_ALPHA"]
        w_mc = 0.5 * (1.0 - alpha)
        w_dir = 0.5 * alpha
    else:
        w_mc, w_dir, alpha = compute_bsmc_weights(config["GAMMA"], config["VALUE_LAMBDA"])

    variant = config.get("BSMC_VARIANT", config.get("E_LAMBDA_VARIANT", "aux"))

    value_loss, critic_losses = bsmc_critic_loss(
        v_i=v_i,
        targets_i=mb.target,
        v_j=v_j,
        targets_j=mb.next_target,
        true_terminal=mb.true_terminal,
        is_timeout=mb.is_timeout,
        is_start=mb.is_start,
        w_mc=w_mc,
        w_dir=w_dir,
        v_aux_i=v_aux_i,
        v_aux_j=v_aux_j,
        variant=variant,
        gamma=config["GAMMA"],
    )

    total_loss = (
        config.get("POLICY_COEFF", 1.0) * loss_actor
        + config.get("VF_COEF", 0.5) * value_loss
        - config.get("ENT_COEF", 0.01) * entropy
    )

    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": critic_losses["magnitude_loss"],
        "dirichlet_loss": critic_losses["dirichlet_loss"],
        "start_loss": critic_losses.get("start_loss", 0.0),
        "actor_loss": loss_actor,
        "entropy": entropy,
        "mc_share": 1.0 - alpha,
        "smooth_share": alpha,
    }
    return total_loss, losses


# Re-export compute_runtime_metrics for convenience
try:
    from core.runtime_metrics import compute_runtime_metrics
except ImportError:
    pass

