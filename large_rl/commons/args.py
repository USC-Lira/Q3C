import argparse

"""=== RecSim ==="""
MIN_QUALITY_SCORE = -100  # The min quality score.
MAX_QUALITY_SCORE = 100  # The max quality score.
MAX_VIDEO_LENGTH = 10.0  # The maximum length of videos.

SKIP_TOKEN = -1
HISTORY_SIZE = 3
DATA_SPLIT = {"offline": 0.6, "online": 0.4}  # {"train": 0.6, "val": 0.05, "test": 0.4}
ML100K_NUM_ITEMS = 1682
ML100K_NUM_USERS = 943
ML100K_NUM_RATINGS = 5
ML100K_ITEM_FEATURES = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
                        'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                        'Western']
ML100K_USER_FEATURES = ['age', 'F', 'M', 'administrator', 'artist', 'doctor', 'educator', 'engineer', 'entertainment',
                        'executive', 'healthcare', 'homemaker', 'lawyer', 'librarian', 'marketing', 'none', 'other',
                        'programmer', 'retired', 'salesman', 'scientist', 'student', 'technician', 'writer']
ML100K_DIM_ITEM = len(ML100K_ITEM_FEATURES)  # 18
ML100K_DIM_USER = len(ML100K_USER_FEATURES)  # 24
USER_HISTORY_COL_NAME = "t-"
USER_HISTORY_COLS = ["t-{}".format(t + 1) for t in range(HISTORY_SIZE)]


def _get_emb_file_name(args):
    """ Called from pretraining of t-sne and the Env of main run """
    if args["env_name"].startswith("recsim"):
        return f"emb/seed{args['env_seed']}-" \
               f"item{args['num_all_actions']}-" \
               f"tsneDim{args['recsim_dim_tsne_embed']}-" \
               f"originalDim{args['recsim_dim_embed']}"


def set_if_none(args, var, value):
    vars(args)[var] = value if vars(args)[var] is None else vars(args)[var]


def set_none(args, var):
    vars(args)[var] = None


def str2bool(v):
    """ Used to convert the command line arg of bool into boolean var """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def set_mine_args(args):
    set_if_none(args, "total_ts", 10000000)
    set_if_none(args, "num_epochs", 5000)
    set_none(args, "num_all_actions")
    # set_if_none(args, "per_train_ts", 100)
    set_if_none(args, "buffer_size", 500000)
    set_if_none(args, "max_episode_steps", 100)
    set_if_none(args, "decay_steps_cr", 5000000)
    set_if_none(args, "decay_steps_act", 5000000)
    set_if_none(args, "eval_num_episodes", 80)
    set_if_none(args, "if_async", True)
    set_if_none(args, "vid_fps", 5)
    set_if_none(args, "epsilon_start_act", 1)
    set_if_none(args, "epsilon_start_cr", 1)
    set_if_none(args, "epsilon_end_act", 0.01)
    set_if_none(args, "epsilon_end_cr", 0.01)
    set_if_none(args, "min_replay_buffer_size", 5000)
    set_if_none(args, "num_updates", 20) # Not active
    set_if_none(args, "env_dim_extra", 0)
    # Changing from before:
    set_if_none(args, "eval_freq", 100)
    set_if_none(args, "num_envs", 20) # previously 5
    # Changing because porting to method side
    # set_if_none(args, "WOLP_actor_lr", 0.0001)
    # set_if_none(args, "WOLP_pairwise_distance_bonus_coef", 0.01)
    # set_if_none(args, "WOLP_cascade_list_len", 1)
    # Move to later


def set_other_args(args):
    set_if_none(args, "vid_fps", 30)
    set_if_none(args, "WOLP_cascade_list_len", 1)
    set_if_none(args, "env_dim_extra", 0)
    # Overwritten by RecSim itself
    set_if_none(args, "total_ts", 100000)
    set_if_none(args, "num_all_actions", 100)
    set_if_none(args, "num_epochs", 100)
    set_if_none(args, "buffer_size", 500000)
    set_if_none(args, "max_episode_steps", 15)
    set_if_none(args, "Qnet_dim_hidden", "64_32")
    set_if_none(args, "decay_steps_cr", 30000)
    set_if_none(args, "decay_steps_act", 30000)
    set_if_none(args, "eval_num_episodes", 20)
    set_if_none(args, "eval_freq", 1)
    set_if_none(args, "num_envs", 1)
    set_if_none(args, "if_async", True)
    set_if_none(args, "WOLP_actor_dim_hiddens", "64_32_32_16")
    set_if_none(args, "WOLP_pairwise_distance_bonus_coef", 0.25)
    set_if_none(args, "epsilon_start_act", 1)
    set_if_none(args, "epsilon_start_cr", 1)
    set_if_none(args, "epsilon_end_act", 0.01)
    set_if_none(args, "epsilon_end_cr", 0.01)
    set_if_none(args, "min_replay_buffer_size", 5000)
    set_if_none(args, "num_updates", 20)
    set_if_none(args, 'WOLP_if_0th_ref_critic', False)
    set_if_none(args, "dim_hidden", 64)
    set_if_none(args, "WOLP_noise_type", 'ou')
    set_if_none(args, "soft_update_tau", 0.001)
    set_if_none(args, "if_grad_clip", True)
    if args.env_name.lower().startswith("mujoco"):
        set_if_none(args, "TD3_target_policy_smoothing", True)
    else:
        set_if_none(args, "TD3_target_policy_smoothing", False)
    set_if_none(args, "WOLP_twin_target", True)
    set_if_none(args, 'WOLP_ar_actor_no_conditioning', True)
    set_if_none(args, 'WOLP_0th_no_perturb', False)
    set_if_none(args, "WOLP_use_main_ref_critic_for_target", False)
    set_if_none(args, "WOLP_use_main_ref_critic_for_action_selection", False)

def set_per_train_ts(args):
    # args.per_train_ts = 50  # Trying to keep it 1 update per data collected
    if args.if_train_every_ts:
        args.per_train_ts = 50 // args.num_envs  # Trying to keep it 1 update per data collected
        set_if_none(args, "eval_freq", 20 * args.num_envs)
    else:
        args.per_train_ts = 50
        set_if_none(args, "eval_freq", 20)

def set_method_based_args(args):
    if args.method_name == "flair":
        raise NotImplementedError
        # OLD
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = True
    elif args.method_name == "refresh_flair_imitate":
        '''
        # OLD
        NOTE: In this version,
        Qi(s, ai | a_(<i)) = max Q(s, aj) for all j <= i
        (No Bellman Backup)
        '''
        raise NotImplementedError
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = True
        args.WOLP_if_ar_imitate = True

        # Refresh #1: Target values to be used from imitation
        args.WOLP_refineQ_target = False
        # Refresh #2: Q on true action
        # THIS IS INACTIVE WHEN if_ar_imitate = True
        args.WOLP_Q_on_true_action = True
        # Refresh #3: This means flair gets to explore the cont. action space
        args.WOLP_if_noise_postQ = False
        # Refresh #4: Turns out exploration delays learning, at least for MineEnv
        args.WOLP_if_dual_exploration = False
        args.WOLP_total_dual_exploration = False


        # -------- FLAIR SPECIFIC : TODO --------
        # Change: Concat state before deepset
        args.WOLP_list_concat_state = True
        # Change: Max pooling
        args.WOLP_ar_list_encoder_deepset_maxpool = True
        # TODO: Trying to fix this
        # TODO: THIS IS A HACK!!
        args.WOLP_ar_use_query_max = False
        args.WOLP_ar_knn_action_rep = True
        # ----------------------------------------

        args.WOLP_discounted_cascading = False
        args.WOLP_t0_no_list_input = True
        args.TwinQ = True
        args.sync_freq = 2
        args.TD3_policy_delay = 2
        args.WOLP_noise_type = 'ou'
        args.DEBUG_type_clamp = 'large'
        args.soft_update_tau = 0.005
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        args.if_train_every_ts = True
        args.TD3_target_policy_smoothing = True
        args.if_grad_clip = True
        args.WOLP_if_0th_ref_critic = True

    elif args.method_name == "flair_inside":
        '''
        Updated: 2023-08-28
        NOTE: In this version,
        Qi(s, ai | a_(<i)) = max Q(s, aj) for all j <= i
        (No Bellman Backup)
        '''
        # ---- Setting up the method ----
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = True
        args.WOLP_if_ar_imitate = True
        set_if_none(args, "WOLP_twin_target", True)
        # ------------------------------

        # ---- Exploration Specific ----
        if args.env_name.lower().startswith("mujoco"):
            # TODO: Likely this should be False to help the actors explore
            set_if_none(args, "WOLP_if_noise_postQ", True)
            # args.WOLP_if_noise_postQ = True
        else:
            # TODO: Check if we still want to do this depending on mujoco results
            set_if_none(args, "WOLP_if_noise_postQ", False)
            # args.WOLP_if_noise_postQ = False # This means the agent gets to explore the cont. action space
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        # Turns out critic exploration delays learning, at least for MineEnv
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = False # Target values to be used from imitation
        args.WOLP_Q_on_true_action = False # THIS IS INACTIVE WHEN if_ar_imitate = True
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        args.WOLP_policy_loss_mean = False
        args.WOLP_separate_update = False
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        # ------------------------------

        # -------- FLAIR SPECIFIC --------
        args.WOLP_t0_no_list_input = True
        set_if_none(args, "WOLP_list_concat_state", True)
        # args.WOLP_list_concat_state = True
        args.WOLP_ar_list_encoder_deepset_maxpool = True
        args.WOLP_ar_use_query_max = False
        args.WOLP_ar_knn_action_rep = True
        # TODO: Set this to True for continuous action space and False for discrete?
        # Or maybe even for discrete, set all Q_k functions to act on the true action?
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, 'WOLP_if_0th_ref_critic', True)
        else:
            set_if_none(args, 'WOLP_if_0th_ref_critic', False)
        set_if_none(args, "WOLP_knn_inside_cascade", False)
        # ----------------------------------------


    elif args.method_name == "savo":
        # ---- Setting up the method ----
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = True
        args.WOLP_if_ar_imitate = True
        set_if_none(args, "WOLP_twin_target", True)
        # ------------------------------

        # ---- Exploration Specific ----
        set_if_none(args, "WOLP_if_noise_postQ", False)
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        # Turns out critic exploration delays learning, at least for MineEnv
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = False # Target values to be used from imitation
        args.WOLP_Q_on_true_action = False # THIS IS INACTIVE WHEN if_ar_imitate = True
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        args.WOLP_policy_loss_mean = False
        args.WOLP_separate_update = False
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        # ------------------------------

        # -------- FLAIR SPECIFIC --------
        args.WOLP_t0_no_list_input = True
        set_if_none(args, "WOLP_list_concat_state", True)
        args.WOLP_ar_list_encoder_deepset_maxpool = True
        args.WOLP_ar_use_query_max = False
        args.WOLP_ar_knn_action_rep = True
        # TODO: Set this to True for continuous action space and False for discrete?
        # Or maybe even for discrete, set all Q_k functions to act on the true action?
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, 'WOLP_if_0th_ref_critic', True)
            set_if_none(args, 'WOLP_if_ar_noise_before_cascade', False)
        else:
            set_if_none(args, 'WOLP_if_0th_ref_critic', False)
            set_if_none(args, 'WOLP_if_ar_noise_before_cascade', True)
        set_if_none(args, "WOLP_knn_inside_cascade", False)
        set_if_none(args, 'WOLP_ar_actor_no_conditioning', True)
        set_if_none(args, 'WOLP_ar_critic_taken_action_update', True)

        # ----------------------------------------


    elif args.method_name == "savo_refined":
        # ---- Setting up the method ----
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = True
        args.WOLP_if_ar_imitate = True
        set_if_none(args, "WOLP_twin_target", True)
        # ------------------------------

        # ---- Exploration Specific ----
        set_if_none(args, "WOLP_if_noise_postQ", False)
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        # Turns out critic exploration delays learning, at least for MineEnv
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = False # Target values to be used from imitation
        args.WOLP_Q_on_true_action = False # THIS IS INACTIVE WHEN if_ar_imitate = True
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        args.WOLP_policy_loss_mean = False
        args.WOLP_separate_update = False
        set_if_none(args, "WOLP_use_main_ref_critic_for_action_selection", True) # NOTE: Changed from False
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        # ------------------------------

        # -------- FLAIR SPECIFIC --------
        args.WOLP_t0_no_list_input = True
        set_if_none(args, "WOLP_list_concat_state", True)
        set_if_none(args, "WOLP_ar_type_list_encoder", "non-shared-deepset") # TODO: Try with LSTM
        set_if_none(args, "WOLP_use_main_ref_critic_for_target", True) # Changed from False
        args.WOLP_ar_list_encoder_deepset_maxpool = True
        args.WOLP_ar_use_query_max = False
        args.WOLP_ar_knn_action_rep = True
        # TODO: Set this to True for continuous action space and False for discrete?
        # Or maybe even for discrete, set all Q_k functions to act on the true action?
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, 'WOLP_if_0th_ref_critic', True)
            set_if_none(args, 'WOLP_if_ar_noise_before_cascade', False)
            set_if_none(args, "WOLP_0th_no_perturb", False) # NOTE: Added New!
        else:
            set_if_none(args, 'WOLP_if_0th_ref_critic', False)
            set_if_none(args, 'WOLP_if_ar_noise_before_cascade', True)
            set_if_none(args, "WOLP_0th_no_perturb", False) # NOTE: Added New!
        set_if_none(args, "WOLP_knn_inside_cascade", False)
        set_if_none(args, 'WOLP_ar_actor_no_conditioning', True)
        set_if_none(args, 'WOLP_ar_critic_taken_action_update', True)

        # ----------------------------------------

    elif args.method_name == "savo_threshold_refined":
        # ---- Setting up the method ----
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = True
        args.WOLP_if_ar_imitate = True
        set_if_none(args, "WOLP_twin_target", True)
        args.WOLP_if_ar_critic_share_weight = True
        args.WOLP_if_0th_ref_critic = True
        args.WOLP_no_ar_critics = True
        args.WOLP_threshold_Q = True
        args.WOLP_threshold_Q_direct_cummax = False
        if not args.env_name.lower().startswith("mujoco"):
            # Needed to resolve the issue of evaluation of unknown actions. extra critic learns the k-NN.
            args.WOLP_if_dual_critic=True
            args.WOLP_if_dual_critic_imitate=True
        # ------------------------------

        # ---- Exploration Specific ----
        set_if_none(args, "WOLP_if_noise_postQ", False)
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        # Turns out critic exploration delays learning, at least for MineEnv
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = False # Target values to be used from imitation
        args.WOLP_Q_on_true_action = False # THIS IS INACTIVE WHEN if_ar_imitate = True
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        args.WOLP_policy_loss_mean = False
        args.WOLP_separate_update = False
        set_if_none(args, "WOLP_use_main_ref_critic_for_action_selection", True) # NOTE: Changed from False
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        # ------------------------------

        # -------- FLAIR SPECIFIC --------
        args.WOLP_t0_no_list_input = True
        set_if_none(args, "WOLP_list_concat_state", True)
        set_if_none(args, "WOLP_use_main_ref_critic_for_target", True) # Changed from False
        args.WOLP_ar_list_encoder_deepset_maxpool = True
        args.WOLP_ar_use_query_max = False
        args.WOLP_ar_knn_action_rep = True
        # TODO: Set this to True for continuous action space and False for discrete?
        # Or maybe even for discrete, set all Q_k functions to act on the true action?
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, 'WOLP_if_0th_ref_critic', True)
            set_if_none(args, 'WOLP_if_ar_noise_before_cascade', False)
            set_if_none(args, "WOLP_0th_no_perturb", False) # NOTE: Added New!
        else:
            set_if_none(args, 'WOLP_if_0th_ref_critic', False)
            set_if_none(args, 'WOLP_if_ar_noise_before_cascade', True)
            set_if_none(args, "WOLP_0th_no_perturb", False) # NOTE: Added New!
        set_if_none(args, "WOLP_knn_inside_cascade", False)
        set_if_none(args, 'WOLP_ar_actor_no_conditioning', True)
        set_if_none(args, 'WOLP_ar_critic_taken_action_update', True)

    elif args.method_name == "flair_mujoco":
        '''
        Updated: 2023-08-28
        NOTE: In this version,
        Qi(s, ai | a_(<i)) = max Q(s, aj) for all j <= i
        (No Bellman Backup)
        '''
        # ---- Setting up the method ----
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = True
        args.WOLP_if_ar_imitate = True
        set_if_none(args, "WOLP_twin_target", True)
        # ------------------------------

        # --- mujoco-env based hyperparameter search ---- #
        args.WOLP_if_ar_noise_before_cascade = False
        args.WOLP_ar_actor_no_conditioning = True
        args.WOLP_ar_critic_taken_action_update = True

        # ---- Exploration Specific ----
        # args.WOLP_if_noise_postQ = False
        set_if_none(args, "WOLP_if_noise_postQ", False)
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        # Turns out critic exploration delays learning, at least for MineEnv
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = False # Target values to be used from imitation
        args.WOLP_Q_on_true_action = False # THIS IS INACTIVE WHEN if_ar_imitate = True
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        args.WOLP_policy_loss_mean = False
        args.WOLP_separate_update = False
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        # ------------------------------

        # -------- FLAIR SPECIFIC --------
        args.WOLP_t0_no_list_input = True
        set_if_none(args, "WOLP_list_concat_state", True)
        # args.WOLP_list_concat_state = True
        args.WOLP_ar_list_encoder_deepset_maxpool = True
        args.WOLP_ar_use_query_max = False
        args.WOLP_ar_knn_action_rep = True
        # TODO: Set this to True for continuous action space and False for discrete?
        # Or maybe even for discrete, set all Q_k functions to act on the true action?
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, 'WOLP_if_0th_ref_critic', True)
        else:
            set_if_none(args, 'WOLP_if_0th_ref_critic', False)
        set_if_none(args, "WOLP_knn_inside_cascade", False)
        # ----------------------------------------


    elif args.method_name == "wolp_dual":
        # ---- Setting up the method ----
        args.WOLP_cascade_list_len = 1
        args.agent_type = "wolp"
        args.WOLP_if_ar = False
        args.WOLP_if_ar_actor_cascade = False
        args.WOLP_if_ar_critic_cascade = False
        args.WOLP_if_ar_cascade_list_enc = False
        set_if_none(args, "WOLP_topK", 3)
        args.WOLP_if_dual_critic = True
        args.WOLP_if_dual_critic_imitate = True
        # Add this to make WOLP_dual work better with k > 1, but we need to correct this to still use TwinQ and only compute the max using the target_Q.
        set_if_none(args, "WOLP_twin_target", True)
        # ------------------------------

        # ---- Exploration Specific ----
        args.WOLP_if_noise_postQ = False
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = True # NOTE: only active if dual_critic_imitate=False
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4

    elif args.method_name == "wolp":
        # ---- Setting up the method ----
        args.WOLP_cascade_list_len = 1
        args.agent_type = "wolp"
        args.WOLP_if_ar = False
        args.WOLP_if_ar_actor_cascade = False
        args.WOLP_if_ar_critic_cascade = False
        args.WOLP_if_ar_cascade_list_enc = False
        set_if_none(args, "WOLP_topK", 3)
        args.WOLP_if_dual_critic = False
        # ------------------------------

        # ---- Exploration Specific ----
        args.WOLP_if_noise_postQ = False
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = False # NOTE: Inconsequential
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4

    elif args.method_name == "ddpg":
        args.agent_type = "ddpg"
        args.WOLP_if_ar = False
        args.WOLP_if_ar_actor_cascade = False
        args.WOLP_if_ar_critic_cascade = False
        args.WOLP_if_ar_cascade_list_enc = False
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.DEBUG_type_clamp = 'large'
        # TODO: Check the cause of difference result on hopper-box for ddpg v/s Ours-len1
    elif args.method_name == "flair_joint":

        # ---- Setting up the method ----
        args.agent_type = "wolp"
        args.WOLP_if_joint_actor = True
        args.WOLP_if_joint_critic = True
        args.WOLP_if_ar_imitate = True
        # ------------------------------

        # ---- Exploration Specific ----
        args.WOLP_if_noise_postQ = False # This means the agent gets to explore the cont. action space
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        # Turns out critic exploration delays learning, at least for MineEnv
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = False # Target values to be used from imitation
        args.WOLP_Q_on_true_action = False # THIS IS INACTIVE WHEN if_ar_imitate = True
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        args.WOLP_policy_loss_mean = False
        args.WOLP_separate_update = False
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        # ------------------------------

    elif args.method_name == "flair_no_ar_critics":
        # Note: This is based on flair_final, but with no AR critics.
        '''
        NOTE: In this version, for all i
        Qi(s, a*| a_(<i)) = R(s, a*) + gamma * Q(s', Agent(s'))
        (No Bellman Backup)
        '''
        raise NotImplementedError
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = True

        args.WOLP_no_ar_critics = True

        # Change #1: Set the target value of all networks to be the same as refineQ's.
        args.WOLP_refineQ_target = True
        # Change #2:
        args.WOLP_discounted_cascading = False
        # Change #3: Q on true action
        # Qi(s, a | a(<i)) <- R(s, a) + gamma * Q' and not ai.
        args.WOLP_Q_on_true_action = True
        # Change #4:
        args.WOLP_if_noise_postQ = True
        # Change #5:
        args.WOLP_t0_no_list_input = True
        # Change #6: Twin Q-network
        args.TwinQ = True
        # Change #7: Policy Delay and Sync Frequency = 2
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        # Change #8: OU
        args.WOLP_noise_type = 'ou'
        args.DEBUG_type_clamp = 'large'
        args.soft_update_tau = 0.005
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        # Change #9: Yes gradient clipping
        args.if_grad_clip = True
        # Change: 0th reference critic
        args.WOLP_if_0th_ref_critic = True
        # # Change: Concat state before deepset
        # args.WOLP_list_concat_state = True
        # # Change: Max pooling
        # args.WOLP_ar_list_encoder_deepset_maxpool = True
    elif args.method_name == "flair_no_cascade":
        raise NotImplementedError
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = False
    elif args.method_name == "savo_threshold":
        # ---- Setting up the method ----
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = True
        args.WOLP_if_ar_imitate = True
        set_if_none(args, "WOLP_twin_target", True)
        args.WOLP_if_ar_critic_share_weight = True
        args.WOLP_if_0th_ref_critic = True
        args.WOLP_no_ar_critics = True
        args.WOLP_threshold_Q = True
        args.WOLP_threshold_Q_direct_cummax = False
        # ------------------------------

        # ---- Exploration Specific ----
        set_if_none(args, "WOLP_if_noise_postQ", False)
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        # Turns out critic exploration delays learning, at least for MineEnv
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = False # Target values to be used from imitation
        args.WOLP_Q_on_true_action = False # THIS IS INACTIVE WHEN if_ar_imitate = True
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        args.WOLP_policy_loss_mean = False
        args.WOLP_separate_update = False
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        # ------------------------------

        # -------- FLAIR SPECIFIC --------
        args.WOLP_t0_no_list_input = True
        set_if_none(args, "WOLP_list_concat_state", True)
        args.WOLP_ar_list_encoder_deepset_maxpool = True
        args.WOLP_ar_use_query_max = False
        args.WOLP_ar_knn_action_rep = True
        # TODO: Set this to True for continuous action space and False for discrete?
        # Or maybe even for discrete, set all Q_k functions to act on the true action?
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, 'WOLP_if_0th_ref_critic', True)
            set_if_none(args, 'WOLP_if_ar_noise_before_cascade', False)
        else:
            set_if_none(args, 'WOLP_if_0th_ref_critic', False)
            set_if_none(args, 'WOLP_if_ar_noise_before_cascade', True)
        set_if_none(args, "WOLP_knn_inside_cascade", False)
        set_if_none(args, 'WOLP_ar_actor_no_conditioning', True)
        set_if_none(args, 'WOLP_ar_critic_taken_action_update', True)

        # ----------------------------------------
    elif args.method_name == "ensemble":
        # ---- Setting up the method ----
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = False
        args.WOLP_if_ar_imitate = True
        set_if_none(args, "WOLP_twin_target", True)
        args.WOLP_if_ar_critic_share_weight = True
        args.WOLP_if_0th_ref_critic = True
        args.WOLP_no_ar_critics = True
        # ------------------------------

        # ---- Exploration Specific ----
        set_if_none(args, "WOLP_if_noise_postQ", False)
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        # Turns out critic exploration delays learning, at least for MineEnv
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = False # Target values to be used from imitation
        args.WOLP_Q_on_true_action = False # THIS IS INACTIVE WHEN if_ar_imitate = True
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        args.WOLP_policy_loss_mean = False
        args.WOLP_separate_update = False
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        # ------------------------------

        # -------- FLAIR SPECIFIC --------
        args.WOLP_t0_no_list_input = True
        set_if_none(args, "WOLP_list_concat_state", True)
        args.WOLP_ar_list_encoder_deepset_maxpool = True
        args.WOLP_ar_use_query_max = False
        args.WOLP_ar_knn_action_rep = True
        # TODO: Set this to True for continuous action space and False for discrete?
        # Or maybe even for discrete, set all Q_k functions to act on the true action?
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, 'WOLP_if_0th_ref_critic', True)
            set_if_none(args, 'WOLP_if_ar_noise_before_cascade', False)
        else:
            set_if_none(args, 'WOLP_if_0th_ref_critic', False)
            set_if_none(args, 'WOLP_if_ar_noise_before_cascade', True)
        set_if_none(args, "WOLP_knn_inside_cascade", False)
        set_if_none(args, 'WOLP_ar_actor_no_conditioning', True)
        set_if_none(args, 'WOLP_ar_critic_taken_action_update', True)

        # ----------------------------------------

    elif args.method_name == "savo_refined_no_linkage":
        # ---- Setting up the method ----
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = False
        args.WOLP_if_ar_imitate = True
        set_if_none(args, "WOLP_twin_target", True)
        # ------------------------------

        # ---- Exploration Specific ----
        set_if_none(args, "WOLP_if_noise_postQ", False)
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        # Turns out critic exploration delays learning, at least for MineEnv
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = False # Target values to be used from imitation
        args.WOLP_Q_on_true_action = False # THIS IS INACTIVE WHEN if_ar_imitate = True
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        args.WOLP_policy_loss_mean = False
        args.WOLP_separate_update = False
        set_if_none(args, "WOLP_use_main_ref_critic_for_action_selection", True) # NOTE: Changed from False
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        # ------------------------------

        # -------- FLAIR SPECIFIC --------
        args.WOLP_t0_no_list_input = True
        set_if_none(args, "WOLP_list_concat_state", True)
        set_if_none(args, "WOLP_ar_type_list_encoder", "non-shared-deepset") # TODO: Try with LSTM
        set_if_none(args, "WOLP_use_main_ref_critic_for_target", True) # Changed from False
        args.WOLP_ar_list_encoder_deepset_maxpool = True
        args.WOLP_ar_use_query_max = False
        args.WOLP_ar_knn_action_rep = True
        # TODO: Set this to True for continuous action space and False for discrete?
        # Or maybe even for discrete, set all Q_k functions to act on the true action?
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, 'WOLP_if_0th_ref_critic', True)
            set_if_none(args, 'WOLP_if_ar_noise_before_cascade', False)
            set_if_none(args, "WOLP_0th_no_perturb", False) # NOTE: Added New!
        else:
            set_if_none(args, 'WOLP_if_0th_ref_critic', False)
            set_if_none(args, 'WOLP_if_ar_noise_before_cascade', True)
            set_if_none(args, "WOLP_0th_no_perturb", False) # NOTE: Added New!
        set_if_none(args, "WOLP_knn_inside_cascade", False)
        set_if_none(args, 'WOLP_ar_actor_no_conditioning', True)
        set_if_none(args, 'WOLP_ar_critic_taken_action_update', True)

        # ----------------------------------------

    elif args.method_name == "flair_no_linkage":
        '''
        Updated: 2023-08-28
        NOTE: In this version,
        Qi(s, ai | a_(<i)) = max Q(s, aj) for all j <= i
        (No Bellman Backup)
        '''
        # ---- Setting up the method ----
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = False
        args.WOLP_if_ar_imitate = True
        set_if_none(args, "WOLP_twin_target", True)
        # ------------------------------

        # ---- Exploration Specific ----
        if args.env_name.lower().startswith("mujoco"):
            # TODO: Likely this should be False to help the actors explore
            set_if_none(args, "WOLP_if_noise_postQ", True)
            # args.WOLP_if_noise_postQ = True
        else:
            # TODO: Check if we still want to do this depending on mujoco results
            set_if_none(args, "WOLP_if_noise_postQ", False)
            # args.WOLP_if_noise_postQ = False # This means the agent gets to explore the cont. action space
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        # Turns out critic exploration delays learning, at least for MineEnv
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = False # Target values to be used from imitation
        args.WOLP_Q_on_true_action = False # THIS IS INACTIVE WHEN if_ar_imitate = True
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        args.WOLP_policy_loss_mean = False
        args.WOLP_separate_update = False
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        # ------------------------------

        # -------- FLAIR SPECIFIC --------
        args.WOLP_t0_no_list_input = True
        set_if_none(args, "WOLP_list_concat_state", True)
        # args.WOLP_list_concat_state = True
        args.WOLP_ar_list_encoder_deepset_maxpool = True
        args.WOLP_ar_use_query_max = False
        args.WOLP_ar_knn_action_rep = True
        # TODO: Set this to True for continuous action space and False for discrete?
        # Or maybe even for discrete, set all Q_k functions to act on the true action?
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, 'WOLP_if_0th_ref_critic', True)
        else:
            set_if_none(args, 'WOLP_if_0th_ref_critic', False)
        set_if_none(args, "WOLP_knn_inside_cascade", False)
        # ----------------------------------------
    elif args.method_name == "flair_no_linkage_old":
        # TODO: Fix no linkage baseline to work with the new code of flair_inside (imitate)
        raise NotImplementedError
        args.agent_type = "wolp"
        args.WOLP_if_ar = True
        args.WOLP_if_ar_actor_cascade = True
        args.WOLP_if_ar_critic_cascade = True
        args.WOLP_if_ar_cascade_list_enc = False

        # NOTE: The following is taken over from flair_final. If that changes, then this needs to change as well.
        # Change #1: Set the target value of all networks to be the same as refineQ's.
        args.WOLP_refineQ_target = True
        # Change #2:
        args.WOLP_discounted_cascading = False
        # Change #3: Q on true action
        # Qi(s, a | a(<i)) <- R(s, a) + gamma * Q' and not ai.
        args.WOLP_Q_on_true_action = True
        # Change #4:
        args.WOLP_if_noise_postQ = True
        # Change #5:
        args.WOLP_t0_no_list_input = True
        # Change #6: Twin Q-network
        args.TwinQ = True
        # Change #7: Policy Delay and Sync Frequency = 2
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        # Change #8: OU
        args.WOLP_noise_type = 'ou'
        args.DEBUG_type_clamp = 'large'
        args.soft_update_tau = 0.005
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        # Change #9: Yes gradient clipping
        args.if_grad_clip = True
        # Change: 0th reference critic
        args.WOLP_if_0th_ref_critic = True

        # # Change: Concat state before deepset
        # args.WOLP_list_concat_state = True
        # # Change: Max pooling
        # args.WOLP_ar_list_encoder_deepset_maxpool = True

    elif args.method_name == "cem":
        args.agent_type = "wolp"
        args.WOLP_if_cem_actor = True
        args.WOLP_cascade_list_len = 1
        args.WOLP_if_ar = False

    elif args.method_name == "greedy_ac":
        # ---- Setting up the method ----
        args.WOLP_cascade_list_len = 1
        args.agent_type = "greedy_ac"
        args.WOLP_if_ar = False
        args.WOLP_if_ar_actor_cascade = False
        args.WOLP_if_ar_critic_cascade = False
        args.WOLP_if_ar_cascade_list_enc = False
        set_if_none(args, "WOLP_topK", 1)
        args.WOLP_if_dual_critic = False

        set_if_none(args, "greedy_N", 30)
        set_if_none(args, "greedy_Nrho", 3)
        # ------------------------------

        # ---- Exploration Specific ----
        args.WOLP_if_noise_postQ = False
        set_if_none(args, "WOLP_noise_type", 'ou')
        args.DEBUG_type_clamp = 'large'
        set_if_none(args, "WOLP_if_dual_exploration", False)
        set_if_none(args, "WOLP_total_dual_exploration", False)
        # ------------------------------

        # ---- Update Specific ----
        args.WOLP_refineQ_target = False # NOTE: Inconsequential
        args.WOLP_discounted_cascading = False
        args.TwinQ = True
        args.TD3_policy_delay = 2
        args.sync_freq = 2
        set_if_none(args, "soft_update_tau", 0.005)
        args.if_train_every_ts = True
        if args.env_name.lower().startswith("mujoco"):
            set_if_none(args, "TD3_target_policy_smoothing", True)
        else:
            set_if_none(args, "TD3_target_policy_smoothing", False)
        set_if_none(args, "if_grad_clip", True)
        # ------------------------------

        # ---- Network Specific ----
        set_if_none(args, "Qnet_dim_hidden", "256_256")
        set_if_none(args, "WOLP_actor_dim_hiddens", "256_256")  # Other envs have "64_32_32_16"
        args.WOLP_if_actor_norm_each = True
        args.WOLP_if_actor_norm_final = True
        args.WOLP_if_critic_norm_each = False
        args.WOLP_if_critic_norm_final = False
        # Learning rates
        args.WOLP_actor_lr = 3e-4
        args.WOLP_critic_lr = 3e-4
        args.WOLP_list_enc_lr = 3e-4
        args.lr = 3e-4
    elif args.method_name == "dqn":
        args.agent_type = "dqn"
    elif args.method_name == 'sac':
        args.agent_type = "sac"
    elif args.method_name in ["td3", "custom_td3"]:
        pass
    else:
        raise NotImplementedError

    if args.run_setup == "debug-cpu":
        args.reacher_save_video = False
        args.device = "cpu"
        args.wand = False
        args.if_async = False
        args.num_envs = 2
        args.prefix = "debug"
        args.min_replay_buffer_size = 0
    if args.run_setup == "debug-gpu":
        args.reacher_save_video = False
        args.device = "cuda"
        args.wand = False
        args.if_async = False
        args.num_envs = 2
        args.prefix = "debug"
        args.min_replay_buffer_size = 0
    elif args.run_setup == "exp-cpu":
        # NOTE: Now we don't need to generate video for all the runs - to make execution faster
        args.reacher_save_video = True
        args.device = "cpu"
        args.wand = True
    elif args.run_setup == "exp":
        # NOTE: Now we don't need to generate video for all the runs - to make execution faster
        args.reacher_save_video = True
        args.device = "cuda"
        args.wand = True
        if args.env_name.lower().startswith("mujoco"):
            args.do_naive_eval = True
    elif args.run_setup == "exp-no-video":
        args.reacher_save_video = False
        args.device = "cuda"
        args.wand = True
        if args.env_name.lower().startswith("mujoco"):
            args.do_naive_eval = True
    args.WOLP_if_ar_detach_list_action = True  # Try if this should be applied to all environments
    # import ipdb;ipdb.set_trace()
    set_if_none(args, "WOLP_if_ar_noise_before_cascade", True)
    # args.WOLP_if_ar_noise_before_cascade = True  # Check its behavior for various agents.
    set_if_none(args, "WOLP_ar_type_list_encoder", "non-shared-deepset")
    # args.WOLP_ar_type_list_encoder = "non-shared-deepset"
    set_if_none(args, "WOLP_if_film_listwise", True)

def set_mujoco_args(args):
    set_method_based_args(args)
    args.obs_enc_apply = False
    set_if_none(args, "num_envs", 10)

    # args.total_ts = 8000000
    if args.env_name.lower() == 'mujoco-reacher':
        set_if_none(args, "max_episode_steps", 50)  # Note: this needs to change for different environments
    elif args.env_name.lower() == 'mujoco-pusher':
        set_if_none(args, "max_episode_steps", 100)  # Note: this needs to change for different environments
    elif args.env_name.lower() in ['mujoco-ant',
                                   'mujoco-half_cheetah',
                                   'mujoco-hopper',
                                   'mujoco-humanoidstandup',
                                   'mujoco-inverted_double_pendulum',
                                   'mujoco-inverted_pendulum',
                                   'mujoco-swimmer',
                                   'mujoco-walker2d',
                                   'mujoco-humanoid']:
        set_if_none(args, "max_episode_steps", 1000)  # Note: this needs to change for different environments
    else:
        raise NotImplementedError

    if args.reacher_validity_type == 'box':
        set_if_none(args, "min_replay_buffer_size", max(10000, args.max_episode_steps * args.num_envs))
        if args.env_name.lower() in ['mujoco-pusher',
                                     'mujoco-half_cheetah','mujoco-ant']:
            set_if_none(args, "total_ts", 3_000_000)
        elif args.env_name.lower() in [
                                    'mujoco-reacher',
                                    'mujoco-hopper',
                                    'mujoco-inverted_double_pendulum',
                                    'mujoco-inverted_pendulum',
                                    'mujoco-swimmer',
                                    'mujoco-walker2d']:
            set_if_none(args, "total_ts", 2_000_000)
        elif args.env_name.lower() in [
                                   'mujoco-humanoidstandup',
                                   'mujoco-humanoid']:
            set_if_none(args, "total_ts", 5_000_000)
    elif args.reacher_validity_type == 'none':
        if args.env_name.lower() in ['mujoco-pusher',
                                     'mujoco-half_cheetah','mujoco-ant']:
            set_if_none(args, "total_ts", 3_000_000)
            set_if_none(args, "min_replay_buffer_size", max(10000, args.max_episode_steps * args.num_envs))
        elif args.env_name.lower() in [
                                    'mujoco-reacher',
                                    'mujoco-hopper',
                                    'mujoco-inverted_double_pendulum',
                                    'mujoco-inverted_pendulum',
                                    'mujoco-swimmer',
                                    'mujoco-walker2d']:
            set_if_none(args, "total_ts", 1_000_000)
            set_if_none(args, "min_replay_buffer_size", max(1000, args.max_episode_steps * args.num_envs))
        elif args.env_name.lower() in [
                                   'mujoco-humanoidstandup',
                                   'mujoco-humanoid']:
            set_if_none(args, "total_ts", 5_000_000)
            set_if_none(args, "min_replay_buffer_size", max(10000, args.max_episode_steps * args.num_envs))
        else:
            raise NotImplementedError

    # Set hidden_dim based on the action space of the environment.
    if args.env_name.lower() in ['mujoco-reacher',
                                 'mujoco-hopper',
                                 'mujoco-inverted_double_pendulum',
                                 'mujoco-inverted_pendulum',
                                 'mujoco-swimmer',]:
        set_if_none(args, "dim_hidden", 64)
    if args.env_name.lower() in ['mujoco-ant',
                                 'mujoco-half_cheetah',
                                 'mujoco-humanoidstandup',
                                 'mujoco-humanoid',
                                 'mujoco-walker2d',
                                 'mujoco-pusher']:
        set_if_none(args, "dim_hidden", 256)

    set_if_none(args, "num_epochs", 8000)  # Previously 4k

    set_if_none(args, "WOLP_actor_lr", 0.001)
    set_if_none(args, "WOLP_dual_exp_if_ignore", False)
    if args.WOLP_dual_exp_if_ignore and args.agent_type != "ddpg":
        set_if_none(args, "epsilon_start_cr", 0.3)
        set_if_none(args, "epsilon_end_cr", 0.01)
    else:
        set_if_none(args, "epsilon_start_cr", 1)
        set_if_none(args, "epsilon_end_cr", 0.01)
    set_if_none(args, "epsilon_start_act", 1)
    set_if_none(args, "epsilon_end_act", 0.01)
    set_if_none(args, "eval_num_episodes", 50)
    set_if_none(args, "decay_steps_cr", 500_000)  # Previously 250k
    set_if_none(args, "decay_steps_act", 500_000)  # Previously 250k
    set_if_none(args, "num_updates", 50) # This is only active for the case of "if_train_every_ts"=False
    set_if_none(args, "Qnet_dim_hidden", "400_300")
    set_if_none(args, "WOLP_actor_dim_hiddens", "64_64")  # Other envs have "64_32_32_16"
    set_if_none(args, "env_dim_extra", 0)
    # args.per_train_ts = 50  # Trying to keep it 1 update per data collected

    set_per_train_ts(args)
    args.video_save_frequency = args.eval_freq * 2 # Only save videos every 2 evals


    args.reacher_bijective_dims = 5

    # import ipdb;ipdb.set_trace()
    # TODO: Give negative reward for finding an invalid action in box environments.
    # if args.method_name not in ["new_flair_postQ_t0_twinQ_delay_td3params",
    #                             "flair-full_td3-ou-smooth-no_layer"]:
    #     args.WOLP_if_actor_norm_each = True  # Try without norm!
    #     args.WOLP_if_actor_norm_final = True  # Try without norm!

    if args.agent_type == "arddpg_cont" and args.WOLP_if_cem_actor:  # CEM
        args.epsilon_start_cr = 1
        set_if_none(args, "WOLP_cascade_list_len", 1)
    elif (args.agent_type == "arddpg_cont" or args.agent_type == "wolp") and \
            args.WOLP_if_ar:  # FLAIR
        # args.eval_epsilon_ac = 0.01
        set_if_none(args, "WOLP_cascade_list_len", 3)
    elif args.method_name in ["flair",
                              "new_flair",
                              "new_flair_postQ",
                              "new_flair_postQ_t0",
                              "flair_imitate",
                              "flair_final",
                              "flair_concat",
                              "flair_joint",
                              "flair_joint_imitate",
                              "flair_no_cascade",
                              "flair_no_linkage",
                              "flair_no_ar_critics",
                              "flair_inside",
                              "flair_mujoco",
                              "savo",
                              "ensemble"]:
        set_if_none(args, "WOLP_cascade_list_len", 3)
    else:
        set_if_none(args, "WOLP_cascade_list_len", 1)


def get_args(ps=None):
    if ps is None:
        ps = argparse.ArgumentParser()

    # Debug
    ps.add_argument('--if_debug', type=str2bool, default=False, help="")

    # W&B
    ps.add_argument('--wand', type=str2bool, default=False, help="whether to use W&B")
    ps.add_argument('--logging', type=str2bool, default=False, help="whether to print any logs")
    ps.add_argument('--prefix', type=str, default="test", help="Assign a prefix to label experiments on W&B")
    ps.add_argument('--group_name', type=str, default=None, help="Assign a prefix to label experiments on W&B")
    ps.add_argument('--save_dir', type=str, default="", help="")
    ps.add_argument('--wandb_dir', type=str, default="/tmp/wandb", help="")

    # Training Procedure
    ps.add_argument('--seed', type=int, default=2022, help="random seed")
    ps.add_argument('--env_seed', type=int, default=2022, help="random seed")
    ps.add_argument('--env_name', type=str, default="reacher", help="random seed")
    ps.add_argument('--agent_type', type=str, default="dqn", help="agent type")
    ps.add_argument('--method_name', type=str, default="flair", help="name of our method to simplify args")
    ps.add_argument('--run_setup', type=str, default="exp", help="whether it is debug or experiment setup")
    ps.add_argument('--total_ts', type=int, default=None, help="num of timesteps")

    ps.add_argument('--num_epochs', type=int, default=None, help="num of epochs")
    ps.add_argument('--per_train_ts', type=int, default=None,
                    help="Number of steps to train in each epoch. Specifying this overwrites the num_epochs parameter")
    ps.add_argument('--batch_size', type=int, default=256, help='replay buffer size')
    ps.add_argument('--buffer_size', type=int, default=None, help='replay buffer size')
    ps.add_argument('--min_replay_buffer_size', type=int, default=None, help='replay buffer size')
    ps.add_argument('--max_episode_steps', type=int, default=None)
    ps.add_argument('--device', type=str, default='cpu', help="cpu or cuda")

    # Learning related in common
    ps.add_argument('--lr', type=float, default=0.001, help="")
    ps.add_argument('--num_updates', type=int, default=None, help="")
    ps.add_argument('--sync_freq', type=int, default=1, help="")
    ps.add_argument('--soft_update_tau', type=float, default=None, help="")
    ps.add_argument('--sync_every_update', type=str2bool, default=False, help="")

    # Agent common args
    ps.add_argument('--agent_save', type=str2bool, default=False, help="")
    ps.add_argument('--agent_save_path', type=str, default="./model_log", help="")
    ps.add_argument('--agent_save_frequency', type=int, default=100, help="")
    ps.add_argument('--agent_load', type=str2bool, default=False, help="")
    ps.add_argument('--agent_load_path', type=str, default="./model_log", help="")
    ps.add_argument('--agent_load_epoch', type=int, default=0, help="")
    ps.add_argument('--Qnet_dim_hidden', type=str, default=None, help="")
    ps.add_argument('--Qnet_gamma', type=float, default=0.99, help="")
    ps.add_argument('--retrieve_Qnet_gamma', type=float, default=0.99,
                    help="In case we need to set a different gamma for retrieval Q net")

    # Scheduler
    ps.add_argument('--epsilon_start_cr', type=float, default=None, help="init value of epsilon decay")
    ps.add_argument('--epsilon_end_cr', type=float, default=None, help="final value of epsilon decay")
    ps.add_argument('--decay_steps_cr', type=int, default=None, help="init value of epsilon decay")
    ps.add_argument('--epsilon_start_act', type=float, default=None, help="init value of epsilon decay")
    ps.add_argument('--epsilon_end_act', type=float, default=None, help="final value of epsilon decay")
    ps.add_argument('--decay_steps_act', type=int, default=None, help="init value of epsilon decay")

    # Evaluation
    ps.add_argument('--eval_epsilon_ac', type=float, default=0.0, help="epsilon during evaluation")
    ps.add_argument('--eval_epsilon_cr', type=float, default=0.0, help="epsilon during evaluation")
    ps.add_argument('--eval_num_episodes', type=int, default=None, help="num of episodes in one training")
    ps.add_argument('--eval_freq', type=int, default=None, help="num of episodes in one training")
    ps.add_argument('--if_visualise_agent', type=str2bool, default=False, help="")
    ps.add_argument('--visualise_agent_freq_epoch', type=int, default=10, help="")
    ps.add_argument('--if_grad_clip', type=str2bool, default=None, help="")
    ps.add_argument('--video_save_frequency', type=int, default=20,
                    help="For mining and reacher envs, the frequency of saving videos")
    ps.add_argument('--do_naive_eval', type=str2bool, default=False, help="")
    ps.add_argument('--do_first_eval', type=str2bool, default=True, help="")

    # Shared over Continuous Environments
    ps.add_argument('--num_envs', type=int, default=None,
                    help="Number of parallel environments to be used for general environments")
    # ps.add_argument('--if_async', type=str2bool, default=False, help="")
    ps.add_argument('--if_async', type=str2bool, default=None, help="")
    ps.add_argument('--num_all_actions', type=int, default=None,
                    help="This is defined for discrete envs and None for continuous envs")
    # ps.add_argument('--vid_fps', type=float, default=30., help="FPS of video to be generated via save_mp4")
    ps.add_argument('--vid_fps', type=float, default=None, help="FPS of video to be generated via save_mp4")

    ps.add_argument('--dim_hidden', type=int, default=None, help='')
    ps.add_argument('--continuous_kNN_sigma', type=float, default=0.01, help='1% noise added for k-nearest neighbors')
    ps.add_argument('--env_dim_extra', type=int, default=None, help='')
    ps.add_argument('--env_act_emb_tSNE', type=str2bool, default=False, help='')
    return ps


def get_recsim_reacher_args(ps=None):
    if ps is None:
        ps = argparse.ArgumentParser()
    # Reacher specific
    ps.add_argument('--reacher_save_video', type=str2bool, default=False,
                    help="Whether or not to save evaluation videos in continuous environments such as reacher")
    ps.add_argument('--reacher_video_dir', type=str, default='./data/reacher_video')
    ps.add_argument('--reacher_action_type', type=str, default='original')
    ps.add_argument('--reacher_bijective_dims', type=int, default=5)
    ps.add_argument('--reacher_validity_type', type=str, default='none')
    ps.add_argument('--mujoco_env_box_seed', type=int, default=123)

    # RecSim specific
    ps.add_argument('--recsim_slate_size', type=int, default=1, help='If listwise RecSim then > 1')
    ps.add_argument('--recsim_user_budget', type=int, default=20, help="")
    ps.add_argument('--recsim_num_categories', type=int, default=30, help="")
    ps.add_argument('--recsim_dim_embed', type=int, default=30, help="")
    ps.add_argument('--recsim_no_click_mass', type=float, default=2, help="")
    ps.add_argument('--recsim_user_dist', type=str, default="sklearn-gmm", help="uniform / modal / gmm")
    ps.add_argument('--recsim_category_dist', type=str, default="random", help="")
    ps.add_argument('--recsim_item_dist', type=str, default="sklearn-gmm", help="")
    ps.add_argument('--recsim_choice_model_type', type=str, default="multinomial", help="deterministic / multinomial")
    ps.add_argument('--recsim_type_user_utility_computation', type=str, default="dot_prod", help="euc_dist / dot_prod")
    ps.add_argument('--recsim_step_penalty', type=float, default=0.5, help="")
    ps.add_argument('--recsim_if_user_update_even_no_click', type=str2bool, default=False, help="")
    ps.add_argument('--recsim_if_user_global_transition', type=str2bool, default=False, help="")
    ps.add_argument('--recsim_budget_logic_type', type=str, default="original", help="new / original / simple-original")
    ps.add_argument('--recsim_if_noisy_obs', type=str2bool, default=False, help="")
    ps.add_argument('--recsim_if_novelty_bonus', type=str2bool, default=False, help="")
    ps.add_argument('--recsim_sklearnGMM_if_sparse_centroids', type=str2bool, default=True, help="")
    ps.add_argument('--recsim_if_tsne_embed', type=str2bool, default=False, help="")
    ps.add_argument('--recsim_dim_tsne_embed', type=int, default=30, help="")
    ps.add_argument('--recsim_if_switch_act_task_emb', type=str2bool, default=False, help="")
    ps.add_argument('--recsim_if_valid_box', type=str2bool, default=False, help="")

    # recsim-data
    ps.add_argument('--recsim_data_dir', type=str, default="./data/movielens/ml_100k/ml-100k")
    ps.add_argument('--recsim_emb_type', type=str, default="pretrained")
    ps.add_argument('--recsim_reward_model_type', type=str, default="normal")
    ps.add_argument('--recsim_rm_obs_enc_type', type=str, default="deepset", help="deepset / lstm / transformer")
    ps.add_argument('--recsim_rm_if_film', type=str2bool, default=False, help="")
    ps.add_argument('--recsim_pre_offline_or_online', type=str, default="offline")
    ps.add_argument('--recsim_rm_dropout', type=float, default=0.0)

    # Creation of Task irrelevant action representations
    ps.add_argument('--recsim_act_emb_lin_scale', type=float, default=1.0, help="")
    ps.add_argument('--recsim_act_emb_lin_shift', type=float, default=1.0, help="")
    ps.add_argument('--recsim_act_emb_if_nonLin_transform', type=str2bool, default=False, help="")
    ps.add_argument('--recsim_act_emb_nonLin_transform_fn', type=int, default=21, help="21 = Tanh")
    return ps


def get_miningWorld_args(ps=None):
    if ps is None:
        ps = argparse.ArgumentParser()
    # agent Hyper-params
    ps.add_argument('--mw_type_obs_enc', type=str, default="cnn", help="cnn / old-cnn / flat-mlp")
    ps.add_argument('--mw_enc_channels', type=str, default="32_64_32", help="cnn / old-cnn / flat-mlp")
    ps.add_argument('--mw_obs_enc_lr', type=float, default=0.001, help="")
    ps.add_argument('--mw_fully_observable', type=str2bool, default=True, help="")
    ps.add_argument('--mw_obs_truth', type=str2bool, default=True,
                    help="now it must be true, or the parameters will be incorrect")

    ps.add_argument('--mw_obs_flatten', type=str2bool, default=False, help="")
    ps.add_argument('--mw_obs_id', type=str2bool, default=True, help="")
    ps.add_argument('--mw_obs_mine_one_hot', type=str2bool, default=True, help="")
    ps.add_argument('--mw_obs_wall', type=str2bool, default=False, help="")
    ps.add_argument('--mw_obs_goal', type=str2bool, default=False, help="")
    ps.add_argument('--mw_action_id', type=str2bool, default=True, help="")
    ps.add_argument('--mw_four_dir_actions', type=str2bool, default=True, help="")
    ps.add_argument('--mw_dir_one_hot', type=str2bool, default=False, help="")

    ps.add_argument('--mw_dim_state', type=int, default=32, help="")
    ps.add_argument('--mw_one_hot_mine_represent', type=str2bool, default=False, help="")
    ps.add_argument('--mw_embedding_perfect', type=str2bool, default=True, help="")

    ps.add_argument('--mw_grid_size', type=int, default=16, help="Size of the whole env. >= sqrt(NumRooms) * RoomSize")
    ps.add_argument('--mw_mine_tree_min_depth', type=int, default=2, help="")
    ps.add_argument('--mw_mine_tree_max_depth', type=int, default=2, help="")
    ps.add_argument('--mw_mine_size', type=int, default=15, help="")
    ps.add_argument('--mw_tool_size', type=int, default=50, help="")
    ps.add_argument('--mw_max_score', type=float, default=0.05, help="")
    ps.add_argument('--mw_time_penalty', type=float, default=0, help="")
    ps.add_argument('--mw_bonus', type=float, default=0.05, help="")
    ps.add_argument('--mw_step_penalty_coef', type=float, default=0.2, help="")
    ps.add_argument('--mw_goal_reaching_reward', type=float, default=30, help="")
    ps.add_argument('--mw_fullness', type=float, default=0.7, help="")
    ps.add_argument('--mw_maxRoomSize', type=int, default=10, help="")
    ps.add_argument('--mw_minRoomSize', type=int, default=10, help="")
    ps.add_argument('--mw_act_emb_lin_scale', type=float, default=1.0, help="")
    ps.add_argument('--mw_act_emb_lin_shift', type=float, default=0.1, help="")
    ps.add_argument('--mw_randomise_grid', type=str2bool, default=False, help="")
    ps.add_argument('--mw_rand_start_pos', type=str2bool, default=False, help="")
    ps.add_argument('--mw_start_from_first_room', type=str2bool, default=True, help="")
    ps.add_argument('--mw_rand_mine_score', type=str2bool, default=False, help="")
    ps.add_argument('--mw_rand_mine_category', type=str2bool, default=True, help="")
    ps.add_argument('--mw_test_save_action_embedding_tsne', type=str2bool, default=False, help="")
    ps.add_argument('--mw_test_save_video', type=str2bool, default=False, help="")
    ps.add_argument('--mw_train_save_video', type=str2bool, default=False, help="")
    ps.add_argument('--mw_video_append_action_candidate', type=str2bool, default=True, help="")
    ps.add_argument('--mw_video_dir', type=str, default='./videos', help="")
    ps.add_argument('--mw_if_high_dim', type=str2bool, default=False,
                    help="whether to transform the aciton space by matrix")
    ps.add_argument('--mw_new_action_dim', type=int, default=30, help="matrix shape")
    ps.add_argument('--mw_tsne_embedding', type=str2bool, default=False, help="matrix shape")
    ps.add_argument('--mw_tsne_dim', type=int, default=16, help="matrix shape")
    ps.add_argument('--mw_show_action_embeddings', type=str2bool, default=False, help="matrix shape")
    return ps


def get_WOLP_args(ps=None):
    if ps is None:
        ps = argparse.ArgumentParser()

    ps.add_argument('--WOLP_actor_lr', type=float, default=None, help="")
    ps.add_argument('--WOLP_critic_lr', type=float, default=1e-3, help="")
    ps.add_argument('--WOLP_ar_critic_lr', type=float, default=None, help="")
    ps.add_argument('--WOLP_list_enc_lr', type=float, default=5e-4, help="")
    # ps.add_argument('--WOLP_actor_dim_hiddens', type=str, default="64_32_32_16", help="")
    ps.add_argument('--WOLP_actor_dim_hiddens', type=str, default=None, help="")
    ps.add_argument('--WOLP_critic_dim_hiddens', type=str, default=None, help="")
    ps.add_argument('--WOLP_if_actor_init_layer', type=str2bool, default=True, help="")
    ps.add_argument('--WOLP_if_critic_init_layer', type=str2bool, default=True, help="")
    ps.add_argument('--WOLP_if_actor_norm_each', type=str2bool, default=True, help="")
    ps.add_argument('--WOLP_if_critic_norm_each', type=str2bool, default=True, help="")
    ps.add_argument('--WOLP_if_actor_norm_final', type=str2bool, default=True, help="")
    ps.add_argument('--WOLP_if_critic_norm_final', type=str2bool, default=True, help="")
    ps.add_argument('--WOLP_if_original_wolp_target_compute', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_topK', type=int, default=None, help="")
    ps.add_argument('--WOLP_type_metric', type=str, default="euc_dist", help="")
    ps.add_argument('--WOLP_if_cem_actor', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_cascade_list_len', type=int, default=None, help="")
    ps.add_argument('--WOLP_if_auto_ent_tune', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_cascade_type_list_reward', type=str, default="none", help="")
    ps.add_argument('--WOLP_slate_dim_out', type=int, default=16, help="")
    ps.add_argument('--WOLP_if_dual_exploration', type=str2bool, default=None, help="")
    ps.add_argument('--WOLP_if_refineQ_single_action_update', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_if_ar', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_if_ar_actor_use_prevAction', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_if_ar_critic_use_prevAction', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_if_ar_contextual_prop', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_if_ar_full_listEnc', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_if_ar_actor_share_weight', type=str2bool, default=True, help="")
    ps.add_argument('--WOLP_if_ar_critic_share_weight', type=str2bool, default=True, help="")
    ps.add_argument('--WOLP_ar_actor_type_init_weight', type=str, default="none", help="none / random / add")
    ps.add_argument('--WOLP_ar_critic_type_init_weight', type=str, default="none", help="none / random / add")
    ps.add_argument('--WOLP_if_ar_actor_cascade', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_if_ar_critic_cascade', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_type_list_encoder', type=str, default=None, help="deepset/lstm/transformer")
    ps.add_argument('--WOLP_type_ar_critic_GRU', type=str, default="both", help="")
    ps.add_argument('--WOLP_ar_type_cell', type=str, default="gru", help="")
    ps.add_argument('--WOLP_if_joint_critic', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_if_joint_actor', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_if_ar_cascade_list_enc', type=str2bool, default=False,
                    help="whether multi-agent with cascading or in parallel")
    ps.add_argument('--WOLP_ar_type_listwise_update', type=str, default="0-th-next-ts",
                    help="next-list-index / 0-th-next-ts")
    ps.add_argument('--WOLP_discounted_cascading', type=str2bool, default=True,
                    help="Try discounted cascading reward scheme for cascaded critics")
    ps.add_argument('--WOLP_refineQ_target', type=str2bool, default=False,
                    help="Refinement Q is evaluated on next state to compute the target value for retrieval critics")
    ps.add_argument('--WOLP_if_ar_selection_bonus', type=str2bool, default=False, help="whether selected by RefineQ")
    ps.add_argument('--WOLP_ar_selection_bonus', type=float, default=0.4, help="")
    ps.add_argument('--WOLP_if_new_exploration', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_if_use_full_next_Q', type=str2bool, default=False,
                    help="whether to use 0-th index or full indices of the next step in the update")
    ps.add_argument('--WOLP_if_dual_critic', type=str2bool, default=False, help="If use the extra critic in Wolp")
    ps.add_argument('--delayed_actor_training', type=int, default=0,
                    help='Delay actor training by how many steps')
    ps.add_argument('--WOLP_ar_if_opt_for_list_enc', type=str2bool, default=True, help="If separate opt for list-enc")
    ps.add_argument('--WOLP_if_pairwise_distance_bonus', type=str2bool, default=False,
                    help="whether to encourage the diversity queries")
    # ps.add_argument('--WOLP_pairwise_distance_bonus_coef', type=float, default=0.25, help="")
    ps.add_argument('--WOLP_pairwise_distance_bonus_coef', type=float, default=None, help="")
    ps.add_argument('--WOLP_if_ar_noise_before_cascade', type=str2bool, default=None, help="")
    ps.add_argument('--WOLP_if_ar_detach_list_action', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_if_new_list_obj', type=str, default="R_RL", help="first: R, ave, max / second: RL, bandit")
    ps.add_argument('--WOLP_allow_kNN_duplicate', type=str2bool, default=True, help="")
    # ps.add_argument('--WOLP_selection_if_boltzmann', type=str2bool, default=True, help="")
    ps.add_argument('--WOLP_selection_if_boltzmann', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_t0_no_list_input', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_if_film_listwise', type=str2bool, default=None, help="")
    ps.add_argument('--WOLP_dual_exp_if_ignore', type=str2bool, default=None, help="")

    # For new_flair
    ps.add_argument('--WOLP_Q_on_true_action', type=str2bool, default=None, help="")
    ps.add_argument('--WOLP_if_noise_postQ', type=str2bool, default=None,
                    help="whether to add noise to the action after Q-seletion does its arg max")
    ps.add_argument('--WOLP_noise_type', type=str, default=None, help="ou / normal")
    ps.add_argument('--WOLP_noise_expl_sigma', type=int, default=0.1, help="standard deviation of gaussian noise")
    ps.add_argument('--WOLP_if_0th_ref_critic', type=str2bool, default=None,
                    help="if copy the reference critic as the 0th AR critic")
    ps.add_argument('--WOLP_if_ar_imitate', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_list_encoder_deepset_maxpool', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_no_ar_critics', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_list_concat_state', type=str2bool, default=None, help="")
    ps.add_argument('--WOLP_total_dual_exploration', type=str2bool, default=None, help="")
    ps.add_argument('--WOLP_ar_use_query_max', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_knn_action_rep', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_mean_policy_loss', type=str2bool, default=False,
                    help="Whether to take a mean or sum over the ar dimension.")
    ps.add_argument('--WOLP_separate_update', type=str2bool, default=False,
                    help="Whether to update the selection and retrieval agents on different batches")
    ps.add_argument('--WOLP_knn_inside_cascade', type=str2bool, default=False,
                    help="If do knn between every list action inside the actor")
    ps.add_argument('--WOLP_ar_fresh_update', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_TD3_target_policy_smoothing', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ignore_knn_for_target', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_use_main_ref_critic_for_target', type=str2bool, default=None, help="")
    ps.add_argument('--WOLP_ar_use_taken_action', type=str2bool, default=False,
                    help="whether to compute Q_ar(s, taken_action | list_action) instead of Q_ar(s, list_action | list_action)")
    ps.add_argument('--WOLP_use_main_ref_critic_for_action_selection', type=str2bool, default=None, help="")
    ps.add_argument('--WOLP_ar_use_mu_star', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_use_star_for_update_conditioning', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_conditioning_star', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_critic_eps_action', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_use_ref_next_Q', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_actor_no_conditioning', type=str2bool, default=None, help="")
    ps.add_argument('--WOLP_ar_critic_scaled_num_updates', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_value_loss_if_sum', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_improvement_as_target', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_critic_taken_action_update', type=str2bool, default=None, help="")
    ps.add_argument('--WOLP_if_min_improvement_0', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_use_conservative_Q_max', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_use_0th_for_target_action_selection', type=str2bool, default=False, help="")
    ps.add_argument('--WOLP_ar_critic_target_smoothing', type=str2bool, default=False, help="")

    # # # For WOLP_dual
    ps.add_argument('--WOLP_if_dual_critic_imitate', type=str2bool, default=None,
                    help="If imitate the ref critic for the dual critic's target")
    ps.add_argument('--WOLP_if_dual_critic_kNN_target', type=str2bool, default=True,
                    help="If use the target ref network to compute kNN targets")
    ps.add_argument('--WOLP_twin_target', type=str2bool, default=None,
                    help="If use twin target network to compute targets")

    # GreedyAC Args
    ps.add_argument('--greedy_N', type=int, default=None,
                    help="number of actions to sample from sampling policy")
    ps.add_argument('--greedy_Nrho', type=int, default=None,
                    help="number of top actions to use for update")
    ps.add_argument('--clip_log_std_threshold', type=int, default=1000,
                    help="number of top actions to use for update")
    ps.add_argument('--greedy_entropy_from_single_sample', type=str2bool, default=True,
                    help="whether to use entropy from single sample in entropy update of sampler policy")
    ps.add_argument('--greedy_alpha', type=float, default=0.01,
                    help="entropy weighting term")
    ps.add_argument('--greedy_td3_exploration', type=str2bool, default=False,
                    help="whether to use entropy from single sample in entropy update of sampler policy")

    # SAVO_threshold Args
    ps.add_argument('--WOLP_threshold_Q', type=str2bool, default=False,
                    help="whether to use primary critic directly for action selection with max thresholding")
    ps.add_argument('--WOLP_threshold_Q_direct_cummax', type=str2bool, default=False,
                    help="whether to use primary critic directly for action selection with max thresholding")

    ps.add_argument('--WOLP_0th_no_perturb', type=str2bool, default=None,
                    help="in the perturbation step, do not perturb the 0th action")
    ps.add_argument('--WOLP_preturb_with_fixed_Gaussian', type=str2bool, default=False,
                    help="Perturn with fixed Gaussian, explore with OU/Gaussian")
    ps.add_argument('--WOLP_twin_main_for_target', type=str2bool, default=True,
                    help="In main_ref_critic_for_target update, use the twin main critic")
    ps.add_argument('--WOLP_discrete_kNN_target_smoothing', type=str2bool, default=None, help="")

    # TD3
    ps.add_argument('--TwinQ', type=str2bool, default=None, help="")
    ps.add_argument('--TD3_policy_delay', type=int, default=1, help="")
    ps.add_argument('--if_train_every_ts', type=str2bool, default=False, help="")
    ps.add_argument('--TD3_target_policy_smoothing', type=str2bool, default=None, help="")
    ps.add_argument('--TD3_policy_noise', type=float, default=0.2, help="")
    ps.add_argument('--TD3_noise_clip', type=float, default=0.5, help="")

    ps.add_argument('--CEM_num_iter', type=int, default=3, help="")
    ps.add_argument('--CEM_num_samples', type=int, default=30, help="")
    ps.add_argument('--CEM_topK', type=int, default=3, help="")
    ps.add_argument('--CEM_rescale_actions', type=str2bool, default=True, help="")

    ps.add_argument('--DEBUG_type_activation', type=str, default="tanh", help="sigmoid / tanh / none")
    ps.add_argument('--DEBUG_size_action_space', type=str, default="large", help="small / large / unbounded")
    ps.add_argument('--DEBUG_type_clamp', type=str, default="none", help="large / small / none")

    # REDQ
    ps.add_argument('--REDQ', type=str2bool, default=False, help="Whether to turn on REDQ")
    ps.add_argument('--REDQ_num', type=int, default=5, help="Number of critics in REDQ")
    ps.add_argument('--REDQ_num_random', type=int, default=2, help="Number of randomized critics to select")
    ps.add_argument('--REDQ_SAVO', type=str2bool, default=False, help="Whether to turn on REDQ for Successive critics")

    # SAC-SAVO
    ps.add_argument('--SAC_SAVO', type=str2bool, default=False, help="Whether to turn on SAC_SAVO")
    ps.add_argument('--sac_savo_resample', type=str2bool, default=False,
                    help="Whether to resample from distribution that is maxed by SAVO")
    ps.add_argument('--sac_savo_evaluate_mean', type=str2bool, default=True,
                    help="Whether to resample from distribution that is maxed by SAVO")
    ps.add_argument('--sac_savo_include_entropy', type=str2bool, default=True,
                    help="Whether to use entropy in SAC maximization")
    ps.add_argument('--sac_savo_append_prev_actions', type=str2bool, default=True,
                    help="Whether to use previous actions in successive actors")

    # PRIMACY BIAS
    ps.add_argument('--resets', type=str2bool, default=False, help="Whether to fully reset the agent at intervals")
    ps.add_argument('--reset_interval', type=int, default=200_000, help="Frequency of resetting")
    ps.add_argument('--agent_full_reset', type=str2bool, default=False, help="Whether to fully reset the agent at intervals")
    ps.add_argument('--replay_ratio', type=int, default=1, help="More updates per step")

    # PENALTY
    ps.add_argument('--WOLP_actor_penalty', type=str2bool, default=False,
                    help="Whether to penalize the actor")
    ps.add_argument('--WOLP_actor_penalty_sigma', type=float, default=0.5,
                    help="Standard deviation of the penalty range")
    ps.add_argument('--WOLP_actor_penalty_weight', type=float, default=1.0,
                    help="Weight of the penalty loss")
    return ps


def add_args(args: argparse.Namespace):
    """ Set the env specific params """

    # ==== Agreed Common setups
    # Overwritten later by method_name:
    args.WOLP_if_critic_norm_each = args.WOLP_if_critic_norm_final = False
    args.WOLP_refineQ_target = False
    # Not overwritten later by method_name:
    args.WOLP_if_refineQ_single_action_update = True
    # args.WOLP_allow_kNN_duplicate = True
    if not args.env_name.lower().startswith('mujoco'):
        args.min_replay_buffer_size = 0
    args.WOLP_dual_exp_if_ignore = False
    # ==== Agreed Common setups

    # args.env_name = "recsim-data"
    # args.device = "cpu"

    if args.env_name.startswith("recsim"):
        # args.recsim_if_valid_box = True
        # args.env_dim_extra = 0
        # args.recsim_act_emb_lin_shift = 0.0
        # args.recsim_if_switch_act_task_emb = True
        # args.env_act_emb_tSNE = True

        # args.WOLP_t0_no_list_input = True
        # args.WOLP_if_ar_detach_list_action = True  # Try if this should be applied to all environments
        # args.WOLP_if_ar_noise_before_cascade = True  # Check its behavior for various agents.
        # args.WOLP_if_film_listwise = True

        # Environment Args
        # args.num_all_actions = 500
        # args.num_all_actions = 500000
        # args.num_all_actions = 100000
        args.env_dim_extra=5
        args.recsim_dim_embed = args.recsim_num_categories = args.recsim_dim_tsne_embed
        set_if_none(args, 'total_ts', 10_000_000)
        args.num_epochs = 10000
        args.buffer_size = 1000000
        args.max_episode_steps = 20
        # args.max_episode_steps = 200
        args.eval_num_episodes = 64
        args.if_visualise_agent = True
        args.visualise_agent_freq_epoch = 10
        args.WOLP_if_new_list_obj = "R_RL"
        args.num_envs = 20
        args.if_async = False
        set_method_based_args(args)
        # Change from other mujoco envs!
        args.if_train_every_ts = False
        set_per_train_ts(args)
        set_if_none(args, 'decay_steps_cr', 4_000_000)
        set_other_args(args)

        # In-active args now
        args.num_updates = 20 # NOT Active
        set_if_none(args, 'eval_freq', 25) # Not Active

        # Commenting out old-working arguments
        '''
        args.lr = 0.001
        args.sync_freq = 1
        args.Qnet_gamma = 0.99

         # Actor-specific
        args.epsilon_start_act = 1.0
        args.epsilon_end_act = 0.1
        args.decay_steps_act = 400000

        args.WOLP_actor_lr = 0.0001
        args.WOLP_critic_lr = 0.001
        args.WOLP_list_enc_lr = 0.0005
        args.WOLP_if_actor_norm_each = args.WOLP_if_actor_norm_final = True
        '''
        # Work around to check if it's no_perturb or not
        if_no_perturb = args.epsilon_start_act == 0.0 and args.epsilon_start_cr == 0.0 and args.epsilon_end_act == 0.0 and args.epsilon_end_cr == 0.0

        if args.env_name == "recsim-500":
            args.num_all_actions = 500
            args.env_name = "recsim"  # remove the flag to change the exploration flags

            if args.method_name.lower() == "wolp_dual":
                args.WOLP_total_dual_exploration = True
                args.epsilon_end_act = 0
                args.epsilon_end_cr = 0.01
                args.epsilon_start_act = 1
                args.epsilon_start_cr = 0.5
                args.decay_steps_act = 2_000_000
                args.decay_steps_cr = 2_000_000
            else:  # same as flair's params
                args.WOLP_total_dual_exploration = True
                args.epsilon_end_act = 0
                args.epsilon_end_cr = 0.01
                args.epsilon_start_act = 1
                args.epsilon_start_cr = 0.5
                args.decay_steps_act = 5_000_000
                args.decay_steps_cr = 10_000_000
        elif args.env_name == "recsim-5k":
            args.num_all_actions = 5000
            args.env_name = "recsim"  # remove the flag to change the exploration flags

            if args.method_name.lower() == "wolp_dual":
                args.WOLP_total_dual_exploration = True
                args.epsilon_end_act = 0.01
                args.epsilon_end_cr = 0.01
                args.epsilon_start_act = 1
                args.epsilon_start_cr = 0.5
                args.decay_steps_act = 2_000_000
                args.decay_steps_cr = 10_000_000
            else:  # same as flair's params
                args.WOLP_total_dual_exploration = True
                args.epsilon_end_act = 0.01
                args.epsilon_end_cr = 0.01
                args.epsilon_start_act = 1
                args.epsilon_start_cr = 0.5
                args.decay_steps_act = 2_000_000
                args.decay_steps_cr = 5_000_000
        elif args.env_name == "recsim-10k":
            args.num_all_actions = 10000
            args.env_name = "recsim"  # remove the flag to change the exploration flags

            if args.method_name.lower() == "wolp_dual":
                args.WOLP_total_dual_exploration = False
                args.epsilon_end_act = 0.01
                args.epsilon_end_cr = 0.01
                args.epsilon_start_act = 1
                args.epsilon_start_cr = 0.5
                args.decay_steps_act = 2_000_000
                args.decay_steps_cr = 2_000_000
            else:  # same as flair's params
                args.WOLP_total_dual_exploration = False
                args.epsilon_end_act = 0.01
                args.epsilon_end_cr = 0.01
                args.epsilon_start_act = 0.5
                args.epsilon_start_cr = 0.5
                args.decay_steps_act = 10_000_000
                args.decay_steps_cr = 2_000_000
        elif args.env_name == "recsim-100k":
            args.num_all_actions = 100000
            args.env_name = "recsim"  # remove the flag to change the exploration flags

            if args.method_name.lower() == "wolp_dual":
                args.WOLP_total_dual_exploration = True
                args.epsilon_end_act = 0.01
                args.epsilon_end_cr = 0.01
                args.epsilon_start_act = 1
                args.epsilon_start_cr = 0.5
                args.decay_steps_act = 2_000_000
                args.decay_steps_cr = 10_000_000
            else:  # same as flair's params
                args.WOLP_total_dual_exploration = False
                args.epsilon_end_act = 0.0
                args.epsilon_end_cr = 0.01
                args.epsilon_start_act = 1.0
                args.epsilon_start_cr = 0.5
                args.decay_steps_act = 5_000_000
                args.decay_steps_cr = 2_000_000
        elif args.env_name == "recsim-500k":
            args.num_all_actions = 500000
            args.env_name = "recsim"  # remove the flag to change the exploration flags

            if args.method_name.lower() == "wolp_dual":
                args.WOLP_total_dual_exploration = False
                args.epsilon_end_act = 0.01
                args.epsilon_end_cr = 0.01
                args.epsilon_start_act = 1
                args.epsilon_start_cr = 0.5
                args.decay_steps_act = 10_000_000
                args.decay_steps_cr = 2_000_000
            else:  # same as flair's params
                args.WOLP_total_dual_exploration = False
                args.epsilon_end_act = 0.0
                args.epsilon_end_cr = 0.01
                args.epsilon_start_act = 1.0
                args.epsilon_start_cr = 0.5
                args.decay_steps_act = 2_000_000
                args.decay_steps_cr = 2_000_000

        # ============= Agent setup
        args.Qnet_dim_hidden = "64_32"
        # DQN-specific
        args.epsilon_start_cr = 0.5
        args.epsilon_end_cr = 0.01
        args.WOLP_actor_dim_hiddens = "64_32_32_16"
        set_if_none(args, 'dim_hidden', 64)
        args.WOLP_slate_dim_out = 32
        args.WOLP_pairwise_distance_bonus_coef = 0.33
        if args.env_name.lower().startswith("recsim-data"):
            args.max_episode_steps = 80
            args.env_dim_extra = 0
            if args.env_name.lower() == "recsim-data-rating5":
                args.recsim_data_dir = "./data/movielens/ml-rating5-2"
                if args.method_name.lower() == "wolp_dual":
                    args.WOLP_total_dual_exploration = True
                    args.epsilon_end_act = 0.01
                    args.epsilon_end_cr = 0.01
                    args.epsilon_start_act = 0.5
                    args.epsilon_start_cr = 0.5
                    args.decay_steps_act = 10_000_000
                    args.decay_steps_cr = 10_000_000
                else:  # same as flair's params
                    args.WOLP_total_dual_exploration = True
                    args.epsilon_end_act = 0.0
                    args.epsilon_end_cr = 0.01
                    args.epsilon_start_act = 1.0
                    args.epsilon_start_cr = 0.5
                    args.decay_steps_act = 10_000_000
                    args.decay_steps_cr = 10_000_000
            else:
                args.recsim_data_dir = "./data/movielens/ml-pre-reward3"
                if args.method_name.lower() == "wolp_dual":
                    # args.WOLP_total_dual_exploration = False
                    args.epsilon_end_act = 0.01
                    # args.epsilon_end_cr = 0.01
                    args.epsilon_start_act = 1.0
                    # args.epsilon_start_cr = 0.5
                    args.decay_steps_act = 2_000_000
                    # args.decay_steps_cr = 2_000_000
                else:  # same as flair's params
                    # args.WOLP_total_dual_exploration = False
                    args.epsilon_end_act = 0.0
                    # args.epsilon_end_cr = 0.01
                    args.epsilon_start_act = 1.0
                    # args.epsilon_start_cr = 0.5
                    args.decay_steps_act = 2_000_000
                    # args.decay_steps_cr = 2_000_000

            args.env_name = "recsim-data"  # remove the flag to change the exploration flags
            args.num_all_actions = ML100K_NUM_ITEMS
            args.if_visualise_agent = False
            # args.total_ts = 2000000
            # args.num_epochs = 2000
            # args.buffer_size = 500000
            # args.decay_steps_cr = 800000
            # args.decay_steps_act = 80000
            # args.recsim_data_dir = "train"
            args.env_dim_extra = 0
            args.recsim_act_emb_lin_shift = 0.0
            args.mw_obs_flatten = False  # because of MineWorld implementation
            args.mw_obs_truth = False  # because of MineWorld implementation

        # Work around for no-perturb
        if if_no_perturb:
            args.epsilon_start_act = 0.0
            args.epsilon_start_cr = 0.0
            args.epsilon_end_act = 0.0
            args.epsilon_end_cr = 0.0
    elif args.env_name == "mine":
        set_mine_args(args)
        set_method_based_args(args)
        # Change from other mujoco envs!
        args.if_train_every_ts = False
        set_per_train_ts(args) # Eval freq stays at 100
        args.Qnet_dim_hidden = "128_128"
        args.WOLP_actor_dim_hiddens = "128_64_64_32"
        set_other_args(args)
        # set_if_none(args, "Qnet_dim_hidden", "128_128")
        # set_if_none(args, "WOLP_actor_dim_hiddens", "128_64_64_32")
        # args.mw_test_save_video = False
        # Safety check
        if args.mw_four_dir_actions:
            MW_ACTION_OFFSET = 4
        else:
            assert False, "some features are not supported so first upgrade the codebase before running!"
            MW_ACTION_OFFSET = 3
        if args.mw_mine_size <= 10:
            assert args.mw_mine_size <= args.mw_tool_size
        else:
            assert (args.mw_mine_size <= args.mw_tool_size <= (
                    (args.mw_mine_size ** 2) // 4)), f"{args.mw_mine_size}, {args.mw_tool_size}"
        # if not (args.mw_mine_size <= args.mw_tool_size <= ((args.mw_mine_size ** 2) // 4)):
        #     args.mw_mine_size = np.round(np.sqrt(4 * args.mw_tool_size)).astype(np.int) + 1
        #     assert args.mw_mine_size <= args.mw_tool_size <= ((args.mw_mine_size ** 2) // 4)
        args.num_all_actions = args.mw_tool_size + MW_ACTION_OFFSET
        # args.num_all_actions = args.mw_tool_size
        # assert not np.alltrue([args.mw_action_id, args.mw_if_high_dim])
        if args.mw_action_id:
            args.mw_action_dim = 4
        else:
            if args.mw_dir_one_hot:
                args.mw_action_dim = 2 * (args.mw_mine_size + 1) + MW_ACTION_OFFSET + 1
            else:
                args.mw_action_dim = 2 * (args.mw_mine_size + 1) + 2 + 1

        if args.mw_if_high_dim:
            args.mw_action_dim = args.mw_new_action_dim

        if args.mw_tsne_embedding:
            args.mw_action_dim = args.mw_tsne_dim

        if args.mw_fully_observable:
            args.mw_observation_size = args.mw_minRoomSize - 2  ## 2 represent the walls of the room
        else:
            args.mw_observation_size = 7

        if args.mw_obs_id:
            args.mw_obs_channel = 3
        else:
            if args.mw_fully_observable:
                if args.mw_dir_one_hot:
                    args.mw_obs_channel = 8
                else:
                    args.mw_obs_channel = 9
            # elif args.mw_if_simple_obs:
            #     args.mw_obs_channel = 1
            else:
                args.mw_obs_channel = 7

    elif args.env_name.startswith('mujoco'):
        set_mujoco_args(args)
        set_other_args(args)
        if args.agent_type == 'ddpg':
            args.WOLP_dual_exp_if_ignore = False

    if args.mw_obs_flatten:
        args.mw_obs_length = args.mw_observation_size * args.mw_observation_size * args.mw_obs_channel

    if args.mw_obs_truth:
        args.mw_obs_channel = 8
        args.mw_observation_size = 1
        args.mw_obs_flatten = True
        args.mw_obs_length = 8
        if args.mw_obs_mine_one_hot:
            args.mw_obs_channel = 8 + args.mw_mine_size
            args.mw_obs_length = 8 + args.mw_mine_size

    # ==== Agent specific modification: Just a safety-net to avoid making mistakes in hyper-param setting!!
    if args.agent_type == "random":
        args.device = "cpu"
    elif args.agent_type == "dqn":
        args.WOLP_if_pairwise_distance_bonus = False
        args.WOLP_dual_exp_if_ignore = False
    elif args.agent_type.startswith("wolp") or args.agent_type.startswith('arddpg_cont'):
        if args.WOLP_ar_type_listwise_update == "0-th-next-ts" and args.WOLP_cascade_type_list_reward == "none":
            # can1
            args.WOLP_if_ar_actor_share_weight = args.WOLP_if_ar_critic_share_weight = False
        if args.WOLP_ar_type_listwise_update == "next-list-index" and args.WOLP_cascade_type_list_reward == "last":
            # can2
            args.WOLP_if_ar_actor_share_weight = args.WOLP_if_ar_critic_share_weight = True
        if args.WOLP_ar_type_listwise_update == "next-list-index" and args.WOLP_cascade_type_list_reward == "none":
            # can3
            args.WOLP_if_ar_actor_share_weight = args.WOLP_if_ar_critic_share_weight = True
        if args.WOLP_ar_type_listwise_update == "next-ts-same-index" and args.WOLP_cascade_type_list_reward == "none":
            # can4
            args.WOLP_if_ar_actor_share_weight = args.WOLP_if_ar_critic_share_weight = False

        if args.WOLP_if_refineQ_single_action_update:
            args.WOLP_if_critic_norm = False
        if args.WOLP_if_ar and args.WOLP_if_joint_critic:
            args.WOLP_discounted_cascading = False
        if args.agent_type.lower() == "wolp-sac":
            set_if_none(args, "WOLP_topK", 1)
        if args.WOLP_if_joint_actor:
            args.WOLP_if_joint_critic = True
        if args.agent_type == "wolp-sac":
            args.WOLP_if_new_list_obj = "R_RL"

    args.WOLP_if_new_list_obj = args.WOLP_if_new_list_obj.lower().split("_")
    set_if_none(args, "WOLP_topK", 1)
    return args


def get_all_args():
    ps = argparse.ArgumentParser()
    ps = get_args(ps=ps)
    ps = get_WOLP_args(ps=ps)
    ps = get_recsim_reacher_args(ps=ps)
    ps = get_miningWorld_args(ps=ps)
    args = ps.parse_args()
    args = add_args(args=args)
    return args
