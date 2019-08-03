import numpy as np
import tensorflow as tf
import gym
import gym_maze
import time
import spinup.algos.vpg.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, hierstep=3):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.init_obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.goal_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.count_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.obshi_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.acthi_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.advhi_buf = np.zeros(size, dtype=np.float32)
        self.rewhi_buf = np.zeros(size, dtype=np.float32)
        self.rethi_buf = np.zeros(size, dtype=np.float32)
        self.valhi_buf = np.zeros(size, dtype=np.float32)
        self.logphi_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam, self.hierstep = gamma, lam, hierstep
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.ptrhi, self.path_start_idxhi = 0, 0

    def store(self, obs, init_obs, goal, count, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.init_obs_buf[self.ptr] = init_obs
        self.goal_buf[self.ptr] = goal
        self.count_buf[self.ptr] = count
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def storehi(self, obshi, acthi, rewhi, valhi, logphi):
        assert self.ptrhi < self.max_size
        self.obshi_buf[self.ptrhi] = obshi
        self.acthi_buf[self.ptrhi] = acthi
        self.rewhi_buf[self.ptrhi] = rewhi
        self.valhi_buf[self.ptrhi] = valhi
        self.logphi_buf[self.ptrhi] = logphi
        self.ptrhi += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def finish_path_hi(self, last_val=0):

        path_slice = slice(self.path_start_idxhi, self.ptrhi)
        rews = np.append(self.rewhi_buf[path_slice], last_val)
        vals = np.append(self.valhi_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma ** self.hierstep * vals[1:] - vals[:-1]
        self.advhi_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.rethi_buf[path_slice] = core.discount_cumsum(rews, self.gamma ** self.hierstep)[:-1]

        self.path_start_idxhi = self.ptrhi

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        #adv_mean = np.mean(self.adv_buf)
        #adv_std = np.std(self.adv_buf)
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.init_obs_buf, self.goal_buf, np.expand_dims(self.count_buf, axis=1), 
                self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]

    def gethi(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # the next two lines implement the advantage normalization trick
        #adv_mean = np.mean(self.advhi_buf[:self.ptrhi])
        #adv_std = np.std(self.advhi_buf[:self.ptrhi])
        adv_mean, adv_std = mpi_statistics_scalar(self.advhi_buf)
        int_num = int(self.ptrhi)
        self.ptrhi, self.path_start_idxhi = 0, 0
        return [self.obshi_buf[:int_num], self.acthi_buf[:int_num],
                (self.advhi_buf[:int_num] - adv_mean) / adv_std, self.rethi_buf[:int_num], self.logphi_buf[:int_num]]


"""
Vanilla Policy Gradient
(with GAE-Lambda for advantage estimation)
"""


def vpg(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=10, c=3):
    """
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure
                                           | to flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to VPG.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        c: hierearchical step length
    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    print("HELLLLLOOOOOOOOOOOOOOOOO")
    print(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    #ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, x_initial_ph, g_ph, a_ph = core.placeholders_from_spaces(env.observation_space,
                                                              env.observation_space,
                                                              env.observation_space,
                                                              env.action_space)
    count_ph = core.placeholder(1)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    xhi_ph, ahi_ph = core.placeholders_from_spaces(env.observation_space, env.observation_space)
    advhi_ph, rethi_ph, logphi_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    x_concat = tf.concat([x_ph, x_initial_ph, g_ph, count_ph], 1)
    pi, logp, logp_pi, v = actor_critic(x_concat, a_ph, action_space=env.action_space)
    pihi, logphi, logphi_pi, vhi = actor_critic(xhi_ph, ahi_ph, action_space=env.observation_space)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, x_initial_ph, g_ph, count_ph, a_ph, adv_ph, ret_ph, logp_old_ph]
    allhi_phs = [xhi_ph, ahi_ph, advhi_ph, rethi_ph, logphi_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]
    gethi_action_ops = [pihi, vhi, logphi_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # VPG objectives
    pi_loss = -tf.reduce_mean(logp * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v) ** 2)
    pihi_loss = -tf.reduce_mean(logphi * advhi_ph)
    vhi_loss = tf.reduce_mean((rethi_ph - vhi) ** 2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute
    approx_klhi = tf.reduce_mean(logphi_old_ph - logphi)      # a sample estimate for KL-divergence, easy to compute
    approx_enthi = tf.reduce_mean(-logphi)                  # a sample estimate for entropy, also easy to compute

    # Optimizers
    train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)
    trainhi_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pihi_loss)
    trainhi_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(vhi_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def update():
        inputs = {k: valbuf for k, valbuf in zip(all_phs, buf.get())}
        inputshi = {k: valbuf for k, valbuf in zip(allhi_phs, buf.gethi())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)
        pihi_l_old, vhi_l_old, enthi = sess.run([pihi_loss, vhi_loss, approx_enthi], feed_dict=inputshi)
        # Policy gradient step
        sess.run(train_pi, feed_dict=inputs)
        sess.run(trainhi_pi, feed_dict=inputshi)

        # Value function learning
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)
            sess.run(trainhi_v, feed_dict=inputshi)

        # Log changes from update
        pi_l_new, v_l_new, kl = sess.run([pi_loss, v_loss, approx_kl], feed_dict=inputs)
        pihi_l_new, vhi_l_new, klhi = sess.run([pihi_loss, vhi_loss, approx_klhi], feed_dict=inputshi)
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     KL=kl, Entropy=ent, 
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old),
                     LossPiHi=pihi_l_old, LossVHi=vhi_l_old, 
                     KLHi=klhi, EntropyHi=enthi, 
                     DeltaLossPiHi=(pihi_l_new - pihi_l_old),
                     DeltaLossVHi=(vhi_l_new - vhi_l_old))

    start_time = time.time()
    reset = env.reset()
    o, x_init, count, r, rhi, r_intr, d, ep_ret, ep_len = reset, reset, 0, 0, 0, 0, False, 0, 0
    g, vhi_t, logphi_t = sess.run(gethi_action_ops, feed_dict={xhi_ph: x_init.reshape(1, -1)})
    buf.storehi(x_init, np.squeeze(g, axis=0), rhi, vhi_t, logphi_t)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops,
                                      feed_dict={x_ph: o.reshape(1, -1), x_initial_ph: x_init.reshape(1, -1),
                                                 g_ph: g.reshape(1, -1), count_ph: np.expand_dims([count%c], axis=1)})
            buf.store(o, x_init, g, count%c, a, r_intr, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a[0])
            ep_ret += r
            ep_len += 1
            r_intr = -np.linalg.norm(o - g, ord=2) #low level reward calculation via simple euclidian distance
            rhi += r
            count += 1
            if count % c == 0 and buf.ptrhi < buf.max_size:
                buf.finish_path(r_intr)
                x_init = o
                g, vhi_t, logpihi_t = sess.run(gethi_action_ops, feed_dict={xhi_ph: x_init.reshape(1, -1)})
                buf.storehi(x_init, np.squeeze(g, axis=0), rhi, vhi_t, logpihi_t)
                logger.store(VValsHi=vhi_t)
                rhi = 0

            terminal = d or (count == max_ep_len)
            if terminal or (t == local_steps_per_epoch - 1):
                if not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' %count)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r_intr if d else sess.run(v,
                                                feed_dict={x_ph: o.reshape(1, -1), x_initial_ph: x_init.reshape(1, -1),
                                                           g_ph: g.reshape(1, -1),
                                                           count_ph: np.expand_dims([count%c], axis=1)})
                if count%c != 0:
                    buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                lasthi_val = rhi if d else sess.run(vhi, feed_dict={xhi_ph: o.reshape(1, -1)})
                buf.finish_path_hi(lasthi_val)
                reset = env.reset()
                o, x_init, count, r, rhi, r_intr, d, ep_ret, ep_len = reset, reset, 0, 0, 0, 0, False, 0, 0
                g, vhi_t, logpihi_t = sess.run(gethi_action_ops, feed_dict={xhi_ph: x_init.reshape(1, -1)})
                buf.storehi(x_init, np.squeeze(g, axis=0), rhi, vhi_t, logpihi_t)
                logger.store(VValsHi=vhi_t)

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform VPG update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('VValsHi', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('LossPiHi', average_only=True)
        logger.log_tabular('LossVHi', average_only=True)
        logger.log_tabular('DeltaLossPiHi', average_only=True)
        logger.log_tabular('DeltaLossVHi', average_only=True)
        logger.log_tabular('EntropyHi', average_only=True)
        logger.log_tabular('KLHi', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCar-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--c', type=int, default='3')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    vpg(lambda: gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, logger_kwargs=logger_kwargs,
        c=args.c)
