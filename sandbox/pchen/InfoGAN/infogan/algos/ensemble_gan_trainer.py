# from sandbox.pchen.InfoGAN.infogan.models.regularized_gan import RegularizedGAN
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Bernoulli, Gaussian, Mixture, DiscretizedLogistic, ConvAR
import sys
import rllab.misc.logger as logger

TINY = 1e-8
TINY_G_P = 1e-2
TINY_D_P = 1e-1


class EnsembleGANTrainer(object):
    def __init__(
            self,
            model,
            dataset,
            batch_size,
            exp_name="experiment",
            log_dir="logs",
            checkpoint_dir="ckt",
            max_epoch=100,
            updates_per_epoch=100,
            snapshot_interval=10000,
            discriminator_learning_rate=2e-4,
            generator_learning_rate=2e-4,
            discriminator_leakage="all",  # [all, single]
            discriminator_priviledge="all",  # [all, single]
            bootstrap_rate=1.0,
            anneal_to=None,
            anneal_len=None,
            natural_step=None,
            natural_g_only=False,
            natural_d_one_sided=False,
            second_order_natural_approx=False,
            natural_anneal_len=None,
            fixed_sampling_noise=False,
            d_multiples=1,
            g_multiples=1,
            tgt_network=False,
            tgt_network_update_freq=1,
    ):
        """
        :type model: EnsembleGAN
        """
        self.natural_d_one_sided = natural_d_one_sided
        self.tgt_network_update_freq = tgt_network_update_freq
        self.natural_anneal_len = natural_anneal_len
        self.second_order_natural_approx = second_order_natural_approx
        self.tgt_network = tgt_network
        if tgt_network:
            assert natural_step is not None
            import pickle
            with tf.variable_scope("tgt"):
                self.tgt_model = pickle.loads(pickle.dumps(model))
            self.tgt_update_op = None

        self.g_multiples = g_multiples
        self.d_multiples = d_multiples
        self.natural_g_only = natural_g_only
        self.fixed_sampling_noise = fixed_sampling_noise
        self.natural_step = natural_step
        self.anneal_len = anneal_len
        self.anneal_to = anneal_to
        self.bootstrap_rate = bootstrap_rate
        self.discriminator_priviledge = discriminator_priviledge
        self.discriminator_leakage = discriminator_leakage
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.snapshot_interval = snapshot_interval
        self.updates_per_epoch = updates_per_epoch
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator_trainer = None
        self.generator_trainer = None
        self.input_tensor = None
        self.log_vars = []
        self.imgs = None
        self.anneal_factor = None

    def init_opt(self):
        self.input_tensor = input_tensor = tf.placeholder(
            tf.float32,
            [self.batch_size, self.dataset.image_dim]
        )

        with pt.defaults_scope(phase=pt.Phase.train):
            z_var = self.model.latent_dist.sample_prior(self.batch_size)
            fake_x, _ = self.model.generate(z_var)
            all_d_logits = self.model.discriminate(
                tf.concat(0, [input_tensor, fake_x]),
                logits=True,
            )
            if self.tgt_network:
                tgt_all_d_logits = self.tgt_model.discriminate(
                    tf.concat(0, [input_tensor, fake_x]),
                    logits=True,
                )
                vars_dict = dict([
                    (v.name, v) for v in tf.trainable_variables()
                ])
                self.tgt_update_op = tf.group(*[
                    tgt_var.assign(vars_dict[tgt_var_name[4:]])
                    for tgt_var_name, tgt_var in vars_dict.items()
                    if tgt_var_name[:4] == "tgt/"
                ])


            if self.discriminator_priviledge == "all":
                real_d_logits = all_d_logits[:self.batch_size]
                fake_d_logits = all_d_logits[self.batch_size:]
                if self.tgt_network:
                    tgt_real_d_logits = tgt_all_d_logits[:self.batch_size]
                    tgt_fake_d_logits = tgt_all_d_logits[self.batch_size:]

                if self.bootstrap_rate != 1.:
                    real_d_logits = tf.nn.dropout(
                        real_d_logits,
                        self.bootstrap_rate
                    ) * self.bootstrap_rate
                    fake_d_logits = tf.nn.dropout(
                        fake_d_logits,
                        self.bootstrap_rate
                    ) * self.bootstrap_rate
            elif self.discriminator_priviledge == "single":
                real_d_logits = tf.reduce_min(
                    all_d_logits[:self.batch_size],
                    reduction_indices=[1],
                )
                fake_d_logits = tf.reduce_max(
                    all_d_logits[self.batch_size:],
                    reduction_indices=[1],
                )
                if self.tgt_network:
                    tgt_real_d_logits = tf.reduce_min(
                        tgt_all_d_logits[:self.batch_size],
                        reduction_indices=[1],
                    )
                    tgt_fake_d_logits = tf.reduce_max(
                        tgt_all_d_logits[self.batch_size:],
                        reduction_indices=[1],
                    )

                assert self.bootstrap_rate == 1.
            else:
                raise Exception("sup")

            # specific to 2nd approx adaptive kl size
            if self.natural_anneal_len or self.natural_step:
                self.natural_step_var = tf.Variable(
                    initial_value=1./TINY_G_P if self.natural_anneal_len else self.natural_step,
                    name="natural_step",
                    trainable=False,
                )
            if self.natural_step is None or self.natural_g_only:
                real_d_tgt = tf.ones_like(real_d_logits)
                fake_d_tgt = tf.zeros_like(fake_d_logits)
            else:
                if self.tgt_network:
                    real_p = tf.stop_gradient(tf.nn.sigmoid(tgt_real_d_logits))
                    fake_p = tf.stop_gradient(tf.nn.sigmoid(tgt_fake_d_logits))
                else:
                    real_p = tf.stop_gradient(tf.nn.sigmoid(real_d_logits))
                    fake_p = tf.stop_gradient(tf.nn.sigmoid(fake_d_logits))
                if self.second_order_natural_approx:
                    real_step = 2. * self.natural_step_var * tf.maximum(real_p, TINY_D_P) * (1. - real_p)
                    fake_step = 2. * self.natural_step_var * fake_p * tf.maximum(1. - fake_p, TINY_D_P)
                else:
                    real_step = fake_step = self.natural_step
                if self.natural_d_one_sided:
                    real_d_tgt = tf.ones_like(real_d_logits)
                else:
                    real_d_tgt = tf.minimum(
                        real_p + real_step,
                        1.,
                    )
                fake_d_tgt = tf.maximum(
                    fake_p - fake_step,
                    0.,
                )
                mean, var = tf.nn.moments(fake_d_tgt, list(range(len(fake_d_tgt.get_shape()))))
                self.log_vars.append(("fake_d_tgt_mean", mean))
                self.log_vars.append(("fake_d_tgt_var", var))

            discriminator_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                real_d_logits,
                real_d_tgt
            ) + tf.nn.sigmoid_cross_entropy_with_logits(
                fake_d_logits,
                fake_d_tgt
            )
            discriminator_loss = tf.reduce_mean(discriminator_losses)

            if self.discriminator_leakage == "all":
                fake_g_logits = all_d_logits[self.batch_size:]
                if self.tgt_network:
                    tgt_fake_g_logits = tgt_all_d_logits[self.batch_size:]
            elif self.discriminator_leakage == "single":
                fake_g_logits = tf.reduce_min(
                    all_d_logits[self.batch_size:],
                    reduction_indices=[1],
                )
                if self.tgt_network:
                    tgt_fake_g_logits = tf.reduce_min(
                        tgt_all_d_logits[self.batch_size:],
                        reduction_indices=[1],
                    )
            else:
                raise Exception("sup")

            if self.natural_step is None:
                fake_g_tgt = tf.zeros_like(fake_g_logits)
            else:
                if self.tgt_network:
                    fake_p = tf.stop_gradient(tf.nn.sigmoid(tgt_fake_g_logits))
                else:
                    fake_p = tf.stop_gradient(tf.nn.sigmoid(fake_g_logits))
                if self.second_order_natural_approx:
                    fake_step = 2. * self.natural_step_var * tf.maximum(fake_p, TINY_G_P) * (1. - fake_p)
                else:
                    fake_step = self.natural_step
                mean, var = tf.nn.moments(fake_step, list(range(len(fake_step.get_shape()))))
                self.log_vars.append(("fake_g_step_mean", mean))
                self.log_vars.append(("fake_g_step_var", var))
                fake_g_tgt = tf.minimum(
                    fake_p + fake_step,
                    1.,
                )
                mean, var = tf.nn.moments(fake_g_tgt, list(range(len(fake_step.get_shape()))))
                self.log_vars.append(("fake_g_tgt_mean", mean))
                self.log_vars.append(("fake_g_tgt_var", var))
            generator_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                fake_g_logits,
                fake_g_tgt
            )
            generator_loss = tf.reduce_mean(generator_losses)

            all_vars = tf.trainable_variables()
            d_vars = [var for var in all_vars if var.name.startswith('d_')]
            g_vars = [var for var in all_vars if var.name.startswith('g_')]


            if self.natural_step is None:
                self.log_vars.append(("discriminator_loss", discriminator_loss))
                self.log_vars.append(("generator_loss", generator_loss))
            else:
                self.log_vars.append(("natural_discriminator_loss", discriminator_loss))
                self.log_vars.append(("natural_generator_loss", generator_loss))
                self.log_vars.append((
                    "discriminator_loss",
                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        real_d_logits,
                        tf.ones_like(real_d_logits)
                    ) + tf.nn.sigmoid_cross_entropy_with_logits(
                        fake_d_logits,
                        tf.zeros_like(fake_d_logits)
                    ))
                ))
                self.log_vars.append((
                    "generator_loss",
                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        fake_g_logits,
                        tf.ones_like(fake_g_logits)
                    ))
                ))
                if self.tgt_network:
                    self.log_vars.append((
                        "tgt_discriminator_loss",
                        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            tgt_real_d_logits,
                            tf.ones_like(real_d_logits)
                        ) + tf.nn.sigmoid_cross_entropy_with_logits(
                            tgt_fake_d_logits,
                            tf.zeros_like(fake_d_logits)
                        ))
                    ))
                    self.log_vars.append((
                        "tgt_generator_loss",
                        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            tgt_fake_g_logits,
                            tf.ones_like(fake_g_logits)
                        ))
                    ))
                    self.log_vars.append(("tgt_max_real_d", tf.reduce_max(tgt_real_d_logits)))
                    self.log_vars.append(("tgt_min_real_d", tf.reduce_min(tgt_real_d_logits)))
                    self.log_vars.append(("tgt_max_fake_d", tf.reduce_max(tgt_fake_d_logits)))
                    self.log_vars.append(("tgt_min_fake_d", tf.reduce_min(tgt_fake_d_logits)))
            self.log_vars.append(("max_real_d", tf.reduce_max(real_d_logits)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d_logits)))
            self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d_logits)))
            self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d_logits)))
            # self.log_vars.append(("min_z_var", tf.reduce_min(z_var)))
            # self.log_vars.append(("max_z_var", tf.reduce_max(z_var)))

            self.anneal_factor = tf.Variable(
                initial_value=1.,
                name="opt_anneal_factor",
                trainable=False,
            )
            discriminator_optimizer = tf.train.AdamOptimizer(
                self.discriminator_learning_rate * self.anneal_factor,
                beta1=0.5
            )
            self.discriminator_trainer = pt.apply_optimizer(
                discriminator_optimizer,
                losses=[discriminator_loss],
                var_list=d_vars
            )

            generator_optimizer = tf.train.AdamOptimizer(
                self.generator_learning_rate * self.anneal_factor,
                beta1=0.5
            )
            self.generator_trainer = pt.apply_optimizer(
                generator_optimizer,
                losses=[generator_loss],
                var_list=g_vars
            )
        with pt.defaults_scope(phase=pt.Phase.test):
            for name, var in self.log_vars:
                tf.scalar_summary(name, var)

            with tf.variable_scope("model", reuse=True) as scope:
                z_var = self.model.latent_dist.sample_prior(
                    self.batch_size
                )
                if self.fixed_sampling_noise:
                    z_var = tf.Variable(
                        initial_value=z_var,
                        trainable=False,
                    )
                img_var, _ = self.model.generate(z_var)

                rows = int(np.sqrt(self.batch_size))
                img_var = tf.reshape(
                    img_var,
                    [self.batch_size, ] + list(self.dataset.image_shape)
                )
                img_var = img_var[:rows * rows, :, :, :]
                imgs = tf.reshape(img_var, [rows, rows, ] + list(self.dataset.image_shape))
                stacked_img = []
                for row in range(rows):
                    row_img = []
                    for col in range(rows):
                        row_img.append(imgs[row, col, :, :, :])
                    stacked_img.append(tf.concat(1, row_img))
                imgs = tf.concat(0, stacked_img)
                imgs = tf.expand_dims(imgs, 0)
                self.imgs = imgs
                tf.image_summary("image", imgs, max_images=3)

    def train(self):

        self.init_opt()

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)

            saver = tf.train.Saver()

            imgs = sess.run(self.imgs, )
            import scipy
            import scipy.misc
            scipy.misc.imsave(
                "%s/init.png" % (self.log_dir),
                imgs.reshape(
                    list(imgs.shape[1:3]) +
                    (
                        []
                        if self.model.image_shape[-1] == 1
                        else [self.model.image_shape[-1]]
                    )
                ),
            )

            counter = 0

            log_vars = [x for _, x in self.log_vars]
            log_keys = [x for x, _ in self.log_vars]

            for epoch in range(self.max_epoch):
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                pbar.start()

                if self.anneal_len:
                    factor = sess.run([
                        self.anneal_factor.assign(
                            (1 - self.anneal_to) * max(0, self.anneal_len - epoch) / self.anneal_len
                            + self.anneal_to
                        )
                    ])
                    print("Factor annealed to %s" % factor)
                if self.natural_anneal_len:
                    eps = sess.run([
                        self.natural_step_var.assign(
                            (1./TINY_P - self.natural_step) * max(0, self.natural_anneal_len - epoch) / self.natural_anneal_len
                            + self.natural_step
                        )
                    ])
                    print("Natural eps annealed to %s" % eps)

                if self.tgt_network and (epoch % self.tgt_network_update_freq == 0):
                    sess.run([self.tgt_update_op])

                all_log_vals = []
                for i in range(self.updates_per_epoch):
                    pbar.update(i)
                    x, _ = self.dataset.train.next_batch(self.batch_size)
                    # x = np.reshape(x, (-1, 28, 28, 1))
                    for _ in range(self.d_multiples):
                        log_vals = sess.run([self.discriminator_trainer] + log_vars, {self.input_tensor: x})[1:]
                    for _ in range(self.g_multiples):
                        sess.run(self.generator_trainer, {self.input_tensor: x})
                    all_log_vals.append(log_vals)
                    counter += 1

                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                        print(("Model saved in file: %s" % fn))

                x, _ = self.dataset.train.next_batch(self.batch_size)
                # if (epoch % (self.max_epoch // 10)) == 0:
                if (epoch % (5)) == 0:
                    summary_str, imgs = sess.run([summary_op, self.imgs], {self.input_tensor: x})
                    import scipy
                    import scipy.misc
                    scipy.misc.imsave(
                        "%s/epoch_%s.png" % (self.log_dir, epoch),
                        imgs.reshape(
                            list(imgs.shape[1:3]) +
                            (
                                []
                                if self.model.image_shape[-1] == 1
                                else [self.model.image_shape[-1]]
                            )
                        ),
                    )
                else:
                    summary_str = sess.run(summary_op, {self.input_tensor: x})
                summary_writer.add_summary(summary_str, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                # log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                for k,v in zip(log_keys, avg_log_vals):
                    logger.record_tabular("%s"%k, v)
                logger.dump_tabular(with_prefix=False)

                # print(log_line)
                # sys.stdout.flush()