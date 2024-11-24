import tensorflow as tf

class GAN(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(GAN, self).__init__(*args, **kwargs)
        self.training_balance = 5
        self.critic_steps = tf.Variable(0, trainable=False)
        # everything else...

    
    def train_generator(self, batch_size):
        """https://www.tensorflow.org/guide/function#conditionals
        """
        random_latent_vectors = self.latent_space_distn((batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_data = self.generator(random_latent_vectors)
            score = self.critic(generated_data, training=False)
            generator_loss = -tf.reduce_mean(score)
        grads = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        self.generator_loss_tracker.update_state(generator_loss)
        return generator_loss


    def dummy(self, *args, **kwargs):
        return tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)
    

    @tf.function
    def train_step(self, batch):
        batch_size = tf.shape(batch)[0] # dynamic for graph mode
        
        # train critic
        critic_loss, critic_real, critic_fake = self.train_critic(batch, batch_size)
        metrics = {
            "critic_loss": critic_loss,
            'critic_real': critic_real,
            'critic_fake': critic_fake
        }
        self.critic_steps.assign_add(1)

        # train generator
        ifelse = tf.math.logical_or(tf.math.equal(self.critic_steps, 1), tf.math.equal(self.critic_steps % self.training_balance, 0))
        train_generator = lambda: self.train_generator(batch_size) # in my (conditional) model I need more args
        generator_loss = tf.cond(ifelse, train_generator, self.dummy)
        metrics["generator_loss"] = self.generator_loss_tracker.result()

        return metrics