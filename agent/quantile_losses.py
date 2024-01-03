class QuantileLoss(snt.Module):
    def __init__(self, loss_type='huber', b_decay=0.9, name: Optional[str] = None):
        super().__init__(name=name)
        self.loss_type = loss_type
        self.b_decay = b_decay
        self.b = tf.Variable(1.0, dtype=tf.float32, name='b', trainable=False)

    def huber(self, x: tf.Tensor, k=1.0):
        return tf.where(tf.abs(x) < k, 0.5 * tf.pow(x, 2), k * (tf.abs(x) - 0.5 * k))


    def generalized_loss(self, td_error: tf.Tensor, b: tf.Tensor):
        """Implements GL: Generalized Huber Loss
                       td_error: TD error
                       b:   threshold parameter
                       """
        abs_u = tf.abs(td_error)
        def f(x): return (1.0 + tf.math.erf(x / tf.sqrt(2.0))) / 2.0
        phi = f(-abs_u/b)
        loss = tf.multiply(abs_u, (1.0 - 2*phi)) + b*tf.sqrt(2.0/math.pi) * \
            tf.exp(-tf.pow(abs_u, 2.0)/(2*b*b)) - b*math.sqrt(2.0/math.pi)
        return loss


    def generalized_loss_approximate(self, td_error: tf.Tensor, b: tf.Tensor):
        """Implements GLA: Taylor Approximation of Generalized Huber Loss (GL)
               td_error: TD error
               b:   threshold parameter
               """
        abs_u = tf.abs(td_error)
        loss = tf.where(abs_u <= b, tf.pow(abs_u, 2.0) /
                        (b*math.sqrt(2.0*math.pi)), abs_u - b*math.sqrt(2.0/math.pi))
        return loss



    def __call__(self,
                q_tm1: QuantileDistribution,
                r_t: tf.Tensor,
                d_t: tf.Tensor,
                q_t: QuantileDistribution):
        """Implements Quantile Regression Loss
        q_tm1: critic distribution of t-1
        r_t:   reward
        d_t:   discount
        q_t:   target critic distribution of t
        loss_type: 'huber', 'GL', 'GL-A'
        """

        z_t = tf.reshape(r_t, (-1, 1)) + tf.reshape(d_t, (-1, 1)) * q_t.values
        z_tm1 = q_tm1.values
        diff = tf.expand_dims(tf.transpose(z_t), -1) - \
            z_tm1    # (n_tau_p, n_batch, n_tau)
        diff_detach = tf.stop_gradient(diff)

        if self.loss_type == 'huber':
            k = 1
            loss = self.huber(diff, k) / k
        else:
            std1 = tf.math.reduce_std(
                tf.cast(z_t, dtype=tf.float32), 1)      # (n_batch,1)
            std2 = tf.math.reduce_std(
                tf.cast(z_tm1, dtype=tf.float32), 1)   # (n_batch,1)
            self.b.assign(self.b*self.b_decay + (1 - self.b_decay)* tf.reduce_mean(tf.abs(std1 - std2)))
            b = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
            b = b.write(0, self.b.value())
            b = tf.cast(b.stack(), tf.float32)
            if self.loss_type == 'GL':
                loss = self.gaussian_loss(diff, b)
            elif self.loss_type == 'GL_A':
                loss = self.gaussian_loss_approximate(diff, b)


        loss *= tf.abs(q_tm1.quantiles -
                    tf.cast(diff_detach < 0, diff_detach.dtype))  # quantile regression loss
        return tf.reduce_mean(loss, (0, -1))
