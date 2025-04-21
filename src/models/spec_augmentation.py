import tensorflow as tf


class SpectrogramAugmentation(tf.keras.layers.Layer):
    """Simple spectrogram augmentation layer using fixed masks"""

    def __init__(self, freq_mask_param=10, time_mask_param=10, **kwargs):
        """Initialize the augmentation layer"""
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

        # Create fixed frequency mask
        self.freq_mask = tf.ones((95, 126, 1))  # Full size mask
        freq_zeros = tf.zeros((freq_mask_param, 126, 1))
        freq_start = (95 - freq_mask_param) // 2

        # Create fixed time mask
        self.time_mask = tf.ones((95, 126, 1))  # Full size mask
        time_zeros = tf.zeros((95, time_mask_param, 1))
        time_start = (126 - time_mask_param) // 2

        # Update masks with zeros at fixed positions
        self.freq_mask = tf.tensor_scatter_nd_update(
            self.freq_mask,
            [
                [i, j, 0]
                for i in range(freq_start, freq_start + freq_mask_param)
                for j in range(126)
            ],
            tf.reshape(freq_zeros, [-1]),
        )

        self.time_mask = tf.tensor_scatter_nd_update(
            self.time_mask,
            [
                [i, j, 0]
                for i in range(95)
                for j in range(time_start, time_start + time_mask_param)
            ],
            tf.reshape(time_zeros, [-1]),
        )

    def build(self, input_shape):
        """Build the layer with input shape information"""
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer"""
        return input_shape

    def call(self, inputs, training=None):
        """Apply augmentation if in training mode"""
        if not training:
            return inputs

        # Get input shape
        batch_size = tf.shape(inputs)[0]
        channels = tf.shape(inputs)[-1]

        # Create masks for all channels
        freq_mask = tf.tile(self.freq_mask, [1, 1, channels])
        time_mask = tf.tile(self.time_mask, [1, 1, channels])

        # Random selection between freq and time mask
        mask = tf.cond(
            tf.random.uniform([], 0, 1) > 0.5, lambda: freq_mask, lambda: time_mask
        )

        # Broadcast mask to batch dimension
        mask = tf.tile(tf.expand_dims(mask, 0), [batch_size, 1, 1, 1])

        return inputs * mask

    def get_config(self):
        """Get layer configuration"""
        config = super().get_config()
        config.update(
            {
                "freq_mask_param": self.freq_mask_param,
                "time_mask_param": self.time_mask_param,
            }
        )
        return config
