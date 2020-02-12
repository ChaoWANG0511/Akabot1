#!/usr/bin/env python
# -*- coding:utf-8 -*-


class EstimatorSpecBuilder(object):


    # Supported loss functions.
    L1_MASK = 'L1_mask'
    WEIGHTED_L1_MASK = 'weighted_L1_mask'

    # Supported optimizers.
    ADADELTA = 'Adadelta'
    SGD = 'SGD'

    # Math constants.
    WINDOW_COMPENSATION_FACTOR = 2./3.
    EPSILON = 1e-10

    def __init__(self, features, params):
        self._features = features
        self._params = params
        # Get instrument name.
        self._mix_name = params['mix_name']
        self._instruments = params['instrument_list']
        # Get STFT/signals parameters
        self._n_channels = params['n_channels']
        self._T = params['T']
        self._F = params['F']
        self._frame_length = params['frame_length']
        self._frame_step = params['frame_step']

    def _build_loss(self, output_dict, labels):
        loss_type = self._params.get('loss_type', self.L1_MASK)
        if loss_type == self.L1_MASK:
            losses = {
                name: tf.reduce_mean(tf.abs(output - labels[name]))
                for name, output in output_dict.items()
            }
        elif loss_type == self.WEIGHTED_L1_MASK:
            losses = {
                name: tf.reduce_mean(
                    tf.reduce_mean(
                        labels[name],
                        axis=[1, 2, 3],
                        keep_dims=True) *
                    tf.abs(output - labels[name]))
                for name, output in output_dict.items()
            }
        else:
            raise ValueError(f"Unkwnown loss type: {loss_type}")
        loss = tf.reduce_sum(list(losses.values()))
        # Add metrics for monitoring each instrument.
        metrics = {k: tf.compat.v1.metrics.mean(v) for k, v in losses.items()}
        metrics['absolute_difference'] = tf.compat.v1.metrics.mean(loss)
        return loss, metrics

    def _build_optimizer(self):
        name = self._params.get('optimizer')
        if name == self.ADADELTA:
            return tf.compat.v1.train.AdadeltaOptimizer()
        rate = self._params['learning_rate']
        if name == self.SGD:
            return tf.compat.v1.train.GradientDescentOptimizer(rate)
        return tf.compat.v1.train.AdamOptimizer(rate)

    def build_predict_model(self):
        """ Builder interface for creating model instance that aims to perform
        prediction / inference over given track. The output of such estimator
        will be a dictionary with a "<instrument>" key per separated instrument
        , associated to the estimated separated waveform of the instrument.

        :returns: An estimator for performing prediction.
        """
        self._build_stft_feature()
        output_dict = self._build_output_dict()
        output_waveform = self._build_output_waveform(output_dict)
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.PREDICT,
            predictions=output_waveform)

    def _build_output_dict(self):
        """ Created a batch_sizexTxFxn_channels input tensor containing
        mix magnitude spectrogram, then an output dict from it according
        to the selected model in internal parameters.

        :returns: Build output dict.
        :raise ValueError: If required model_type is not supported.
        """
        input_tensor = self._features[f'{self._mix_name}_spectrogram']
        model = self._params.get('model', None)
        if model is not None:
            model_type = model.get('type', self.DEFAULT_MODEL)
        else:
            model_type = self.DEFAULT_MODEL
        try:
            apply_model = get_model_function(model_type)
        except ModuleNotFoundError:
            raise ValueError(f'No model function {model_type} found')
        return apply_model(
            input_tensor,
            self._instruments,
            self._params['model']['params'])

    def build_train_model(self, labels):
        """ Builder interface for creating model instance that aims to perform
        model training. The output of such estimator will be a dictionary
        with a key "<instrument>_spectrogram" per separated instrument,
        associated to the estimated separated instrument magnitude spectrogram.

        :param labels: Model labels.
        :returns: An estimator for performing model training.
        """
        output_dict = self._build_output_dict()
        loss, metrics = self._build_loss(output_dict, labels)
        optimizer = self._build_optimizer()
        train_operation = optimizer.minimize(
                loss=loss,
                global_step=tf.compat.v1.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_operation,
            eval_metric_ops=metrics,
        )


def model_fn(features, labels, mode, params, config):
    builder = EstimatorSpecBuilder(features, params)
    return builder.build_train_model(labels)

