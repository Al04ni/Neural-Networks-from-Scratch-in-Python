from keras.initializers import VarianceScaling


def glorot_uniform(seed=None):
    """Glorot uniform initializer, also called Xavier uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    # Arguments
    seed: A Python integer. Used to seed the random generator.
    # Returns
    An initializer.
    # References
    Glorot & Bengio, AISTATS 2010
    http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    return VarianceScaling(scale=1.,
                           mode='fan_avg',
                           distribution='uniform',
                           seed=seed)