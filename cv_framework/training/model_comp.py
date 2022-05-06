import keras
import gin
from cv_framework.metrics.metrics import muilticlass_logloss
from cv_framework.model_definitions import standard_arch_dict

@gin.configurable
def opt_params(optimizer, params):
    params_default_dict = {
        'SGD':{'lr':0.01, 'momentum':0.0, 'decay':0.0, 'nesterov':False},
        'RMSprop':{'lr':0.001, 'rho':0.9, 'epsilon':None, 'decay':0.0},
        'Adagrad':{'lr':0.01, 'epsilon':None, 'decay':0.0},
        'Adadelta':{'lr':1.0, 'rho':0.95, 'epsilon':None, 'decay':0.0},
        'Adam':{'lr':0.001, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':None, 'decay':0.0, 'amsgrad':False},
        'Adamax':{'lr':0.002, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':None, 'decay':0.0},
        'Nadam':{'lr':0.002, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':None, 'schedule_decay':0.004},
    }

    values = params.values()
    if all(x is None for x in values):
        return params_default_dict[optimizer]
    allowed_opt_params = {
        'SGD':[
            'lr',
            'momentum',
            'decay',
            'nestrov'
        ],
        'RMSprop':[
            'lr',
            'rho',
            'epsilon',
            'decay'
        ],
        'Adagrad':[
            'lr',
            'epsilon',
            'decay'
        ],
        'Adadelta':[
            'lr',
            'rho',
            'epsilon',
            'decay'
        ],
        'Adam':[
            'lr',
            'beta_1',
            'beta_2',
            'epsilon',
            'decay',
            'amsgrad'
        ],
        'Adamax':[
            'lr',
            'beta_1',
            'beta_2',
            'epsilon',
            'decay'
        ],
        'Nadam':[
            'lr',
            'beta_1',
            'beta_2',
            'epsilon',
            'schedule_decay'
        ]
    }

    for key in list(params.keys()):
        if key not in allowed_opt_params[optimizer]:
            params.pop(key)
        elif not params[key]:
            params[key] = params_default_dict[optimizer][key]
    return params

@gin.configurable
def set_optimizer(optimizer, lr=None, momentum=None, decay=None, nestrov=None, rho=None, epsilon=None,
                  beta_1=None, beta_2=None, amsgrad=None, schedule_decay=None):
    allowed_optimizers = [
        'SGD',
        'RMSprop',
        'Adagrad',
        'Adadelta',
        'Adam',
        'Adamax',
        'Nadam'
    ]

    params =  {
        'lr':lr,
        'momentum':momentum,
        'decay':decay,
        'nestrov':nestrov,
        'rho':rho,
        'epsilon':epsilon,
        'beta_1':beta_1,
        'beta_2':beta_2,
        'amsgrad':amsgrad,
        'schedule_decay':schedule_decay
    }

    if optimizer not in allowed_optimizers:
        raise ValueError(f'Optimizer {optimizer} not in the allowed list: {allowed_optimizers}')

    fin_params = opt_params(optimizer, params)

    optimizers = {
        'SGD':keras.optimizers.SGD,
        'RMSprop':keras.optimizers.RMSprop,
        'Adagrad':keras.optimizers.Adagrad,
        'Adadelta':keras.optimizers.Adadelta,
        'Adam':keras.optimizers.Adam,
        'Adamax':keras.optimizers.adamax,
        'Nadam':keras.optimizers.Nadam,
        }

    return optimizers[optimizer](** fin_params)

@gin.configurable
def loss_function(loss=None):
    allowed_losses = [
        'mean_squared_error',
        'mean_absolute_error',
        'mean_absolute_percentage_error',
        'mean_squared_logarithmic_error',
        'squared_hinge',
        'hinge',
        'categorical_hinge',
        'logcosh',
        'categorical_crossentropy',
        'sparse_categorical_crossentropy',
        'binary_crossentropy',
        'kullback_leibler_divergence',
        'poisson',
        'cosine_proximity'
    ]
    if not loss:
        raise Exception('Loss value must be set!')

    if loss not in allowed_losses:
        raise ValueError(f'Losses {loss} is not in allowed losses: {allowed_losses}')
    else:
        return loss

@gin.configurable
def batch_metrics(metrics=None):
    batch_metrics = ['acc', muilticlass_logloss]
    if not metrics:
        return batch_metrics
    elif isinstance(metrics, list):
        return batch_metrics + metrics
    else:
        raise ValueError('Batch metrics must be a list.')

@gin.configurable
def comp_model(model_name=None, model_arch=None, loss=None, metrics=None, input_shape=None, classes=None, lr=None,
               momentum=None, decay=None, nestrov=None, rho=None, epsilon=None, beta_1=None, beta_2=None, amsgrad=None,
               schedule_decay=None, optimizer=None):
    print(f'model name: {model_name}')
    cnn_model = model_arch(input_shape=input_shape, classes=classes)
    opt = set_optimizer(optimizer=optimizer, lr=lr, momentum=momentum, decay=decay, nestrov=nestrov, rho=rho, epsilon=epsilon,
                    beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad, schedule_decay=schedule_decay)
    loss = loss_function(loss=loss)
    metrics = batch_metrics(metrics=metrics)
    cnn_model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return cnn_model
