import gin
import keras
from cv_framework.metrics.metrics import Summary_metrics

@gin.configurable
def callback_list(calls=None, gen=None, checkpoint_name='test.h5'):
    added_calls = [] if not calls else calls
    check_name = f'{checkpoint_name}_checkpoint.h5'

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        check_name,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        period=1
    )
    tb_callback = keras.callbacks.TensorBoard(log_dir='./logs')
    earlystopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)
    rop_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        verbose=0,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    )
    Summary_metrics_callback = Summary_metrics(val_gen=gen)

    base_calls = [tb_callback, checkpoint_callback, earlystopping_callback, rop_callback, Summary_metrics_callback]
    return base_calls + added_calls

@gin.configurable
def fit_generator(model_name=None,model=None, gen=None, epochs=100, validation_data=None, class_weight=None, workers=1,
                 use_multiprocessing=False):
    epoch_steps = (len(gen.classes)//gen.batch_size) + 1
    val_steps = (len(validation_data.classes)//validation_data.batch_size) + 1
    callbacks = callback_list(gen=validation_data, checkpoint_name=model_name)
    return model.fit_generator(
        generator=gen,
        steps_per_epoch=epoch_steps,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=val_steps,
        class_weight=class_weight,
        workers=workers,
        use_multiprocessing=use_multiprocessing
    )

@gin.configurable
def save_model(model=None, model_name='test.h5'):
    model.save(model_name)
