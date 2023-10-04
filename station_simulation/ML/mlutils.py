import sys
import tensorflow.keras.backend as kb
import tensorflow as tf

# for i in progressbar([1,2,10,5], "Computing: "):
#    compute()

def progressbar(it, prefix="", size=60):
    """
    Plot a progress bar.
    """
    file=sys.stdout
    count = len(it)

    def show(j):
        x = int(size*j/count)
        file.write(f"{prefix}[{'#'*x}{'.'*(size-x)}] {j}/{count}\r")
        file.flush()
    
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def get_loss_function(batch_size,T):
    """
    Returns customn loss function

    ATTENTION: this is in progress and not functional!
    """
    def loss(y_actual,y_pred):
        # both tensors: [batch, time, 1]
        #kb.print_tensor(y_actual)
        #kb.print_tensor(y_pred)
        y_actual = kb.cast(y_actual, dtype='float32')
        y_pred   = kb.cast(y_pred,   dtype='float32')
        custom_loss = kb.square(y_actual-y_pred)
        return custom_loss

    return loss