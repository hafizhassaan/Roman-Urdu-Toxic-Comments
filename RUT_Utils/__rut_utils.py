import warnings
warnings.filterwarnings( 'ignore' )
import numpy as np
from sklearn.metrics import f1_score
from keras.callbacks import Callback
import keras.backend as K

def optimize_threshold( y_true, y_pred ):
    thresholds = np.arange( 0.1, 0.9, 0.001 )
    vscores = np.zeros( thresholds.shape[0] )
    for th in range( thresholds.shape[0] ):
        vscores[ th ] = f1_score( y_true, ( y_pred >= thresholds[ th ] ).astype( 'int32' ) )
    return thresholds[ np.argmax( vscores ) ] # return threshold that has max val score

class F1Metrics( Callback ):
    def __init__( self, filepath, patience=10, decay=True, decay_rate=0.1, decay_after=2, softmax=False ):
        self.file_path = filepath
        self.patience = patience
        self.patience_counter = 0
        self.decay = decay
        self.decay_rate = decay_rate
        self.decay_after = decay_after
        self.softmax = softmax
    
    def __optimize_threshold_for_f1( self, y_true, y_pred ):
        thresholds = np.arange( 0.1, 0.9, 0.001 )
        vscores = np.zeros( thresholds.shape[0] )
        for th in range( thresholds.shape[0] ):
            vscores[ th ] = f1_score( y_true, ( y_pred >= thresholds[ th ] ).astype( 'int32' ) )
        return thresholds[ np.argmax( vscores ) ]

    def on_train_begin( self, logs=None ):
        self.val_f1s = []
        self.best_val_f1 = 0

    def on_epoch_end( self, epoch, logs={} ):
        val_predict = self.model.predict( self.validation_data[0] )
        val_targ = self.validation_data[1]
        
        if self.softmax == True:
            val_targ = [ np.argmax(y, axis=None, out=None) for y in val_targ ]
            val_predict = val_predict[ :, 1 ]
        
        threshold = self.__optimize_threshold_for_f1( val_targ, val_predict )
        
        _val_f1 = f1_score( val_targ, ( val_predict >= threshold ).astype( 'int32' ) )
        self.val_f1s.append(_val_f1)
        
        printstatement = 'Epoch: {:03d}'.format( epoch ) + ' --MaxValF1: {:0.8f}'.format( max( self.val_f1s ) ) + \
        ' --CurValF1: {:0.8f}'.format( _val_f1 ) + ' --Patience: {:02d}'.format( self.patience_counter )
        
        self.patience_counter = self.patience_counter + 1
        
        if _val_f1 > self.best_val_f1:
            self.model.save(self.file_path, overwrite=True)
            self.best_val_f1 = _val_f1
            printstatement = printstatement + ' --improved f1: {:0.8f}'.format( self.best_val_f1 )
            self.patience_counter = 0
        
        print( printstatement )
        
        if ( (self.decay==True) & ((self.patience_counter % self.decay_after) == 0) & (self.patience_counter != 0) ):
            K.set_value( self.model.optimizer.lr, ( K.get_value( self.model.optimizer.lr ) * self.decay_rate ) )
        
        if self.patience_counter == self.patience:
            self.model.stop_training = True
            print( 'Training stopped due to the patience parameter.' + \
                  ' --Patience: {:02d}'.format( self.patience_counter ) )
        
        return
