import warnings
warnings.filterwarnings( 'ignore' )

from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
from thundersvm import SVC

from keras.models import Model
from keras.layers import Input, Embedding
from keras.layers import CuDNNGRU, CuDNNLSTM, Conv1D, Conv2D, Dense, Bidirectional, GRU, LSTM, MaxPool1D
from keras.layers import SpatialDropout1D, Dropout, Concatenate, concatenate, Softmax, Flatten, Reshape
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import multi_gpu_model
from keras.optimizers import *

def NB_Model(  ):
    model = BernoulliNB()
    return model
def RF_Model( n_est=500, crit='entropy', bootstrap=True, oob_score=True, class_weight='balanced' ):
    model = RandomForestClassifier( n_estimators=n_est, criterion=crit, n_jobs=-1,
                                   bootstrap=bootstrap, oob_score=oob_score, class_weight=class_weight )
    return model
def LR_Model( pen='l2', c=1, sol='saga', class_weight='balanced' ):
    if pen == 'elasticnet':
        sol = 'saga'
    model = LogisticRegression( penalty=pen, C=c, class_weight=class_weight, solver=sol, n_jobs=-1 )
    return model
def SVM_Model( c=1, ker='rbf', gam=0.5, max_iter=5000, class_weight='balanced' ):
    model = SVC( C=c, kernel=ker, gamma=gam, probability=True, class_weight=class_weight, max_iter=max_iter, verbose=True )
    return model

def CNN_George( tokenizer, max_len, embed_size=300, embedding_matrix=[], embed_trainable=False,
              emb_weights_init='', optimizer='', multi_gpu_flag=False, gpus=2 ):
    filter_sizes = [ 3,4,5 ]
    num_filters = 32
    inp = Input( shape=(max_len, ), name='InputLayer' )
    if embedding_matrix == []:
        x = Embedding( input_dim=len(tokenizer.word_index)+1, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable, name='Embedding' )( inp )
    else:
        x = Embedding( input_dim=len(tokenizer.word_index)+1, output_dim=embed_size,
                      weights=[embedding_matrix], trainable=embed_trainable, name='Embedding' )( inp )
    x = Reshape( ( max_len, embed_size, 1 ), name='Reshape' )( x )
    pooled = [  ]
    for j,i in enumerate(filter_sizes):
        conv = Conv2D( num_filters, kernel_size=(i, embed_size), activation='relu', name='Conv2D_'+str(j) )( x )
        globalmax = GlobalMaxPooling2D( name='GlobalMaxPool2D'+str(j) )( conv )
        pooled.append( globalmax )
    z = Concatenate( axis=1, name='ConcatenateLayer' )( pooled )
    z = Dense( 2, name='FC' )( z )
    outp = Softmax( name='SoftmaxLayer' )( z )
    model = Model( inputs=inp, outputs=outp )
    
    if multi_gpu_flag:
        model = multi_gpu_model(model, gpus=gpus)
    
    model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'] )
    return model

def focal_loss(gamma=1., alpha=.1):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed
def BGRU_P( tokenizer, max_len, embed_size=300, embedding_matrix=[], embed_trainable=False,
              spdrpt=.1, drpt=.1, emb_weights_init='', fc_weights_init='', fc_act='',
              optimizer='', fcl_loss_alp=.1, fcl_loss_gam=1, multi_gpu_flag=False, gpus=2 ):
    inp = Input( shape=(max_len, ), name='InputLayer' )
    if embedding_matrix == []:
        x = Embedding( input_dim=len(tokenizer.word_index)+1, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable, name='Embedding' )( inp )
    else:
        x = Embedding( input_dim=len(tokenizer.word_index)+1, output_dim=embed_size,
                      weights=[embedding_matrix], trainable=embed_trainable, name='Embedding' )( inp )
    x = SpatialDropout1D( spdrpt, name='SpatialDropout1D' )( x )
    x1 = Bidirectional( CuDNNGRU( 128, return_sequences=True ), name='BiGRU1' )(x)
    x2 = Bidirectional( CuDNNGRU( 64, return_sequences=True ), name='BiGRU2' )(x)

    conc = concatenate( [ x1, x2 ] )

    pooled = [  ]

    avg_pool = GlobalAveragePooling1D( name='GlobalAvgPool' )(conc)
    max_pool = GlobalMaxPooling1D( name='GlobalMaxPool' )(conc)

    pooled.append( avg_pool )
    pooled.append( max_pool )

    x = Concatenate( axis=1, name='ConcatenateLayer' )( pooled )

    x = Dropout( drpt, name='DropoutConcat' ) ( x )

    fc1 = Dense( 100, activation=fc_act, kernel_initializer=fc_weights_init, name='FC1' )( x )
    fc2 = Dense( 50, activation=fc_act, kernel_initializer=fc_weights_init, name='FC2' )( fc1 )
    outp = Dense( 1, activation='sigmoid', name='Output' )( fc2 )
    model = Model( inputs=inp, outputs=outp )
    
    if multi_gpu_flag:
        model = multi_gpu_model(model, gpus=gpus)
    
    model.compile( loss=[focal_loss(alpha=fcl_loss_alp, gamma=fcl_loss_gam)],
                  optimizer=optimizer, metrics=['accuracy'] )
    return model

def CNN_GRU( tokenizer, max_len, embed_size=300, embedding_matrix=[], embed_trainable=False,
            drpt=.1, emb_weights_init='', optimizer='', ker_regularizer='', multi_gpu_flag=False, gpus=2 ):
    inp = Input( shape=(max_len, ), name='InputLayer' )
    if embedding_matrix == []:
        x = Embedding( input_dim=len(tokenizer.word_index)+1, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable, name='Embedding' )( inp )
    else:
        x = Embedding( input_dim=len(tokenizer.word_index)+1, output_dim=embed_size,
                      weights=[embedding_matrix], trainable=embed_trainable, name='Embedding' )( inp )
    x = Dropout( drpt, name='EmbDropout' )( x )
    conv = Conv1D( filters=100, kernel_size=4, padding='same', activation='relu',
                  kernel_regularizer=ker_regularizer, name='conv1' )( x )
    pool = MaxPool1D( pool_size=4, name='pool1' )( conv )
    gru = CuDNNGRU( 100, return_sequences=True, name='GRU' )( pool )
    z = GlobalMaxPooling1D( name='globalmaxpooling' )( gru )
    z = Dense( 2, kernel_regularizer=ker_regularizer, name='Output' )( z )
    outp = Softmax( name='SoftmaxLayer' )( z )
    model = Model( inputs=inp, outputs=outp )
    
    if multi_gpu_flag:
        model = multi_gpu_model(model, gpus=gpus)
    
    model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'] )
    return model

def BLSTM( tokenizer, max_len, embed_size=300, embedding_matrix=[], embed_trainable=False,
              spdrpt=.1, drpt=.1, emb_weights_init='', fc_weights_init='', fc_act='',
              optimizer='', lstm_units=32, multi_gpu_flag=False, gpus=2 ):
    inp = Input( shape=(max_len, ), name='InputLayer' )
    if embedding_matrix == []:
        x = Embedding( input_dim=len(tokenizer.word_index)+1, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable, name='Embedding' )( inp )
    else:
        x = Embedding( input_dim=len(tokenizer.word_index)+1, output_dim=embed_size,
                      weights=[embedding_matrix], trainable=embed_trainable, name='Embedding' )( inp )
    x = SpatialDropout1D( spdrpt, name='SpatialDropout1D' )( x )
    z = Bidirectional( CuDNNLSTM( lstm_units ), name='BLSTM' )( x )
    z = Dropout( drpt, name='Dropout' )( z )
    fc1 = Dense( 100, activation=fc_act, kernel_initializer=fc_weights_init, name='FC1' )( z )
    fc2 = Dense( 50, activation=fc_act, kernel_initializer=fc_weights_init, name='FC2' )( fc1 )
    outp = Dense( 1, activation='sigmoid', name='Output' )( fc2 )
    model = Model( inputs=inp, outputs=outp )

    if multi_gpu_flag:
        model = multi_gpu_model( model, gpus=gpus )

    model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'] )
    return model

def BGRU( tokenizer, max_len, embed_size=300, embedding_matrix=[], embed_trainable=False,
              spdrpt=.1, drpt=.1, emb_weights_init='', fc_weights_init='', fc_act='',
              optimizer='', gru_units=32, multi_gpu_flag=False, gpus=2 ):
    inp = Input( shape=(max_len, ), name='InputLayer' )
    if embedding_matrix == []:
        x = Embedding( input_dim=len(tokenizer.word_index)+1, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable, name='Embedding' )( inp )
    else:
        x = Embedding( input_dim=len(tokenizer.word_index)+1, output_dim=embed_size,
                      weights=[embedding_matrix], trainable=embed_trainable, name='Embedding' )( inp )
    x = SpatialDropout1D( spdrpt, name='SpatialDropout1D' )( x )
    z = Bidirectional( CuDNNGRU( gru_units ), name='BGRU' )( x )
    z = Dropout( drpt, name='Dropout' )( z )
    fc1 = Dense( 100, activation=fc_act, kernel_initializer=fc_weights_init, name='FC1' )( z )
    fc2 = Dense( 50, activation=fc_act, kernel_initializer=fc_weights_init, name='FC2' )( fc1 )
    outp = Dense( 1, activation='sigmoid', name='Output' )( fc2 )
    model = Model( inputs=inp, outputs=outp )

    if multi_gpu_flag:
        model = multi_gpu_model( model, gpus=gpus )

    model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'] )
    return model

def CNN_Tweaked( tokenizer, max_len, embed_size=300, embedding_matrix=[], embed_trainable=False,
                spdrpt=.1, drpt=.1, emb_weights_init='', conv_weights_init='', conv_act='',
                fc_weights_init='', fc_act='', optimizer='', multi_gpu_flag=False, gpus=2 ):
    filter_sizes = [ 1,2,3,4,5 ]
    num_filters = 32

    inp = Input( shape=(max_len, ), name='InputLayer' )
    if embedding_matrix == []:
        x = Embedding( input_dim=len(tokenizer.word_index)+1, output_dim=embed_size,
                      embeddings_initializer=emb_weights_init, trainable=embed_trainable, name='Embedding' )( inp )
    else:
        x = Embedding( input_dim=len(tokenizer.word_index)+1, output_dim=embed_size,
                      weights=[embedding_matrix], trainable=embed_trainable, name='Embedding' )( inp )
    x = SpatialDropout1D( spdrpt, name='SpatialDropout1D' )( x )
    x = Reshape( ( max_len, embed_size, 1 ), name='Reshape' )( x )
    pooled = [  ]
    for j,i in enumerate(filter_sizes):
        conv = Conv2D( num_filters, kernel_size=(i, embed_size), activation=conv_act, name='Conv2D_'+str(j) )( x )
        globalmax = GlobalMaxPooling2D( name='GlobalMaxPool2D'+str(j) )( conv )
        pooled.append( globalmax )
    z = Concatenate( axis=1, name='ConcatenateLayer' )( pooled )
    z = Dropout( drpt, name='DropoutConcat' )( z )
    fc1 = Dense( 100, activation=fc_act, kernel_initializer=fc_weights_init, name='FC1' )( z )
    fc2 = Dense( 50, activation=fc_act, kernel_initializer=fc_weights_init, name='FC2' )( fc1 )
    outp = Dense( 1, activation='sigmoid', name='Output' )( fc2 )
    model = Model( inputs=inp, outputs=outp )
    
    if multi_gpu_flag:
        model = multi_gpu_model(model, gpus=gpus)
    
    model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'] )
    return model
