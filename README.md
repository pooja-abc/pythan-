# pythan

! Pip install keras_tuner
   Important Tensorflow as if from Tensorflow import keras import numly as rap
 Fashion-mnist=keras datesets_mnist
 {}(train_image, train_lables),(test_images, test_tables)=Fashion_mnist_load_data() 
    Train_images=train_images/2550
     Test_images=test_images/2550 
 {}train_images {0}.shape
     (28, 28) 
 {}train_images=Train_images.reshape(len(train_images), 28,28,1) 

 dif build_model(hp):
   model=keras.sequential{(
    Keras. Layers. Conv 2D(
    Filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16) 
    Kernel=size=hp.choise('conv_1_kernel1,' value={3,5}), 
    Activations='relu'
    Input_shape=(28,28,1) 
 ), 
    Keras-Lagers. Conv2D(
    filters=hp.int('conv_2_filter, 'min_value=32, max_value=64, step=16), 
    Kernel_size=hp.choice('conv_2_kernel1', 
     Value={3,5}, 
     Activation='relu'

 ), 
    Keras. Layers. Flatten (), 
     Keras. Layers. Dense(
           Units=hp.choise('conv_2_kernel, values={3,5}), 
           Activation='repu
), 
  Keras. Layers. Flatten(), 
  Keras. Layers. Dense(
         Units=hp.Int('dense_1, units', min_value=32, max_value=128, step=16), 
         Activation='relu'
 ),
   keras.layer.dense(10, activation='softmax') #output layer
 }) 
   Model. Compile (optimizer=keras.optimizers.adam(hp.choice('learning_rate', values={1e-2, 1e-3})), 
                                          Loss='sparse_categorical_crossentronpy'
                                          Metrics={accuracy'}) 

  Returns model
{}  from kerastuner important Random search
    from kerastuner. Engine. Hyperparameters import hyperparameters

{} tuner_search=Random search (build_model1, 
                 Objective='val accuracy', 
                 max_traials=5, director=output', project_name="minst fansion ") 
{}tuner_search.seach(train_images, train_labels_labels; epochs=3, validation_split=0.1) 
     INFO:Tensorflow:oracle triggered exist
{}model=tuner_seach.get_best_model(num_models=1) {0}


{}model.summary() 

Model 'sequential

Layer(type                   output shape                param #__
-----------------------------------------------------------------------------------

Conv 2D(conv 2D)              (None,22,22,48)            76848
flattten(Flastten)            (None, 23232)                0
Dense(dense)                  (None, 80)                  18458649
Dense1(dense)                 (None, 10)
____________________________________________________________________________________

Total params:1,936, 938
Trainable param:1,936,938
Non_trainable:0

          












  
