import keras
import keras.backend as K
from distutils.version import LooseVersion, StrictVersion
from collections import OrderedDict

from st_core import Structure

_keras_var_class = K.variable(0).__class__

class KerasLayerWrapper(object):
    def __init__(self,keras_layer,keras_model_wrapper=None,root_layer=None):
        self.keras_layer = keras_layer
        self.keras_model_wrapper = keras_model_wrapper
        
        if root_layer is None:
            root_layer = self
            
        self.root_layer = root_layer
        
        self.from_keras()
        
        return
    
    def __getattr__(self,name):
        return getattr(self.keras_layer,name)
    
    def from_keras(self):
        
        return
    
    @property
    def _inbound_nodes(self):
        if StrictVersion(keras.__version__)<StrictVersion("2.1.3"):
            return self.keras_layer.inbound_nodes
        else:
            return self.keras_layer._inbound_nodes
        return
    
    def get_weights_tensors(self):
        weights = {}
        kl = self.keras_layer
        for w_name in kl.__dict__:
            if isinstance(kl.__dict__[w_name],_keras_var_class):
                    weights[w_name] = kl.__dict__[w_name]
        
        weights = OrderedDict([(_key,weights[_key]) for _key in sorted(weights.keys())])
            
        return weights
    
    def get_trainable_weights_tensors(self):
        weights = self.get_weights_tensors()
        return OrderedDict([(_w_name,_w) for _w_name,_w in weights.iteritems()])
    
    def get_weights_gradients_tensors(self):
        weights = Structure(self.get_trainable_weights_tensors())
        
        grads = Structure(K.gradients(self.keras_model_wrapper.total_loss,weights.to_list()))
        
        return grads.reshape(weights.shape).object
    
    def get_inputs_tensors(self):
        return [ [_tens for _tens in node.input_tensors if _tens not in self.keras_model_wrapper.inputs]
                for node in self._inbound_nodes]
    
    def get_outputs_tensors(self):
        return [ [_tens for _tens in node.output_tensors if _tens not in self.keras_model_wrapper.inputs]
                for node in self._inbound_nodes]
    
    @property
    def inputs_layers(self):
        return [node.inbound_layers for node in self._inbound_nodes]
    
    @property
    def outputs_layers(self):
        return [node.outbound_layer for node in self._inbound_nodes]


class KerasModelWrapper(object):
    def __init__(self,keras_model):
        self.keras_model = keras_model
        self.layers = None
        self.layers_names = []
        
        #===============================
        # get_tensors functions results
        #===============================
        self._layers_outputs_tensors = None
        self._layers_inputs_tensors = None
        self._layers_weights_tensors = None
        self._layers_trainable_weights_tensors = None
        self._layers_weights_gradients_tensors = None
        
        #===============================
        # compiled Keras functions
        #===============================
        self._grad_func = None
        self._inp_func = None
        self._out_func = None
        
        self.from_keras()
        
        return
            
    def __getitem__(self,name):
        return self.layers[name]
    
    def __getattr__(self,name):
        return getattr(self.keras_model,name)
    
    def _read_keras_layer(self,lr,lr_dict = None,root_layer = None):
        if lr_dict is None:
            lr_dict = OrderedDict()
        
        lr_dict[lr.name] = KerasLayerWrapper(lr,root_layer=root_layer,keras_model_wrapper=self)
        
        if root_layer is None:
            root_layer = lr_dict[lr.name]
        
        if "cell" in lr.__dict__:
            self._read_keras_layer(lr.cell,lr_dict,
                                               root_layer=root_layer)
            
        if "layer" in lr.__dict__:
            self._read_keras_layer(lr.layer,lr_dict,
                                               root_layer=root_layer)
        
        return lr_dict
    
    def from_keras(self):
        # read self.keras_model and fill inner fields
        model = self.keras_model
        
        # fill layers
        self.layers = OrderedDict()
        layers = self.layers
        for wrap_lr in model.layers:
            _lr_dict = self._read_keras_layer(wrap_lr)

            if set(layers.keys()) & set(_lr_dict.keys()):
                for _l_name in set(layers.keys()) & set(_lr_dict.keys()):
                    # change keys in _lr_dict
                    _counter = 0
                    _new_l_name = None
                    while True:
                        _counter += 1
                        _new_l_name = _l_name+"$"+str(_counter)
                        if _new_l_name not in layers.keys():
                            break
                            
                    _lr_dict[_new_l_name] = _lr_dict[_l_name]
                    del _lr_dict[_l_name]
            
            # merge _lr_dict and layers
            layers.update(_lr_dict)
            
        # fill layers names
        self.layers_names = self.layers.keys()
        self.layers_names.sort()
            
        return
    
    #=====================================================
    # Functions to get tensors
    #    TODO: Create memoization decorator
    #=====================================================    
    
    def get_layers_outputs_tensors(self):
        if not self._layers_outputs_tensors:
            layers = self.layers
            self._layers_outputs_tensors = \
                OrderedDict([(l_name,layers[l_name].get_outputs_tensors()) for l_name in layers])
        return self._layers_outputs_tensors
        
    def get_layers_inputs_tensors(self):
        if not self._layers_inputs_tensors:
            layers = self.layers
            self._layers_inputs_tensors = \
                OrderedDict([(l_name,layers[l_name].get_inputs_tensors()) for l_name in layers])
        return self._layers_inputs_tensors

    def get_layers_weights_gradients_tensors(self):
        if not self._layers_weights_gradients_tensors:
            layers = self.layers
            self._layers_weights_gradients_tensors = \
                OrderedDict([(l_name,layers[l_name].get_weights_gradients_tensors()) for l_name in layers])
        return self._layers_weights_gradients_tensors
    
    def get_layers_weights_tensors(self):
        if not self._layers_weights_tensors:            
            layers = self.layers
            self._layers_weights_tensors = \
                OrderedDict([(l_name,layers[l_name].get_weights_tensors()) for l_name in layers])
        return self._layers_weights_tensors
    
    def get_layers_trainable_weights_tensors(self):
        if not self._layers_trainable_weights_tensors:
            layers = self.layers
            self._layers_trainable_weights_tensors = \
                OrderedDict([(l_name,layers[l_name].get_trainable_weights_tensors()) for l_name in layers])
        return self._layers_trainable_weights_tensors
        
    #=====================================================
    # Functions to get values
    #=====================================================
    
    def get_layers_weights(self):
        weights_tens = Structure(self.get_layers_weights_tensors())
        vals = Structure(K.batch_get_value(weights_tens.to_list()))
        return vals.reshape(weights_tens.shape).object
    
    def get_layers_trainable_weights(self):
        weights_tens = Structure(self.get_layers_trainable_weights_tensors())
        vals = Structure(K.batch_get_value(weights_tens.to_list()))
        return vals.reshape(weights_tens.shape).object
    
    def get_layers_outputs(self,data):
        data = data if isinstance(data,list) else [data]
        outputs = Structure(self.get_layers_outputs_tensors())
        if not self._out_func:
            # create output function
            _inp_tens = []
            _inp_tens.extend(self.inputs if isinstance(self.inputs,list) else [self.inputs])
            _out_tens = outputs.to_list()
            self._out_func = K.function(inputs=_inp_tens,outputs=_out_tens)
        
        return Structure(self._out_func(data)).reshape(outputs.shape).object
    
    def get_layers_inputs(self,data):
        data = data if isinstance(data,list) else [data]
        inputs = Structure(self.get_layers_inputs_tensors())
        if not self._inp_func:
            # create output function
            _inp_tens = []
            _inp_tens.extend(self.inputs if isinstance(self.inputs,list) else [self.inputs])
            _out_tens = inputs.to_list()
            self._inp_func = K.function(inputs=_inp_tens,outputs=_out_tens)
        
        return Structure(self._inp_func(data)).reshape(inputs.shape).object
    
    def get_layers_weights_gradients(self,data,labels):
        data = data if isinstance(data,list) else [data]
        labels = labels if isinstance(labels,list) else [labels]
        grads = Structure(self.get_layers_weights_gradients_tensors())
        if not self._grad_func:
            # create output function
            _inp_tens = []
            _inp_tens.extend(self.inputs if isinstance(self.inputs,list) else [self.inputs])
            _inp_tens.extend(self.sample_weights)
            _inp_tens.extend(self.targets)
            _inp_tens.append(K.learning_phase())
            
            _out_tens = grads.to_list()
            self._grad_func = K.function(inputs=_inp_tens,outputs=_out_tens)
        
        _inp_val = []
        _inp_val.extend(data)
        _inp_val.append(len(data[0])*[1.0])
        _inp_val.extend(labels)
        _inp_val.append(0) # learning phase
        return Structure(self._grad_func(_inp_val)).reshape(grads.shape).object
    
    #==================
    # Other functions
    #==================
    
    def compile(self,*args,**kwargs):
        
        #===============================
        # get_tensors functions results
        #===============================
        self._layers_outputs_tensors = None
        self._layers_inputs_tensors = None
        self._layers_weights_tensors = None
        self._layers_trainable_weights_tensors = None
        self._layers_weights_gradients_tensors = None
        
        #===============================
        # compiled Keras functions
        #===============================
        self._grad_func = None
        self._inp_func = None
        self._out_func = None
        
        self.keras_model.compile(*args,**kwargs)
        return
    
    def get_SVG():
        return
    
    def pack_weights():
        return
    
    def pack_gradients():
        return
