from GraphConvertor.weightBuffers_pb2 import WeightsData
import gc
import numpy as np

def read_pb(pb_file):
    weights = WeightsData()
    with open(pb_file, 'rb') as f:
        weights.ParseFromString(f.read())

    gc.collect()
    weights_data = {}

    for vec in weights.IntegerWeights:
        weights_data[vec.name] = np.array(vec.values, dtype=np.int32)
    
    for vec in weights.FloatWeights:
        weights_data[vec.name] = np.array(vec.values, dtype=np.float32)
    
    for int_const in weights.IntConstants:
        weights_data[int_const.name] = int_const.values

    for float_const in weights.FloatConstants:
        weights_data[float_const.name] = float_const.values

    for bool_const in weights.boolConstants:
        weights_data[bool_const.name] = bool_const.values
    
    del weights
    gc.collect()
    return weights_data
            
        # # Organize vectors by type
        # int_vectors = {vec.name: list(vec.values) for vec in collection.int_vectors}
        # float_vectors = {vec.name: list(vec.values) for vec in collection.float_vectors}
        # string_vectors = {vec.name: list(vec.values) for vec in collection.string_vectors}
        
