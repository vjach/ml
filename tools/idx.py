import sys
import struct 
import collections
import numpy as np

def decode(filename):
    data_type_list = {0x08:('uint8', 'B', 1), 0x09:('int8', 'b', 1), 0x0B:('int16', 'h', 2), 0x0C:('int32', 'i', 4), 0x0D:('float32', 'f', 4), 0x0E:('float64', 'd', 8)}

    try:
        f = open(filename, 'rb')
        data = f.read(4)
    except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            return None

    (magic_id,) = struct.unpack('>H',data[:2]) 
    if magic_id != 0:
        raise ValueError('Invalid magic prefix:{}'.format(magic_id))
    
    (dtype_id, ) = struct.unpack('>B', data[2])

    if dtype_id not in data_type_list.keys():
        raise ValueError('Invalid data type: {}'.format(data_type))
    
    (no_dims, ) = struct.unpack('>B', data[3])

    try:
        data = f.read(no_dims * 4)
        dec_str = '>' + 'I' * no_dims
        ddim_size = struct.unpack(dec_str, data)
    except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)

    (dtype, ddec, el_size) = data_type_list[dtype_id]
    result_len =  np.prod(np.array(ddim_size))
    result = np.empty(shape = [result_len], dtype=dtype)
    
    try:
        raw_len = result_len * el_size
        data = f.read(raw_len)
        dec_str = '>' + ddec * raw_len
        result[:result_len] = struct.unpack(dec_str, data)
    except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)

    result = np.reshape(result, ddim_size)
    return result

    
