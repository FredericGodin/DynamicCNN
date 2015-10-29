__author__ = 'Frederic Godin  (frederic.godin@ugent.be / www.fredericgodin.com)'

import numpy


def read_and_sort_matlab_data(x_file,y_file,padding_value=15448):


    sorted_dict = {}
    x_data = []
    i=0
    file = open(x_file,"r")
    for line in file:
        words = line.split(",")
        result = []
        length=None
        for word in words:
            word_i = int(word)
            if word_i == padding_value and length==None:
                length = len(result)
            result.append(word_i)
        x_data.append(result)

        if length==None:
            length=len(result)

        if length in sorted_dict:
            sorted_dict[length].append(i)
        else:
            sorted_dict[length]=[i]
        i+=1

    file.close()

    file = open(y_file,"r")
    y_data = []
    for line in file:
        words = line.split(",")
        y_data.append(int(words[0])-1)
    file.close()

    new_train_list = []
    new_label_list = []
    lengths = []
    for length, indexes in sorted_dict.items():
        for index in indexes:
            new_train_list.append(x_data[index])
            new_label_list.append(y_data[index])
            lengths.append(length)

    return numpy.asarray(new_train_list,dtype=numpy.int32),numpy.asarray(new_label_list,dtype=numpy.int32),lengths

def pad_to_batch_size(array,batch_size):
    rows_extra = batch_size - (array.shape[0] % batch_size)
    if len(array.shape)==1:
        padding = numpy.zeros((rows_extra,),dtype=numpy.int32)
        return numpy.concatenate((array,padding))
    else:
        padding = numpy.zeros((rows_extra,array.shape[1]),dtype=numpy.int32)
        return numpy.vstack((array,padding))

def extend_lenghts(length_list,batch_size):
    elements_extra = batch_size - (len(length_list) % batch_size)
    length_list.extend([length_list[-1]]*elements_extra)

