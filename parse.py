def read_training_file(train_file):
    X = []
    Y = []
    with open(train_file) as infile:
        line = infile.readline()
        while line:
            attributes = line.split()
            Y.append(attributes.pop())
            X.append(attributes)
            line = infile.readline()
    return X,Y 

def read_attribute_file(attr_file):
    '''read whether attribute is discrete or continuous'''
    lst = []
    infile = open(attr_file)
    for line in infile:
        print line
        line_fields = line.split(':')
        if 'C' in line:
            attribute = line_fields
        else:
            attribute = []
            attribute.append(line_fields[0])
            values_list = line_fields[1].split(',')
            no_question_list = values_list[:-1] # remove '?.'
            attribute.append(no_question_list)
        print attribute
        lst.append(attribute)
    return lst

