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
        line = line.translate(None,' ')
        line_fields = line.split(':')
        if 'C' in line:
            attribute = line_fields
        else:
            attribute = []
            attribute.append(line_fields[0])
            values_list = line_fields[1].split(',')
            if '?' in values_list[-1]: #for some reason we have
                values_list[-1] = values_list[-1].translate(None,'()?.\n ')
            if values_list[-1] == '':
                values_list = values_list [:-1]
            attribute.append(values_list)
        lst.append(attribute)
    return lst

