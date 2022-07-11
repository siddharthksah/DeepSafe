import os


def get_reference():
    models_list = []
    reference_list = []
    for model in os.listdir("./models"):
        if model[-6:] == "_image" or model[-6:] == "_video":
            models_list.append(model)

    for model in models_list:
        text_file = str("./models/" + model + "/reference.txt")
        #print(text_file)

        f = open(text_file, 'r')
        variable = f.readline()
        #print(variable.split(",")[0])
        reference_list.append(variable.split(",")[0])
        reference_list.append(variable.split(",")[1])
        reference_list.append(variable.split(",")[2][:-1])

    return reference_list, len(reference_list)


#print(get_reference())