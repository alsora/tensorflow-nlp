def align_data(data):
    """Given dict with lists, creates aligned strings
    Adapted from Assignment 3 of CS224N
    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]
    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                             data_align["y"] = "O O    O  "
    """

    data_aligned = dict()
    
    # Sequence tagging task
    if (len(data['output']) == len(data['input'])):

        spacings = [max([len(seq[i]) for seq in data.values()])
                    for i in range(len(data[list(data.keys())[0]]))]

        # for each entry, create aligned string
        for key, seq in data.items():
            str_aligned = ""
            for token, spacing in zip(seq, spacings):
                str_aligned += token + " " * (spacing - len(token) + 1)

            data_aligned[key] = str_aligned
    
    # Text classification task
    elif (len(data['output']) == 1):

        data_aligned['input'] = data['input']
        data_aligned['output'] = data['output']

    # Sequence2Sequence task
    else:

        data_aligned['input'] = data['input']
        data_aligned['output'] = data['output']





    return data_aligned