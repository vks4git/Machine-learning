__author__ = 'vks'


def k_fold(k, data_x, data_y, model):
    result = 0.0
    chunk = len(data_x) // k
    for iter in range(k):
        learn_x = []
        learn_y = []
        validate_x = []
        validate_y = []
        for i in range(len(data_x)):
            if i // chunk == iter:
                validate_x.append(data_x[i])
                validate_y.append(data_y[i])
            else:
                learn_x.append(data_x[i])
                learn_y.append(data_y[i])
        model.fit(learn_x, learn_y)
        errors = 0
        for i in range(len(validate_x)):
            if model.predict(validate_x[i]) != validate_y[i]:
                errors += 1
        result += (len(validate_x) - errors) / len(validate_x)
    return result / k
