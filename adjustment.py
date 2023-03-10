from predict import *
from random import uniform


def random_parameter(test_normal_loader, test_abnormal_loader, model, opt, min_score):
    TP = 0
    FP = 0
    batch = opt.batch

    # Test the model
    with torch.no_grad():
        for i in range(0, len(test_normal_loader), batch):
            right = min(len(test_normal_loader), i + batch)
            anomaly_num, _ = predict_anomaly_num(model, test_normal_loader[i:right], opt, min_score)
            FP += anomaly_num
    with torch.no_grad():
        for i in range(0, len(test_abnormal_loader), batch):
            right = min(len(test_abnormal_loader), i + batch)
            anomaly_num, _ = predict_anomaly_num(model, test_abnormal_loader[i:right], opt, min_score)
            TP += anomaly_num
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / max((TP + FP), 1)
    R = 100 * TP / max((TP + FN), 1)
    F1 = 2 * P * R / max((P + R), 1)
    print('min_score: ' + str(min_score) + ' F1: ' + str(F1) + ' Precision: ' + str(P) + ' Recall: ' + str(R))
    return min_score, F1


def adjustment(opt, model, low, up):
    print('Parameter adjustment')
    test_normol = './data/' + opt.dataset + '/Dev/dev_normal'
    test_abnormol = './data/' + opt.dataset + '/Dev/dev_abnormal'
    model_file = 'models/' + opt.dataset + '/' + model
    window_size = opt.window_size
    test_normal_loader = generate(test_normol, window_size)
    test_abnormal_loader = generate(test_abnormol, window_size)
    model = get_model(opt, opt.window_size, opt.num_class, False, model_file=model_file)
    model.eval()
    best_min_score = 0
    best_F1 = 0
    flag = True
    # Run min_score randomly given
    for i in range(opt.rands):
        if flag:
            min_score, F1 = random_parameter(test_normal_loader, test_abnormal_loader, model, opt,
                                             opt.min_score)  # First try with preset hyperparameters
            flag = False
        else:
            min_score, F1 = random_parameter(test_normal_loader, test_abnormal_loader, model, opt, uniform(low, up))
        if F1 > best_F1:
            best_min_score = min_score
            best_F1 = F1
    print('best_F1: ' + str(best_F1) + ' best_parameters: ' + str(best_min_score))
    return best_min_score