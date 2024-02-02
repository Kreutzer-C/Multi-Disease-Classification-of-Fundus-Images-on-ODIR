from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def calculate_metrics(predictions, labels, threshold):
    precision_list = []
    recall_list = []
    f1_list = []
    spt_list = []
    acc_list = []

    binary_predictions = (predictions > threshold).astype(int)
    for i in range(labels.shape[1]):
        pred_class_i = binary_predictions[:,i]
        label_class_i = labels[:,i]

        acc = accuracy_score(label_class_i, pred_class_i)
        acc_list.append(acc)

        precision, recall, f1_score, support = precision_recall_fscore_support(label_class_i, pred_class_i, labels=[1], zero_division="warn")
        precision_list.append(precision[0])
        recall_list.append(recall[0])
        f1_list.append(f1_score[0])
        spt_list.append(support[0])

    return acc_list, precision_list, recall_list, f1_list, spt_list