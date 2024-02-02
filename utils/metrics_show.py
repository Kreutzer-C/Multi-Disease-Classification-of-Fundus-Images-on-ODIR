
def show_metrics(acc, precision, recall, f1, num):
    name = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    print(">>>======Val/Test Metrics======<<<")
    for i in range(len(precision)):
        print("Class_Name:{}  Num:{}  |  Accuracy:{:.4f} Precision:{:.4f} Recall:{:.4f} F1:{:.4f}".format(
            name[i], num[i], acc[i], precision[i], recall[i], f1[i]
        ))
    average_acc = sum(acc)/len(acc)
    average_f1 = sum(f1)/len(f1)
    print("***Average            |  Acc:{:.4f} F1:{:.4f}".format(average_acc,average_f1))
    print(">>>============================<<<")

    return average_acc, average_f1