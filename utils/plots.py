import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def pie_chart(df_evaluated):
    correct = (df_evaluated['judge_score'] == df_evaluated['score']).sum()
    incorrect = (df_evaluated['judge_score'] != df_evaluated['score']).sum()
    plt.pie([correct, incorrect],
            labels=['Correct', 'Incorrect'],
            colors=['green', 'red'],
            autopct='%1.1f%%')
    plt.title('Accuracy Overview')
    plt.show()


def dist_plot(df_evaluated): 

    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    # Force include both 0 and 1 values even if they don't exist in the data
    true_counts = df_evaluated['score'].value_counts().reindex([0, 1], fill_value=0)
    true_counts.plot(kind='bar', color='salmon')
    plt.title('True Score Distribution')
    plt.xlabel('Score (0 or 1)')
    plt.ylabel('Frequency')
    plt.xticks(ticks=[0, 1], labels=[0, 1])

    plt.subplot(122)
    # Force include both 0 and 1 values even if they don't exist in the data
    judge_counts = df_evaluated['judge_score'].value_counts().reindex([0, 1], fill_value=0)
    judge_counts.plot(kind='bar', color='skyblue')
    plt.title('Selene-Mini Score Distribution')
    plt.xlabel('Selene-Mini Score (0 or 1)')
    plt.ylabel('Frequency')
    plt.xticks(ticks=[0, 1], labels=[0, 1])

    plt.tight_layout()
    plt.show()





def conf_mat(df_evaluated):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(df_evaluated['score'], df_evaluated['judge_score'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted (Selene-Mini Score)')
    plt.ylabel('Actual (True Score)')
    plt.show()