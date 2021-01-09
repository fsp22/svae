import matplotlib
matplotlib.use('PS')

import matplotlib.pyplot as plt


# CHART

def save_loss_chart(lossTrain, lossTest, lastHitRate, 
                    fname, labelTest='Validation', labelMetric='NDCG@10'):
    """
    Create a chart with loss per epoch
    :param lossTrain: list of loss value per epoch
    :param lossTest: list of loss value per epoch
    :param lastHitRate: list of hr value per epoch
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    ax1.plot(lossTrain, color='b', )
    #ax1.set_yscale('log')
    ax1.set_title('Train')
    ax2.plot(lossTest, color='r')
    #ax2.set_yscale('log')
    ax2.set_title(labelTest)
    ax3.plot(lastHitRate)
    ax3.set_title(labelMetric)
    plt.savefig(fname)
