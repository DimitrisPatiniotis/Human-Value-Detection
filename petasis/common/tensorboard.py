from transformers.integrations import TensorBoardCallback
# from pretty_confusion_matrix import pp_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

display_labels = None

class MTTensorBoardCallback(TensorBoardCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # print("MTTensorBoardCallback")
        # print("logs:", logs)
        logs_copy = {}
        for k, v in logs.items():
            if not k.endswith("_mcm"):
                logs_copy[k] = v
        super().on_log(args, state, control, logs_copy, **kwargs)
        for k, v in logs.items():
            if k.endswith("_mcm"):
                figure = self.generate_cm_grid(v)
                self.tb_writer.add_figure(k, figure, global_step=int(logs["epoch"]))
        self.tb_writer.flush()

    def generate_cm_grid(self, mcm):
        f, axes = plt.subplots(4, 5, figsize=(25, 15))
        axes = axes.ravel()
        for i, cm in enumerate(mcm):
            disp = ConfusionMatrixDisplay(np.array(cm), display_labels=[0, i+1])
            disp.plot(ax=axes[i], values_format='.4g')
            disp.ax_.set_title(f'{display_labels[i]} ({i+1})')
            if i<15:
                disp.ax_.set_xlabel('')
            if i%5!=0:
                disp.ax_.set_ylabel('')
            disp.im_.colorbar.remove()
        plt.subplots_adjust(wspace=0.10, hspace=0.2)
        f.colorbar(disp.im_, ax=axes)
        return f
