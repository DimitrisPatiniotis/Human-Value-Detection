from transformers.integrations import TensorBoardCallback
# from pretty_confusion_matrix import pp_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

display_labels = None
filename_suffix = ""


class MTTensorBoardCallback(TensorBoardCallback):
    def _init_summary_writer(self, args, log_dir=None):
        print("==================================================")
        print("filename_suffix:", filename_suffix, self.tb_writer)
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writer = self._SummaryWriter(log_dir=log_dir+"-"+filename_suffix)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(self.tb_writer, "add_hparams"):
                self.tb_writer.add_hparams(args.to_sanitized_dict(), metric_dict={},
                                           run_name=".")

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
