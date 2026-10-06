from lib.supervised.config import Config
from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
from train_sga_base import TrainSGABase
import torch

# -------------------------------------------------------------------------------------
# ------------------------------- BASELINE METHODS ---------------------------------
# -------------------------------------------------------------------------------------

class TrainSTTranAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.sga.sttran_ant import STTranAnt
        self._model = STTranAnt(mode=self._conf.mode,
                                attention_class_num=len(self._test_dataset.attention_relationships),
                                spatial_class_num=len(self._test_dataset.spatial_relationships),
                                contact_class_num=len(self._test_dataset.contacting_relationships),
                                obj_classes=self._test_dataset.object_classes,
                                enc_layer_num=self._conf.enc_layer,
                                dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._init_transformer_loss_function_heads()

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = False

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = True
        self._enable_ant_bb_subject_loss = False
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = True

    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.max_future)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        num_ff = self._conf.max_window
        num_cf = self._conf.baseline_context
        pred = self._model(entry, num_cf, num_ff)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_ant_loss(pred)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_baseline_evaluation_score(pred, gt)


class TrainSTTranGenAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.sga.sttran_gen_ant import STTranGenAnt

        self._model = STTranGenAnt(mode=self._conf.mode,
                                   attention_class_num=len(self._test_dataset.attention_relationships),
                                   spatial_class_num=len(self._test_dataset.spatial_relationships),
                                   contact_class_num=len(self._test_dataset.contacting_relationships),
                                   obj_classes=self._test_dataset.object_classes,
                                   enc_layer_num=self._conf.enc_layer,
                                   dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._init_transformer_loss_function_heads()

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = True

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = True
        self._enable_ant_bb_subject_loss = False
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = True

    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.max_future)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        num_ff = self._conf.max_window
        num_cf = self._conf.baseline_context
        pred = self._model(entry, num_cf, num_ff)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_gen_ant_loss(pred)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_baseline_evaluation_score(pred, gt)


class TrainDsgDetrAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._init_matcher()

    def init_model(self):
        from lib.supervised.sga.dsgdetr_ant import DsgDetrAnt

        self._model = DsgDetrAnt(mode=self._conf.mode,
                                 attention_class_num=len(self._test_dataset.attention_relationships),
                                 spatial_class_num=len(self._test_dataset.spatial_relationships),
                                 contact_class_num=len(self._test_dataset.contacting_relationships),
                                 obj_classes=self._test_dataset.object_classes,
                                 enc_layer_num=self._conf.enc_layer,
                                 dec_layer_num=self._conf.dec_layer).to(device=self._device)
        self._init_matcher()
        self._init_transformer_loss_function_heads()

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = False

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = True
        self._enable_ant_bb_subject_loss = False
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = True

    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.max_future)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        num_ff = self._conf.max_window
        num_cf = self._conf.baseline_context
        pred = self._model(entry, num_cf, num_ff)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_ant_loss(pred)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_baseline_evaluation_score(pred, gt)


class TrainDsgDetrGenAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._init_matcher()

    def init_model(self):
        from lib.supervised.sga.dsgdetr_gen_ant import DsgDetrGenAnt

        self._model = DsgDetrGenAnt(mode=self._conf.mode,
                                    attention_class_num=len(self._test_dataset.attention_relationships),
                                    spatial_class_num=len(self._test_dataset.spatial_relationships),
                                    contact_class_num=len(self._test_dataset.contacting_relationships),
                                    obj_classes=self._test_dataset.object_classes,
                                    enc_layer_num=self._conf.enc_layer,
                                    dec_layer_num=self._conf.dec_layer).to(device=self._device)
        self._init_matcher()
        self._init_transformer_loss_function_heads()

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = True

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = True
        self._enable_ant_bb_subject_loss = False
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = True

    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.max_future)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        num_ff = self._conf.max_window
        num_cf = self._conf.baseline_context
        pred = self._model(entry, num_cf, num_ff)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_gen_ant_loss(pred)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_baseline_evaluation_score(pred, gt)


# -------------------------------------------------------------------------------------
# ------------------------------- SCENE SAYER METHODS ---------------------------------
# -------------------------------------------------------------------------------------

class TrainODE(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.sga.scene_sayer_ode import SceneSayerODE

        self._model = SceneSayerODE(mode=self._conf.mode,
                                    attention_class_num=len(self._test_dataset.attention_relationships),
                                    spatial_class_num=len(self._test_dataset.spatial_relationships),
                                    contact_class_num=len(self._test_dataset.contacting_relationships),
                                    obj_classes=self._test_dataset.object_classes,
                                    max_window=self._conf.max_window).to(device=self._device)

        self._init_matcher()
        self._init_diffeq_loss_function_heads()

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = True

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = True
        self._enable_ant_bb_subject_loss = True
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = True

    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        entry["gt_annotation"] = gt_annotation
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size,self._conf.mode)
        entry["gt_annotation"] = gt_annotation
        pred = self._model(entry, True)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_scene_sayer_loss(pred, self._conf.ode_ratio)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_scene_sayer_evaluation_score(pred, gt)


class TrainSDE(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.sga.scene_sayer_sde import SceneSayerSDE

        self._model = SceneSayerSDE(mode=self._conf.mode,
                                    attention_class_num=len(self._test_dataset.attention_relationships),
                                    spatial_class_num=len(self._test_dataset.spatial_relationships),
                                    contact_class_num=len(self._test_dataset.contacting_relationships),
                                    obj_classes=self._test_dataset.object_classes,
                                    max_window=self._conf.max_window,
                                    brownian_size=self._conf.brownian_size).to(device=self._device)

        self._init_matcher()
        self._init_diffeq_loss_function_heads()

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = True

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = True
        self._enable_ant_bb_subject_loss = True
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = True

    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        entry["gt_annotation"] = gt_annotation
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        entry["gt_annotation"] = gt_annotation
        pred = self._model(entry, True)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_scene_sayer_loss(pred, self._conf.sde_ratio)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_scene_sayer_evaluation_score(pred, gt)

# -------------------------------------------------------------------------------------
# ------------------------------- COUPLED SGA METHODS ---------------------------------
# -------------------------------------------------------------------------------------

class TrainCoupledSGA(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._sc_loss = None

    def init_model(self):
        from lib.supervised.sga.coupled_sga import CoupledSGA
        from lib.supervised.sga.structural_consistency_loss import StructuralConsistencyLoss

        self._model = CoupledSGA(
            mode=self._conf.mode,
            attention_class_num=len(self._test_dataset.attention_relationships),
            spatial_class_num=len(self._test_dataset.spatial_relationships),
            contact_class_num=len(self._test_dataset.contacting_relationships),
            obj_classes=self._test_dataset.object_classes,
            max_window=self._conf.max_window,
            euler_step_size=getattr(self._conf, 'euler_step_size', 1.0),
            gat_heads=getattr(self._conf, 'gat_heads', 4)
        ).to(device=self._device)

        self._init_matcher()
        self._init_diffeq_loss_function_heads()

        # Initialize Structural Consistency Loss
        lambda_sc = getattr(self._conf, 'lambda_sc', 0.1)
        self._sc_loss = StructuralConsistencyLoss(
            spatial_num_classes=len(self._test_dataset.spatial_relationships),
            contacting_num_classes=len(self._test_dataset.contacting_relationships),
            lambda_sc=lambda_sc
        ).to(device=self._device)

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = True

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = True
        self._enable_ant_bb_subject_loss = True
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = True

    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        entry["gt_annotation"] = gt_annotation
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        entry["gt_annotation"] = gt_annotation
        pred = self._model(entry, True)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        ode_ratio = getattr(self._conf, 'ode_ratio', 1.0)
        losses = self.compute_scene_sayer_loss(pred, ode_ratio)

        # Delayed SC Loss: only apply after sc_start_epoch (curriculum learning)
        current_epoch = getattr(self, '_current_epoch', 0)
        sc_start_epoch = getattr(self._conf, 'sc_start_epoch', 0)
        apply_sc = current_epoch >= sc_start_epoch

        if apply_sc:
            # Add Structural Consistency Loss on anticipated distributions
            if "anticipated_spatial_distribution" in pred and "anticipated_contacting_distribution" in pred:
                ant_spatial = pred["anticipated_spatial_distribution"]
                ant_contact = pred["anticipated_contacting_distribution"]
                # Flatten window dimension: (window, N, C) -> (window*N, C)
                if ant_spatial.dim() == 3:
                    ant_spatial = ant_spatial.reshape(-1, ant_spatial.size(-1))
                    ant_contact = ant_contact.reshape(-1, ant_contact.size(-1))
                if ant_spatial.size(0) > 0:
                    losses["structural_consistency_loss"] = self._sc_loss(ant_spatial, ant_contact)

            # Also apply SC loss on observed distributions
            if "spatial_distribution" in pred and "contacting_distribution" in pred:
                obs_spatial = pred["spatial_distribution"]
                obs_contact = pred["contacting_distribution"]
                if obs_spatial.size(0) > 0:
                    losses["structural_consistency_loss_observed"] = self._sc_loss(obs_spatial, obs_contact)
        else:
            if current_epoch == getattr(self, '_last_sc_log_epoch', -1):
                pass
            else:
                print(f"\n[Epoch {current_epoch}] SC loss DISABLED (starts at epoch {sc_start_epoch})", flush=True)
                self._last_sc_log_epoch = current_epoch

        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_scene_sayer_evaluation_score(pred, gt)


# -------------------------------------------------------------------------------------

def main():
    conf = Config()
    # Enable cudnn.benchmark only for fixed-size input methods (ode/sde)
    # DO NOT enable for coupledsga - variable graph sizes cause constant re-benchmarking
    if conf.method_name in ("ode", "sde"):
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False

    if conf.method_name == "ode":
        evaluate_class = TrainODE(conf)
    elif conf.method_name == "sde":
        evaluate_class = TrainSDE(conf)
    elif conf.method_name == "sttran_ant":
        evaluate_class = TrainSTTranAnt(conf)
    elif conf.method_name == "sttran_gen_ant":
        evaluate_class = TrainSTTranGenAnt(conf)
    elif conf.method_name == "dsgdetr_ant":
        evaluate_class = TrainDsgDetrAnt(conf)
    elif conf.method_name == "dsgdetr_gen_ant":
        evaluate_class = TrainDsgDetrGenAnt(conf)
    elif conf.method_name == "coupledsga":
        evaluate_class = TrainCoupledSGA(conf)
    else:
        raise NotImplementedError

    evaluate_class.init_method_training()


if __name__ == "__main__":
    main()
