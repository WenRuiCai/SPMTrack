from trackit.models import ModelBuildingContext, ModelImplSuggestions
from trackit.models.backbone.builder import build_backbone
from trackit.miscellanies.pretty_format import pretty_format
from .sample_data_generator import build_sample_input_data_generator


def get_SPMTrack_build_context(config: dict):
    print('SPMTrack model config:\n' + pretty_format(config['model']))
    return ModelBuildingContext(lambda impl_advice: build_SPMTrack_model(config, impl_advice),
                                lambda impl_advice: get_SPMTrack_build_string(config['model']['type'], impl_advice),
                                build_sample_input_data_generator(config))


def build_SPMTrack_model(config: dict, model_impl_suggestions: ModelImplSuggestions):
    model_config = config['model']
    common_config = config['common']
    backbone = build_backbone(model_config['backbone'],
                              torch_jit_trace_compatible=model_impl_suggestions.torch_jit_trace_compatible)
    model_type = model_config['type']
    if model_type == 'dinov2':
        if model_impl_suggestions.optimize_for_inference:
            from .SPMTrack_full_finetune import SPMTrackBaseline_DINOv2
            model = SPMTrackBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                                         model_config['tmoe']['r'], model_config['tmoe']['alpha'],
                                         model_config['tmoe']['dropout'], model_config['tmoe']['use_rsexpert'],
                                         model_config['tmoe']['expert_nums'], model_config['tmoe']['init_method'],
                                         shared_expert=model_config['tmoe']['shared_expert'], route_compression=model_config['tmoe']['route_compression'])
        else:
            from .SPMTrack import SPMTrack_DINOv2
            model = SPMTrack_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                                 model_config['tmoe']['r'], model_config['tmoe']['alpha'],
                                 model_config['tmoe']['dropout'], model_config['tmoe']['use_rsexpert'],
                                 model_config['tmoe']['expert_nums'], model_config['tmoe']['init_method'],
                                 shared_expert=model_config['tmoe']['shared_expert'], route_compression=model_config['tmoe']['route_compression'])
    elif model_type == 'dinov2_full_finetune':
        from .SPMTrack_full_finetune import SPMTrackBaseline_DINOv2
        model = SPMTrackBaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")
    return model


def get_SPMTrack_build_string(model_type: str, model_impl_suggestions: ModelImplSuggestions):
    build_string = 'SPMTrack'
    if 'full_finetune' in model_type:
        build_string += '_full_finetune'
    else:
        if model_impl_suggestions.optimize_for_inference:
            build_string += '_merged'
    if model_impl_suggestions.torch_jit_trace_compatible:
        build_string += '_disable_flash_attn'
    return build_string
