from __future__ import annotations

from functools import wraps
import inspect
import sys
from typing import Any, TypedDict, cast


def _build_default_fast_image_processor_kwargs() -> Any:
    from transformers.processing_utils import ImagesKwargs

    typed_dict_factory = cast(Any, TypedDict)
    return typed_dict_factory(
        "DefaultFastImageProcessorKwargs",
        dict(ImagesKwargs.__annotations__),
        total=False,
    )


def _is_eagle3_dynamic_processor_class(processor_cls: object) -> bool:
    if not isinstance(processor_cls, type):
        return False
    if getattr(processor_cls, "__name__", "") != "Eagle3_VLProcessor":
        return False
    module = sys.modules.get(getattr(processor_cls, "__module__", ""))
    module_file = str(getattr(module, "__file__", "") or "")
    return module_file.endswith("processing_eagle3_vl.py")


def _is_eagle3_dynamic_fast_image_processor_class(processor_cls: object) -> bool:
    if not isinstance(processor_cls, type):
        return False
    if getattr(processor_cls, "__name__", "") != "Eagle3_VLImageProcessorFast":
        return False
    module = sys.modules.get(getattr(processor_cls, "__module__", ""))
    module_file = str(getattr(module, "__file__", "") or "")
    return module_file.endswith("image_processing_eagle3_vl_fast.py")


def _get_callable_signature(callable_obj: Any) -> inspect.Signature | None:
    try:
        return inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return None


def _filter_kwargs_for_callable(
    callable_obj: Any,
    kwargs: dict[str, Any],
    default_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signature = _get_callable_signature(callable_obj)
    if signature is None:
        return dict(kwargs)

    merged_kwargs = dict(kwargs)
    if default_kwargs:
        merged_kwargs = {**default_kwargs, **merged_kwargs}

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return merged_kwargs

    accepted_kwargs = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {
        name: value for name, value in merged_kwargs.items() if name in accepted_kwargs
    }


def _call_with_supported_kwargs(
    callable_obj: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    filtered_kwargs = _filter_kwargs_for_callable(callable_obj, kwargs)
    return callable_obj(*args, **filtered_kwargs)


def _call_with_supported_and_default_kwargs(
    callable_obj: Any,
    default_kwargs: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    filtered_kwargs = _filter_kwargs_for_callable(
        callable_obj,
        kwargs,
        default_kwargs=default_kwargs,
    )
    return callable_obj(*args, **filtered_kwargs)


def _patch_eagle3_fast_image_processor_prepare_input_images(
    processor_cls: object,
) -> None:
    if not isinstance(processor_cls, type):
        return

    if getattr(processor_cls, "_gr00t_prepare_input_images_compat_installed", False):
        return

    if hasattr(processor_cls, "_prepare_input_images"):
        setattr(processor_cls, "_gr00t_prepare_input_images_compat_installed", True)
        return

    def _compat_prepare_input_images(
        self: Any,
        images: Any,
        do_convert_rgb: bool | None = None,
        input_data_format: Any = None,
        device: Any = None,
        **kwargs: Any,
    ) -> Any:
        process_image = getattr(self, "_process_image", None)
        prepare_images_structure = getattr(self, "_prepare_images_structure", None)
        if not callable(process_image) or not callable(prepare_images_structure):
            raise AttributeError(
                f"{type(self).__name__} is missing the legacy _prepare_images_structure/_process_image helpers"
            )

        prepared_images = _call_with_supported_kwargs(
            prepare_images_structure,
            images,
            **kwargs,
        )
        process_image_kwargs = {
            "do_convert_rgb": do_convert_rgb,
            "input_data_format": input_data_format,
            "device": device,
        }
        return [
            _call_with_supported_kwargs(
                process_image,
                image,
                **process_image_kwargs,
            )
            for image in prepared_images
        ]

    setattr(processor_cls, "_prepare_input_images", _compat_prepare_input_images)
    setattr(processor_cls, "_gr00t_prepare_input_images_compat_installed", True)


def _patch_eagle3_fast_image_processor_grouping_helpers(
    processor_cls: object,
) -> None:
    if not isinstance(processor_cls, type):
        return

    module = sys.modules.get(processor_cls.__module__)
    if module is None or getattr(
        module, "_gr00t_grouping_helpers_compat_installed", False
    ):
        return

    group_images_by_shape = getattr(module, "group_images_by_shape", None)
    reorder_images = getattr(module, "reorder_images", None)

    if callable(group_images_by_shape):

        @wraps(group_images_by_shape)
        def _compat_group_images_by_shape(*args: Any, **kwargs: Any) -> Any:
            return _call_with_supported_and_default_kwargs(
                group_images_by_shape,
                {"disable_grouping": None},
                *args,
                **kwargs,
            )

        setattr(module, "group_images_by_shape", _compat_group_images_by_shape)

    if callable(reorder_images):

        @wraps(reorder_images)
        def _compat_reorder_images(*args: Any, **kwargs: Any) -> Any:
            return _call_with_supported_kwargs(reorder_images, *args, **kwargs)

        setattr(module, "reorder_images", _compat_reorder_images)

    setattr(module, "_gr00t_grouping_helpers_compat_installed", True)


def _patch_eagle3_fast_image_processor_preprocess(processor_cls: object) -> None:
    if not isinstance(processor_cls, type):
        return

    if getattr(processor_cls, "_gr00t_preprocess_resample_compat_installed", False):
        return

    def _compat_preprocess(
        self: Any, images: Any, videos: Any = None, **kwargs: Any
    ) -> Any:
        from transformers.image_utils import PILImageResampling

        validate_kwargs = getattr(
            sys.modules.get(processor_cls.__module__), "validate_kwargs"
        )
        pil_torch_interpolation_mapping = getattr(
            sys.modules.get(processor_cls.__module__), "pil_torch_interpolation_mapping"
        )

        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=self.valid_kwargs.__annotations__.keys(),
        )

        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")

        if images is not None:
            images = _call_with_supported_kwargs(
                self._prepare_input_images,
                images=images,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        if videos is not None:
            videos = _call_with_supported_kwargs(
                self._prepare_input_images,
                images=videos,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        resample = kwargs.get("resample", getattr(self, "resample", None))
        kwargs = self._further_process_kwargs(**kwargs)
        self._validate_preprocess_kwargs(**kwargs)

        if "interpolation" not in kwargs and resample is not None:
            kwargs["interpolation"] = (
                pil_torch_interpolation_mapping[resample]
                if isinstance(resample, (PILImageResampling, int))
                else resample
            )
        kwargs.pop("resample", None)

        if images is not None:
            return _call_with_supported_kwargs(self._preprocess, images, **kwargs)
        if videos is not None:
            return _call_with_supported_kwargs(self._preprocess, videos, **kwargs)
        return None

    _patch_eagle3_fast_image_processor_grouping_helpers(processor_cls)
    setattr(processor_cls, "preprocess", _compat_preprocess)
    setattr(processor_cls, "_gr00t_preprocess_resample_compat_installed", True)


def _patch_eagle3_processor_from_args_and_dict(processor_cls: object) -> None:
    if not isinstance(processor_cls, type):
        return

    if getattr(processor_cls, "_gr00t_from_args_and_dict_compat_installed", False):
        return

    original_from_args_and_dict = getattr(processor_cls, "from_args_and_dict")
    original_from_args_and_dict_func = getattr(
        original_from_args_and_dict, "__func__", original_from_args_and_dict
    )

    @wraps(original_from_args_and_dict_func)
    def _compat_from_args_and_dict(
        cls: type[Any], args: Any, processor_dict: dict[str, Any], **kwargs: Any
    ) -> Any:
        processor_dict = processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        if "processor_class" in processor_dict:
            del processor_dict["processor_class"]

        validate_result = cls.validate_init_kwargs(
            processor_config=processor_dict,
            valid_kwargs=cls.valid_kwargs,
        )
        if isinstance(validate_result, tuple) and len(validate_result) == 2:
            unused_kwargs, valid_kwargs = validate_result
            init_kwargs = dict(valid_kwargs)
        else:
            unused_kwargs = validate_result
            init_kwargs = processor_dict

        processor = cls(*args, **init_kwargs)

        for key in set(kwargs.keys()):
            if hasattr(processor, key):
                setattr(processor, key, kwargs.pop(key))

        kwargs.update(dict(unused_kwargs))

        logger = getattr(sys.modules.get(cls.__module__), "logger", None)
        if logger is not None:
            logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, kwargs
        return processor

    setattr(
        processor_cls,
        "from_args_and_dict",
        classmethod(_compat_from_args_and_dict),
    )
    setattr(processor_cls, "_gr00t_from_args_and_dict_compat_installed", True)


def _patch_loaded_eagle3_dynamic_processors() -> None:
    for module in tuple(sys.modules.values()):
        processor_cls = getattr(module, "Eagle3_VLProcessor", None)
        _patch_eagle3_dynamic_processor_class(processor_cls)

        image_processor_cls = getattr(module, "Eagle3_VLImageProcessorFast", None)
        _patch_eagle3_dynamic_processor_class(image_processor_cls)


def _patch_eagle3_dynamic_processor_class(processor_cls: object) -> None:
    if _is_eagle3_dynamic_processor_class(processor_cls):
        _patch_eagle3_processor_from_args_and_dict(processor_cls)
    if _is_eagle3_dynamic_fast_image_processor_class(processor_cls):
        _patch_eagle3_fast_image_processor_prepare_input_images(processor_cls)
        _patch_eagle3_fast_image_processor_preprocess(processor_cls)


def _install_eagle3_dynamic_processor_loader_compat(
    loader_module: Any, installed_attr_name: str
) -> None:
    if getattr(loader_module, installed_attr_name, False):
        return

    original_get_class_from_dynamic_module = loader_module.get_class_from_dynamic_module

    @wraps(original_get_class_from_dynamic_module)
    def _compat_get_class_from_dynamic_module(*args: Any, **kwargs: Any) -> type[Any]:
        processor_cls = original_get_class_from_dynamic_module(*args, **kwargs)
        _patch_eagle3_dynamic_processor_class(processor_cls)
        return processor_cls

    setattr(
        loader_module,
        "get_class_from_dynamic_module",
        _compat_get_class_from_dynamic_module,
    )
    setattr(loader_module, installed_attr_name, True)


def _install_eagle3_dynamic_processor_compat() -> None:
    from transformers.models.auto import image_processing_auto, processing_auto

    _patch_loaded_eagle3_dynamic_processors()
    _install_eagle3_dynamic_processor_loader_compat(
        processing_auto,
        "_gr00t_eagle3_dynamic_processor_compat_installed",
    )
    _install_eagle3_dynamic_processor_loader_compat(
        image_processing_auto,
        "_gr00t_eagle3_dynamic_image_processor_compat_installed",
    )


def _is_siglip2_dynamic_modeling_module(module: object) -> bool:
    module_file = str(getattr(module, "__file__", "") or "")
    return (
        module_file.endswith("modeling_siglip2.py")
        and "transformers_modules" in module_file
    )


def _patch_siglip2_dynamic_module_globals(module: object) -> None:
    if getattr(module, "_gr00t_siglip2_class_alias_compat_installed", False):
        return

    try:
        from transformers.models.siglip2 import modeling_siglip2 as installed_siglip2
    except Exception:
        return

    for symbol_name in ("Siglip2Model", "Siglip2ForImageClassification"):
        if not hasattr(module, symbol_name) and hasattr(installed_siglip2, symbol_name):
            setattr(module, symbol_name, getattr(installed_siglip2, symbol_name))

    setattr(module, "_gr00t_siglip2_class_alias_compat_installed", True)


def _patch_loaded_siglip2_dynamic_modules() -> None:
    for module in tuple(sys.modules.values()):
        if _is_siglip2_dynamic_modeling_module(module):
            _patch_siglip2_dynamic_module_globals(module)


def _install_siglip2_dynamic_modeling_compat() -> None:
    from transformers.models.auto import auto_factory

    _patch_loaded_siglip2_dynamic_modules()

    if getattr(auto_factory, "_gr00t_siglip2_dynamic_modeling_compat_installed", False):
        return

    original_get_class_from_dynamic_module = auto_factory.get_class_from_dynamic_module

    @wraps(original_get_class_from_dynamic_module)
    def _compat_get_class_from_dynamic_module(*args: Any, **kwargs: Any) -> type[Any]:
        model_cls = original_get_class_from_dynamic_module(*args, **kwargs)
        _patch_loaded_siglip2_dynamic_modules()
        return model_cls

    setattr(
        auto_factory,
        "get_class_from_dynamic_module",
        _compat_get_class_from_dynamic_module,
    )
    setattr(auto_factory, "_gr00t_siglip2_dynamic_modeling_compat_installed", True)


def _is_eagle3_dynamic_config_class(config_cls: object) -> bool:
    if not isinstance(config_cls, type):
        return False
    if getattr(config_cls, "__name__", "") != "Eagle3_VLConfig":
        return False
    module = sys.modules.get(getattr(config_cls, "__module__", ""))
    module_file = str(getattr(module, "__file__", "") or "")
    return module_file.endswith("configuration_eagle3_vl.py")


def _is_eagle3_dynamic_modeling_module(module: object) -> bool:
    module_file = str(getattr(module, "__file__", "") or "")
    return (
        module_file.endswith("modeling_eagle3_vl.py")
        and "transformers_modules" in module_file
    )


def _resolve_eagle3_initializer_range(config: Any) -> float:
    initializer_range = getattr(config, "initializer_range", None)
    if initializer_range is not None:
        return float(initializer_range)

    text_config = getattr(config, "text_config", None)
    initializer_range = getattr(text_config, "initializer_range", None)
    if initializer_range is not None:
        return float(initializer_range)

    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            nested_text_config = get_text_config()
        except Exception:
            nested_text_config = None
        initializer_range = getattr(nested_text_config, "initializer_range", None)
        if initializer_range is not None:
            return float(initializer_range)

    return 0.02


def _patch_eagle3_pretrained_model_init_weights(model_cls: type[Any]) -> None:
    if getattr(model_cls, "_gr00t_init_weights_compat_installed", False):
        return

    original_init_weights = getattr(model_cls, "_init_weights")

    @wraps(original_init_weights)
    def _compat_init_weights(self: Any, module: Any) -> Any:
        config = getattr(self, "config", None)
        if getattr(config, "initializer_range", None) is not None:
            return original_init_weights(self, module)

        initializer_range = _resolve_eagle3_initializer_range(config)
        if config is not None:
            try:
                setattr(config, "initializer_range", initializer_range)
            except Exception:
                pass
            if getattr(config, "initializer_range", None) is not None:
                return original_init_weights(self, module)

        from torch import nn

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        return None

    setattr(model_cls, "_init_weights", _compat_init_weights)
    setattr(model_cls, "_gr00t_init_weights_compat_installed", True)


def _patch_eagle3_initializer_range_compat(config_cls: object) -> None:
    if not isinstance(config_cls, type):
        return

    if getattr(config_cls, "_gr00t_initializer_range_compat_installed", False):
        return

    original_init = getattr(config_cls, "__init__")

    @wraps(original_init)
    def _compat_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if getattr(self, "initializer_range", None) is None:
            setattr(self, "initializer_range", _resolve_eagle3_initializer_range(self))

    setattr(config_cls, "__init__", _compat_init)
    setattr(config_cls, "_gr00t_initializer_range_compat_installed", True)


def _patch_loaded_eagle3_dynamic_modeling_modules() -> None:
    for module in tuple(sys.modules.values()):
        if not _is_eagle3_dynamic_modeling_module(module):
            continue
        pretrained_model_cls = getattr(module, "Eagle3_VLPreTrainedModel", None)
        if isinstance(pretrained_model_cls, type):
            _patch_eagle3_pretrained_model_init_weights(pretrained_model_cls)


def _patch_loaded_eagle3_dynamic_configs() -> None:
    for module in tuple(sys.modules.values()):
        config_cls = getattr(module, "Eagle3_VLConfig", None)
        if _is_eagle3_dynamic_config_class(config_cls):
            _patch_eagle3_initializer_range_compat(config_cls)


def _install_eagle3_dynamic_config_compat() -> None:
    from transformers import dynamic_module_utils

    _patch_loaded_eagle3_dynamic_configs()

    if getattr(
        dynamic_module_utils, "_gr00t_eagle3_dynamic_config_compat_installed", False
    ):
        return

    original_get_class_from_dynamic_module = (
        dynamic_module_utils.get_class_from_dynamic_module
    )

    @wraps(original_get_class_from_dynamic_module)
    def _compat_get_class_from_dynamic_module(*args: Any, **kwargs: Any) -> type[Any]:
        loaded_cls = original_get_class_from_dynamic_module(*args, **kwargs)
        if _is_eagle3_dynamic_config_class(loaded_cls):
            _patch_eagle3_initializer_range_compat(loaded_cls)
        return loaded_cls

    setattr(
        dynamic_module_utils,
        "get_class_from_dynamic_module",
        _compat_get_class_from_dynamic_module,
    )
    setattr(dynamic_module_utils, "_gr00t_eagle3_dynamic_config_compat_installed", True)


def _install_eagle3_dynamic_modeling_compat() -> None:
    from transformers.models.auto import auto_factory

    _patch_loaded_eagle3_dynamic_modeling_modules()

    if getattr(auto_factory, "_gr00t_eagle3_dynamic_modeling_compat_installed", False):
        return

    original_get_class_from_dynamic_module = auto_factory.get_class_from_dynamic_module

    @wraps(original_get_class_from_dynamic_module)
    def _compat_get_class_from_dynamic_module(*args: Any, **kwargs: Any) -> type[Any]:
        model_cls = original_get_class_from_dynamic_module(*args, **kwargs)
        _patch_loaded_eagle3_dynamic_modeling_modules()
        return model_cls

    setattr(
        auto_factory,
        "get_class_from_dynamic_module",
        _compat_get_class_from_dynamic_module,
    )
    setattr(auto_factory, "_gr00t_eagle3_dynamic_modeling_compat_installed", True)


def install_transformers_image_processor_fast_compat() -> None:
    try:
        from transformers.configuration_utils import PretrainedConfig
        from transformers import image_utils as transformers_image_utils
        from transformers import video_utils as transformers_video_utils
        from transformers import (
            image_processing_utils_fast as image_processing_utils_fast,
        )

        if not hasattr(PretrainedConfig, "_attn_implementation_autoset"):
            setattr(PretrainedConfig, "_attn_implementation_autoset", False)
        if not hasattr(transformers_image_utils, "VideoInput") and hasattr(
            transformers_image_utils, "ImageInput"
        ):
            setattr(
                transformers_image_utils,
                "VideoInput",
                getattr(transformers_image_utils, "ImageInput"),
            )
        if not hasattr(
            image_processing_utils_fast, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING"
        ):
            setattr(
                image_processing_utils_fast,
                "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING",
                "",
            )
        if not hasattr(
            image_processing_utils_fast,
            "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS",
        ):
            setattr(
                image_processing_utils_fast,
                "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS",
                "",
            )
        if not hasattr(image_processing_utils_fast, "DefaultFastImageProcessorKwargs"):
            setattr(
                image_processing_utils_fast,
                "DefaultFastImageProcessorKwargs",
                _build_default_fast_image_processor_kwargs(),
            )
        if not hasattr(transformers_image_utils, "make_batched_videos") and hasattr(
            transformers_video_utils, "make_batched_videos"
        ):
            setattr(
                transformers_image_utils,
                "make_batched_videos",
                getattr(transformers_video_utils, "make_batched_videos"),
            )
        _install_eagle3_dynamic_config_compat()
        _install_eagle3_dynamic_modeling_compat()
        _install_eagle3_dynamic_processor_compat()
        _install_siglip2_dynamic_modeling_compat()
    except Exception:
        return
