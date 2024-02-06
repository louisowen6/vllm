"""A simple, flexible implementation of a GPT model.
Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""
import math
import warnings
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput

import logging

log = logging.getLogger(__name__)

#################################################################
# fc.py
from torch import nn
FC_CLASS_REGISTRY = {'torch': nn.Linear}
try:
    import transformer_engine.pytorch as te
    FC_CLASS_REGISTRY['te'] = te.Linear
except:
    pass
#################################################################

#################################################################
# configuration_mpt.py
"""A HuggingFace-style model configuration."""
import warnings
from typing import Any, Dict, Optional, Union
from transformers import PretrainedConfig
attn_config_defaults: Dict = {'attn_type': 'multihead_attention', 'attn_pdrop': 0.0, 'attn_impl': 'triton', 'qk_ln': False, 'clip_qkv': None, 'softmax_scale': None, 'prefix_lm': False, 'attn_uses_sequence_id': False, 
'alibi': False, 'alibi_bias_max': 8}
ffn_config_defaults: Dict = {'ffn_type': 'mptmlp'}
init_config_defaults: Dict = {'name': 'kaiming_normal_', 'fan_mode': 'fan_in', 'init_nonlinearity': 'relu', 'init_div_is_residual': True, 'emb_init_std': None, 'emb_init_uniform_lim': None, 'init_std': None, 
'init_gain': 0.0}

class MPTConfig(PretrainedConfig):
    model_type = 'mpt'

    def __init__(self, d_model: int=2048, n_heads: int=16, n_layers: int=24, expansion_ratio: int=4, max_seq_len: int=2048, vocab_size: int=50368, resid_pdrop: float=0.0, emb_pdrop: float=0.0, learned_pos_emb: 
bool=True, attn_config: Dict=attn_config_defaults, ffn_config: Dict=ffn_config_defaults, init_device: str='cpu', logit_scale: Optional[Union[float, str]]=None, no_bias: bool=False, embedding_fraction: float=1.0, 
norm_type: str='low_precision_layernorm', use_cache: bool=False, init_config: Dict=init_config_defaults, fc_type: str='torch', verbose: Optional[int]=None, **kwargs: Any):
        """The MPT configuration class.
        Args:
            d_model (int): The size of the embedding dimension of the model.
            n_heads (int): The number of attention heads.
            n_layers (int): The number of layers in the model.
            expansion_ratio (int): The ratio of the up/down scale in the ffn.
            max_seq_len (int): The maximum sequence length of the model.
            vocab_size (int): The size of the vocabulary.
            resid_pdrop (float): The dropout probability applied to the attention output before combining with residual.
            emb_pdrop (float): The dropout probability for the embedding layer.
            learned_pos_emb (bool): Whether to use learned positional embeddings
            attn_config (Dict): A dictionary used to configure the model's attention module:
                attn_type (str): type of attention to use. Options: multihead_attention, multiquery_attention, grouped_query_attention
                attn_pdrop (float): The dropout probability for the attention layers.
                attn_impl (str): The attention implementation to use. One of 'torch', 'flash', or 'triton'.
                qk_ln (bool): Whether to apply layer normalization to the queries and keys in the attention layer.
                clip_qkv (Optional[float]): If not None, clip the queries, keys, and values in the attention layer to
                    this value.
                softmax_scale (Optional[float]): If not None, scale the softmax in the attention layer by this value. If None,
                    use the default scale of ``1/sqrt(d_keys)``.
                prefix_lm (Optional[bool]): Whether the model should operate as a Prefix LM. This requires passing an
                    extra `prefix_mask` argument which indicates which tokens belong to the prefix. Tokens in the prefix
                    can attend to one another bi-directionally. Tokens outside the prefix use causal attention.
                attn_uses_sequence_id (Optional[bool]): Whether to restrict attention to tokens that have the same sequence_id.
                    When the model is in `train` mode, this requires passing an extra `sequence_id` argument which indicates
                    which sub-sequence each token belongs to.
                    Defaults to ``False`` meaning any provided `sequence_id` will be ignored.
                alibi (bool): Whether to use the alibi bias instead of position embeddings.
                alibi_bias_max (int): The maximum value of the alibi bias.
                kv_n_heads (Optional[int]): For grouped_query_attention only, allow user to specify number of kv heads.
            ffn_config (Dict): A dictionary used to configure the model's ffn module:
                ffn_type (str): type of ffn to use. Options: mptmlp, te_ln_mlp
            init_device (str): The device to use for parameter initialization.
            logit_scale (Optional[Union[float, str]]): If not None, scale the logits by this value.
            no_bias (bool): Whether to use bias in all layers.
            verbose (int): The verbosity level. 0 is silent.
            embedding_fraction (float): The fraction to scale the gradients of the embedding layer by.
            norm_type (str): choose type of norm to use
            use_cache (bool): Whether or not the model should return the last key/values attentions
            init_config (Dict): A dictionary used to configure the model initialization:
                init_config.name: The parameter initialization scheme to use. Options: 'default_', 'baseline_',
                    'kaiming_uniform_', 'kaiming_normal_', 'neox_init_', 'small_init_', 'xavier_uniform_', or
                    'xavier_normal_'. These mimic the parameter initialization methods in PyTorch.
                init_div_is_residual (Union[int, float, str, bool]): Value to divide initial weights by if ``module._is_residual`` is True.
                emb_init_std (Optional[float]): The standard deviation of the normal distribution used to initialize the embedding layer.
                emb_init_uniform_lim (Optional[Union[Tuple[float, float], float]]): The lower and upper limits of the uniform distribution
                    used to initialize the embedding layer. Mutually exclusive with ``emb_init_std``.
                init_std (float): The standard deviation of the normal distribution used to initialize the model,
                    if using the baseline_ parameter initialization scheme.
                init_gain (float): The gain to use for parameter initialization with kaiming or xavier initialization schemes.
                fan_mode (str): The fan mode to use for parameter initialization with kaiming initialization schemes.
                init_nonlinearity (str): The nonlinearity to use for parameter initialization with kaiming initialization schemes.
                ---
                See llmfoundry.models.utils.param_init_fns.py for info on other param init config options
            fc_type (str): choose fc layer implementation. Options: torch and te. te layers support fp8 when using H100 GPUs.
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.attn_config = attn_config
        self.ffn_config = ffn_config
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.use_cache = use_cache
        self.init_config = init_config
        self.fc_type = fc_type
        if verbose is not None:
            warnings.warn(DeprecationWarning('verbose argument for MPTConfig is now ignored and will be removed. Use python_log_level instead.'))
        if 'name' in kwargs:
            del kwargs['name']
        if 'loss_fn' in kwargs:
            del kwargs['loss_fn']
        if self.attn_config.get('alibi', False):
            self.learned_pos_emb = False
            warnings.warn(f'alibi is turned on, setting `learned_pos_emb` to `False.`')
        super().__init__(**kwargs)
        self._validate_config()

    def _set_config_defaults(self, config: Dict[str, Any], config_defaults: Dict[str, Any]) -> Dict[str, Any]:
        for (k, v) in config_defaults.items():
            if k not in config:
                config[k] = v
        return config

    def _validate_config(self) -> None:
        self.attn_config = self._set_config_defaults(self.attn_config, attn_config_defaults)
        self.ffn_config = self._set_config_defaults(self.ffn_config, ffn_config_defaults)
        self.init_config = self._set_config_defaults(self.init_config, init_config_defaults)
        if self.d_model % self.n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        if any((prob < 0 or prob > 1 for prob in [self.attn_config['attn_pdrop'], self.resid_pdrop, self.emb_pdrop])):
            raise ValueError("self.attn_config['attn_pdrop'], resid_pdrop, emb_pdrop are probabilities and must be between 0 and 1")
        if self.attn_config['attn_impl'] not in ['torch', 'flash', 'triton']:
            raise ValueError(f"Unknown attn_impl={self.attn_config['attn_impl']}")
        if self.attn_config['prefix_lm'] and self.attn_config['attn_impl'] not in ['torch', 'triton']:
            raise NotImplementedError('prefix_lm only implemented with torch and triton attention.')
        if self.attn_config['alibi'] and self.attn_config['attn_impl'] not in ['torch', 'triton']:
            raise NotImplementedError('alibi only implemented with torch and triton attention.')
        if self.attn_config['attn_uses_sequence_id'] and self.attn_config['attn_impl'] not in ['torch', 'triton']:
            raise NotImplementedError('attn_uses_sequence_id only implemented with torch and triton attention.')
        if self.embedding_fraction > 1 or self.embedding_fraction <= 0:
            raise ValueError('model.embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!')
        if isinstance(self.logit_scale, str) and self.logit_scale != 'inv_sqrt_d_model':
            raise ValueError(f"self.logit_scale={self.logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.")
        if self.init_config.get('name', None) is None:
            raise ValueError(f"self.init_config={self.init_config!r} 'name' needs to be set.")
        if not self.learned_pos_emb and (not self.attn_config['alibi']):
            warnings.warn(f'Positional information not being provided to the model using either learned_pos_emb or alibi.')
        if self.fc_type == 'te' or self.ffn_config['ffn_type'] == 'te_ln_mlp':
            try:
                import transformer_engine.pytorch as te
                del te
            except:
                raise ImportError('TransformerEngine import fail. `fc_type: te` requires TransformerEngine be installed. ' + 'The required version of transformer_engine also requires FlashAttention v1.0.6 is 
installed:\n' + 'pip install flash-attn==1.0.6 --no-build-isolation \n' + 'pip install git+https://github.com/NVIDIA/TransformerEngine.git@144e4888b2cdd60bd52e706d5b7a79cb9c1a7156')
        if self.ffn_config['ffn_type'] == 'mptmlp':
            self.ffn_config['fc_type'] = self.fc_type
        elif self.ffn_config['ffn_type'] == 'te_ln_mlp':
            self.ffn_config['bias'] = not self.no_bias
#################################################################

#################################################################
# adapt_tokenizer.py
from typing import Any
from transformers import AutoTokenizer, PreTrainedTokenizerBase
NUM_SENTINEL_TOKENS: int = 100

def adapt_tokenizer_for_denoising(tokenizer: PreTrainedTokenizerBase) -> None:
    """Adds sentinel tokens and padding token (if missing).
    Expands the tokenizer vocabulary to include sentinel tokens
    used in mixture-of-denoiser tasks as well as a padding token.
    All added tokens are added as special tokens. No tokens are
    added if sentinel tokens and padding token already exist.
    """
    sentinels_to_add = [f'<extra_id_{i}>' for i in range(NUM_SENTINEL_TOKENS)]
    tokenizer.add_tokens(sentinels_to_add, special_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.add_tokens('<pad>', special_tokens=True)
        tokenizer.pad_token = '<pad>'
        assert tokenizer.pad_token_id is not None
    sentinels = ''.join([f'<extra_id_{i}>' for i in range(NUM_SENTINEL_TOKENS)])
    _sentinel_token_ids = tokenizer(sentinels, add_special_tokens=False).input_ids
    tokenizer.sentinel_token_ids = _sentinel_token_ids

class AutoTokenizerForMOD(AutoTokenizer):
    """AutoTokenizer + Adaptation for MOD.
    A simple wrapper around AutoTokenizer to make instantiating
    an MOD-adapted tokenizer a bit easier.
    MOD-adapted tokenizers have sentinel tokens (e.g., <extra_id_0>),
    a padding token, and a property to get the token ids of the
    sentinel tokens.
    """

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> PreTrainedTokenizerBase:
        """See `AutoTokenizer.from_pretrained` docstring."""
        tokenizer = super().from_pretrained(*args, **kwargs)
        adapt_tokenizer_for_denoising(tokenizer)
        return tokenizer
#################################################################

#################################################################
# norm.py
from typing import Dict, List, Optional, Type, Union
import torch

def _cast_if_autocast_enabled(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor

class LPLayerNorm(torch.nn.LayerNorm):

    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], eps: float=1e-05, elementwise_affine: bool=True, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)

def rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor]=None, eps: float=1e-05) -> torch.Tensor:
    output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output

class RMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], eps: float=1e-05, weight: bool=True, dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)

class LPRMSNorm(RMSNorm):

    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], eps: float=1e-05, weight: bool=True, dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, weight=weight, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        with torch.autocast(enabled=False, device_type=x.device.type):
            return rms_norm(downcast_x, downcast_weight, self.eps).to(dtype=x.dtype)
NORM_CLASS_REGISTRY: Dict[str, Type[torch.nn.Module]] = {'layernorm': torch.nn.LayerNorm, 'low_precision_layernorm': LPLayerNorm, 'rmsnorm': RMSNorm, 'low_precision_rmsnorm': LPRMSNorm}
#################################################################

#################################################################
# ffn.py
"""GPT Blocks used for the GPT Model."""
from typing import Any, Optional
import torch
import torch.nn as nn
try:
    import transformer_engine.pytorch as te
except:
    te = None


class MPTMLP(nn.Module):

    def __init__(
        self,
        config: MPTConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        hidden_size = config.d_model
        expansion_ratio = config.expansion_ratio
        intermediate_size = expansion_ratio * hidden_size
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=not config.no_bias,
            linear_method=linear_method,
        )
        quant_config = getattr(linear_method, "quant_config", None)
        self.act = get_act_fn("gelu", quant_config, intermediate_size)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=not config.no_bias,
            linear_method=linear_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)
        x = self.act(x)
        x, _ = self.down_proj(x)
        return x
#################################################################

#################################################################
# param_init_fns.py

import math
import warnings
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union
import torch
from torch import nn
try:
    import transformer_engine.pytorch as te
except:
    te = None

def torch_default_param_init_fn_(module: nn.Module, **kwargs: Any) -> None:
    del kwargs
    if hasattr(module, 'reset_parameters') and isinstance(module.reset_parameters, Callable):
        module.reset_parameters()

def fused_init_helper_(module: nn.Module, init_fn_: Callable) -> None:
    _fused = getattr(module, '_fused', None)
    if _fused is None:
        raise RuntimeError(f'Internal logic error')
    assert isinstance(module.weight, torch.Tensor)
    (dim, splits) = _fused
    splits = (0, *splits, module.weight.size(dim))
    for (s, e) in zip(splits[:-1], splits[1:]):
        slice_indices = [slice(None)] * module.weight.ndim
        slice_indices[dim] = slice(s, e)
        init_fn_(module.weight[slice_indices])

def generic_param_init_fn_(module: nn.Module, init_fn_: Callable, n_layers: int, d_model: Optional[int]=None, init_div_is_residual: Union[int, float, str, bool]=True, emb_init_std: Optional[float]=None, 
emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]]=None, **kwargs: Any) -> None:
    del kwargs
    init_div_is_residual = init_div_is_residual
    if init_div_is_residual is False:
        div_is_residual = 1.0
    elif init_div_is_residual is True:
        div_is_residual = math.sqrt(2 * n_layers)
    elif isinstance(init_div_is_residual, float) or isinstance(init_div_is_residual, int):
        div_is_residual = init_div_is_residual
    elif init_div_is_residual.isnumeric():
        div_is_residual = float(init_div_is_residual)
    else:
        div_is_residual = 1.0
        raise ValueError(f'Expected init_div_is_residual to be boolean or numeric, got {init_div_is_residual}')
    if isinstance(module, tuple(set(FC_CLASS_REGISTRY.values()))):
        if hasattr(module, '_fused'):
            fused_init_helper_(module, init_fn_)
        else:
            init_fn_(module.weight)
        if module.bias is not None:
            assert isinstance(module.bias, torch.Tensor)
            torch.nn.init.zeros_(module.bias)
        if init_div_is_residual is not False and getattr(module, '_is_residual', False):
            with torch.no_grad():
                module.weight.div_(div_is_residual)
    elif isinstance(module, nn.Embedding):
        if emb_init_std is not None:
            std = emb_init_std
            if std == 0:
                warnings.warn(f'Embedding layer initialized to 0.')
            emb_init_fn_ = partial(torch.nn.init.normal_, mean=0.0, std=std)
        elif emb_init_uniform_lim is not None:
            lim = emb_init_uniform_lim
            if isinstance(lim, Sequence):
                if len(lim) > 2:
                    raise ValueError(f'Uniform init requires a min and a max limit. User input: {lim}.')
                if lim[0] == lim[1]:
                    warnings.warn(f'Embedding layer initialized to {lim[0]}.')
            else:
                if lim == 0:
                    warnings.warn(f'Embedding layer initialized to 0.')
                lim = [-lim, lim]
            (a, b) = lim
            emb_init_fn_ = partial(torch.nn.init.uniform_, a=a, b=b)
        else:
            emb_init_fn_ = init_fn_
        emb_init_fn_(module.weight)
    elif isinstance(module, tuple(set(NORM_CLASS_REGISTRY.values()))):
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            torch.nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        if module._qkv_same_embed_dim:
            assert module.in_proj_weight is not None
            assert module.q_proj_weight is None and module.k_proj_weight is None and (module.v_proj_weight is None)
            assert d_model is not None
            _d = d_model
            splits = (0, _d, 2 * _d, 3 * _d)
            for (s, e) in zip(splits[:-1], splits[1:]):
                init_fn_(module.in_proj_weight[s:e])
        else:
            assert module.q_proj_weight is not None and module.k_proj_weight is not None and (module.v_proj_weight is not None)
            assert module.in_proj_weight is None
            init_fn_(module.q_proj_weight)
            init_fn_(module.k_proj_weight)
            init_fn_(module.v_proj_weight)
        if module.in_proj_bias is not None:
            torch.nn.init.zeros_(module.in_proj_bias)
        if module.bias_k is not None:
            torch.nn.init.zeros_(module.bias_k)
        if module.bias_v is not None:
            torch.nn.init.zeros_(module.bias_v)
        init_fn_(module.out_proj.weight)
        if init_div_is_residual is not False and getattr(module.out_proj, '_is_residual', False):
            with torch.no_grad():
                module.out_proj.weight.div_(div_is_residual)
        if module.out_proj.bias is not None:
            torch.nn.init.zeros_(module.out_proj.bias)
    elif te is not None and isinstance(module, te.LayerNormMLP):
        if isinstance(module.layer_norm_weight, torch.Tensor):
            torch.nn.init.ones_(module.layer_norm_weight)
        if isinstance(module.layer_norm_bias, torch.Tensor):
            torch.nn.init.zeros_(module.layer_norm_bias)
        init_fn_(module.fc1_weight)
        if module.fc1_bias is not None:
            assert isinstance(module.fc1_bias, torch.Tensor)
            torch.nn.init.zeros_(module.fc1_bias)
        init_fn_(module.fc2_weight)
        if module.fc2_bias is not None:
            assert isinstance(module.fc2_bias, torch.Tensor)
            torch.nn.init.zeros_(module.fc2_bias)
        with torch.no_grad():
            module.fc2_weight.div_(div_is_residual)
    else:
        for _ in module.parameters(recurse=False):
            raise NotImplementedError(f'{module.__class__.__name__} parameters are not initialized by param_init_fn.')

def _normal_init_(std: float, mean: float=0.0) -> Callable:
    return partial(torch.nn.init.normal_, mean=mean, std=std)

def _normal_param_init_fn_(module: nn.Module, std: float, n_layers: int, d_model: Optional[int]=None, init_div_is_residual: Union[int, float, str, bool]=True, emb_init_std: Optional[float]=None, emb_init_uniform_lim: 
Optional[Union[Tuple[float, float], float]]=None, **kwargs: Any) -> None:
    del kwargs
    init_fn_ = _normal_init_(std=std)
    generic_param_init_fn_(module=module, init_fn_=init_fn_, d_model=d_model, n_layers=n_layers, init_div_is_residual=init_div_is_residual, emb_init_std=emb_init_std, emb_init_uniform_lim=emb_init_uniform_lim)

def baseline_param_init_fn_(module: nn.Module, init_std: Optional[float], n_layers: int, d_model: Optional[int]=None, init_div_is_residual: Union[int, float, str, bool]=True, emb_init_std: Optional[float]=None, 
emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]]=None, **kwargs: Any) -> None:
    del kwargs
    if init_std is None:
        raise ValueError("You must set model.init_config['init_std'] to a float value to use the default initialization scheme.")
    _normal_param_init_fn_(module=module, std=init_std, d_model=d_model, n_layers=n_layers, init_div_is_residual=init_div_is_residual, emb_init_std=emb_init_std, emb_init_uniform_lim=emb_init_uniform_lim)

def small_param_init_fn_(module: nn.Module, n_layers: int, d_model: int, init_div_is_residual: Union[int, float, str, bool]=True, emb_init_std: Optional[float]=None, emb_init_uniform_lim: Optional[Union[Tuple[float, 
float], float]]=None, **kwargs: Any) -> None:
    del kwargs
    std = math.sqrt(2 / (5 * d_model))
    _normal_param_init_fn_(module=module, std=std, d_model=d_model, n_layers=n_layers, init_div_is_residual=init_div_is_residual, emb_init_std=emb_init_std, emb_init_uniform_lim=emb_init_uniform_lim)

def neox_param_init_fn_(module: nn.Module, n_layers: int, d_model: int, emb_init_std: Optional[float]=None, emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]]=None, **kwargs: Any) -> None:
    """From section 2.3.1 of GPT-NeoX-20B:
    An Open-Source AutoregressiveLanguage Model — Black et. al. (2022)
    see https://github.com/EleutherAI/gpt-neox/blob/9610391ab319403cef079b438edd016a2443af54/megatron/model/init_functions.py#L151
    and https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py
    """
    del kwargs
    residual_div = n_layers / math.sqrt(10)
    small_param_init_fn_(module=module, d_model=d_model, n_layers=n_layers, init_div_is_residual=residual_div, emb_init_std=emb_init_std, emb_init_uniform_lim=emb_init_uniform_lim)

def kaiming_uniform_param_init_fn_(module: nn.Module, n_layers: int, d_model: Optional[int]=None, init_div_is_residual: Union[int, float, str, bool]=True, emb_init_std: Optional[float]=None, emb_init_uniform_lim: 
Optional[Union[Tuple[float, float], float]]=None, init_gain: float=0, fan_mode: str='fan_in', init_nonlinearity: str='leaky_relu', **kwargs: Any) -> None:
    del kwargs
    kaiming_uniform_ = partial(nn.init.kaiming_uniform_, a=init_gain, mode=fan_mode, nonlinearity=init_nonlinearity)
    generic_param_init_fn_(module=module, init_fn_=kaiming_uniform_, d_model=d_model, n_layers=n_layers, init_div_is_residual=init_div_is_residual, emb_init_std=emb_init_std, 
emb_init_uniform_lim=emb_init_uniform_lim)

def kaiming_normal_param_init_fn_(module: nn.Module, n_layers: int, d_model: Optional[int]=None, init_div_is_residual: Union[int, float, str, bool]=True, emb_init_std: Optional[float]=None, emb_init_uniform_lim: 
Optional[Union[Tuple[float, float], float]]=None, init_gain: float=0, fan_mode: str='fan_in', init_nonlinearity: str='leaky_relu', **kwargs: Any) -> None:
    del kwargs
    kaiming_normal_ = partial(torch.nn.init.kaiming_normal_, a=init_gain, mode=fan_mode, nonlinearity=init_nonlinearity)
    generic_param_init_fn_(module=module, init_fn_=kaiming_normal_, d_model=d_model, n_layers=n_layers, init_div_is_residual=init_div_is_residual, emb_init_std=emb_init_std, emb_init_uniform_lim=emb_init_uniform_lim)

def xavier_uniform_param_init_fn_(module: nn.Module, n_layers: int, d_model: Optional[int]=None, init_div_is_residual: Union[int, float, str, bool]=True, emb_init_std: Optional[float]=None, emb_init_uniform_lim: 
Optional[Union[Tuple[float, float], float]]=None, init_gain: float=0, **kwargs: Any) -> None:
    del kwargs
    xavier_uniform_ = partial(torch.nn.init.xavier_uniform_, gain=init_gain)
    generic_param_init_fn_(module=module, init_fn_=xavier_uniform_, d_model=d_model, n_layers=n_layers, init_div_is_residual=init_div_is_residual, emb_init_std=emb_init_std, emb_init_uniform_lim=emb_init_uniform_lim)

def xavier_normal_param_init_fn_(module: nn.Module, n_layers: int, d_model: Optional[int]=None, init_div_is_residual: Union[int, float, str, bool]=True, emb_init_std: Optional[float]=None, emb_init_uniform_lim: 
Optional[Union[Tuple[float, float], float]]=None, init_gain: float=0, **kwargs: Any) -> None:
    del kwargs
    xavier_normal_ = partial(torch.nn.init.xavier_normal_, gain=init_gain)
    generic_param_init_fn_(module=module, init_fn_=xavier_normal_, d_model=d_model, n_layers=n_layers, init_div_is_residual=init_div_is_residual, emb_init_std=emb_init_std, emb_init_uniform_lim=emb_init_uniform_lim)
MODEL_INIT_REGISTRY = {'default_': torch_default_param_init_fn_, 'baseline_': baseline_param_init_fn_, 'kaiming_uniform_': kaiming_uniform_param_init_fn_, 'kaiming_normal_': kaiming_normal_param_init_fn_, 
'neox_init_': neox_param_init_fn_, 'small_init_': small_param_init_fn_, 'xavier_uniform_': xavier_uniform_param_init_fn_, 'xavier_normal_': xavier_normal_param_init_fn_}
#################################################################

#################################################################
# meta_init_context.py
from contextlib import contextmanager
from typing import Any, Callable, Optional
import torch
import torch.nn as nn

@contextmanager
def init_empty_weights(include_buffers: bool=False):
    """Meta initialization context manager.
    A context manager under which models are initialized with all parameters
    on the meta device, therefore creating an empty model. Useful when just
    initializing the model would blow the available RAM.
    Args:
        include_buffers (`bool`, *optional*, defaults to `False`): Whether or
            not to also put all buffers on the meta device while initializing.
    Example:
    ```python
    import torch.nn as nn
    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```
    <Tip warning={true}>
    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    </Tip>
    """
    with init_on_device(torch.device('meta'), include_buffers=include_buffers) as f:
        yield f

@contextmanager
def init_on_device(device: torch.device, include_buffers: bool=False):
    """Device initialization context manager.
    A context manager under which models are initialized with all parameters
    on the specified device.
    Args:
        device (`torch.device`): Device to initialize all parameters on.
        include_buffers (`bool`, *optional*, defaults to `False`): Whether or
            not to also put all buffers on the meta device while initializing.
    Example:
    ```python
    import torch.nn as nn
    with init_on_device(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(self: torch.nn.Module, name: str, param: Optional[torch.nn.Parameter]):
        old_register_parameter(self, name, param)
        if param is not None:
            parameter = self._parameters[name]
            assert parameter is not None
            param_cls = type(parameter)
            kwargs = parameter.__dict__
            self._parameters[name] = param_cls(parameter.to(device), **kwargs)

    def register_empty_buffer(self: torch.nn.Module, name: str, tensor: Optional[torch.Tensor], persistent: bool=True):
        old_register_buffer(self, name, tensor, persistent=persistent)
        if tensor is not None:
            named_buffer = self._buffers[name]
            assert named_buffer is not None
            self._buffers[name] = named_buffer.to(device)
    if include_buffers:
        tensor_constructors_to_patch = {torch_function_name: getattr(torch, torch_function_name) for torch_function_name in ['empty', 'zeros', 'ones', 'full']}
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn: Callable):

        def wrapper(*args: Any, **kwargs: Any):
            kwargs['device'] = device
            return fn(*args, **kwargs)
        return wrapper
    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for (torch_function_name, old_torch_function) in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)
#################################################################

#################################################################
# hf_prefixlm_converter.py

"""Converts Huggingface Causal LM to Prefix LM.
Conversion does lightweight surgery on a HuggingFace
Causal LM to convert it to a Prefix LM.
Prefix LMs accepts a `bidirectional_mask` input in `forward`
and treat the input prompt as the prefix in `generate`.
"""
from types import MethodType
from typing import Any, List, MutableMapping, Optional, Tuple, Union
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM

_SUPPORTED_GPT_MODELS = (
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
)
CAUSAL_GPT_TYPES = Union[
    GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM
]


def _convert_gpt_causal_lm_to_prefix_lm(model: CAUSAL_GPT_TYPES) -> CAUSAL_GPT_TYPES:
    """Converts a GPT-style Causal LM to a Prefix LM.
    Supported HuggingFace model classes:
        - `GPT2LMHeadModel`
        - `GPTNeoForCausalLM`
        - `GPTNeoXForCausalLM`
        - `GPTJForCausalLM`
    See `convert_hf_causal_lm_to_prefix_lm` for more details.
    """
    if hasattr(model, "_prefix_lm_converted"):
        return model
    assert isinstance(model, _SUPPORTED_GPT_MODELS)
    assert (
        model.config.add_cross_attention == False
    ), "Only supports GPT-style decoder-only models"

    def _get_attn_modules(model: CAUSAL_GPT_TYPES) -> List[torch.nn.Module]:
        """Helper that gets a list of the model's attention modules.
        Each module has a `bias` buffer used for causal masking. The Prefix LM
        conversion adds logic to dynamically manipulate these biases to support
        Prefix LM attention masking.
        """
        attn_modules = []
        if isinstance(model, GPTNeoXForCausalLM):
            blocks = model.gpt_neox.layers
        else:
            blocks = model.transformer.h
        for block in blocks:
            if isinstance(model, GPTNeoForCausalLM):
                if block.attn.attention_type != "global":
                    continue
                attn_module = block.attn.attention
            elif isinstance(model, GPTNeoXForCausalLM):
                attn_module = block.attention
            else:
                attn_module = block.attn
            attn_modules.append(attn_module)
        return attn_modules

    setattr(model, "_original_forward", getattr(model, "forward"))
    setattr(model, "_original_generate", getattr(model, "generate"))

    def forward(
        self: CAUSAL_GPT_TYPES,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        bidirectional_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Wraps original forward to enable PrefixLM attention."""

        def call_og_forward():
            if isinstance(self, GPTNeoXForCausalLM):
                return self._original_forward(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                return self._original_forward(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        if bidirectional_mask is None:
            return call_og_forward()
        assert isinstance(bidirectional_mask, torch.Tensor)
        attn_modules = _get_attn_modules(model)
        (b, s) = bidirectional_mask.shape
        max_length = attn_modules[0].bias.shape[-1]
        if s > max_length:
            raise ValueError(
                f"bidirectional_mask sequence length (={s}) exceeds the "
                + f"max length allowed by the model ({max_length})."
            )
        assert s <= max_length
        if s < max_length:
            pad = torch.zeros(
                (int(b), int(max_length - s)),
                dtype=bidirectional_mask.dtype,
                device=bidirectional_mask.device,
            )
            bidirectional_mask = torch.cat([bidirectional_mask, pad], dim=1)
        bidirectional = bidirectional_mask.unsqueeze(1).unsqueeze(1)
        for attn_module in attn_modules:
            assert isinstance(attn_module.bias, torch.Tensor)
            attn_module.bias.data = torch.logical_or(
                attn_module.bias.data, bidirectional
            )
        output = call_og_forward()
        for attn_module in attn_modules:
            attn_module.bias.data = torch.tril(attn_module.bias.data[0, 0])[None, None]
        return output

    def generate(self: CAUSAL_GPT_TYPES, *args: Any, **kwargs: Any):
        """Wraps original generate to enable PrefixLM attention."""
        attn_modules = _get_attn_modules(model)
        for attn_module in attn_modules:
            attn_module.bias.data[:] = 1
        output = self._original_generate(*args, **kwargs)
        for attn_module in attn_modules:
            attn_module.bias.data = torch.tril(attn_module.bias.data[0, 0])[None, None]
        return output

    setattr(model, "forward", MethodType(forward, model))
    setattr(model, "generate", MethodType(generate, model))
    setattr(model, "_prefix_lm_converted", True)
    return model


_SUPPORTED_HF_MODELS = _SUPPORTED_GPT_MODELS
CAUSAL_LM_TYPES = Union[
    GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM
]


def convert_hf_causal_lm_to_prefix_lm(model: CAUSAL_LM_TYPES) -> CAUSAL_LM_TYPES:
    """Converts a HuggingFace Causal LM to a Prefix LM.
    Supported HuggingFace model classes:
        - `GPT2LMHeadModel`
        - `GPTNeoForCausalLM`
        - `GPTNeoXForCausalLM`
        - `GPTJForCausalLM`
    Conversion to a Prefix LM is done by modifying the `forward` method, and possibly also the
    `generate` method and/or select underlying methods depending on the model class.
    These changes preserve the model API, but add a new input to `forward`: "bidirectional_mask".
    Notes on training:
        To actually train the converted model as a Prefix LM, training batches will need to indicate
        the prefix/target structure by including `bidirectional_mask` as part of the batch inputs.
        **This is not a standard input and requires custom layers either within or after your dataloader.**
        In addition to adding `bidirectional_mask` to the batch, this custom code should modify `labels`
        such that `batch['labels'][batch['bidirectional_mask'] == 1] == -100`.
        That is, the prefix portion of the sequence should not generate any loss. Loss should only be
        generated by the target portion of the sequence.
    Notes on `GPTNeoForCausalLM`:
        To simplify the implementation, "global" and "local" attention layers are handled differently.
        For "global" layers, we handle conversion as described above. For "local" layers, which use a
        causal attention mask within a restricted local window, we do not alter the masking.
    Notes on `forward` method conversion:
        After conversion, the `forward` method will handle a new input, `bidirectional_mask`,
        which should be a [batch_size, seq_length] byte tensor, where 1 indicates token positions
        belonging to the prefix (prefix tokens can attend to one another bidirectionally), and
        0 indicates token positions belonging to the target.
        The new `forward` method will incorporate `bidirectional_mask` (if supplied) into the existing
        causal mask, call the original `forward` method, and (if the causal mask is a buffer) reset
        the causal masks before returning the result.
    Notes on `generate` method conversion:
        After conversion, the `generate` method will have the same signature but will internally
        convert all causal masks to be purely bidirectional, call the original `generate` method, and
        (where appropriate) reset the causal masks before returning the result.
        This works thanks to the logic of the HuggingFace `generate` API, which first encodes the token
        "prompt" passed to `generate` (which is treated as the prefix) and then sequentially generates
        each new token. Encodings are cached as generation happens, so all prefix tokens can attend to one
        another (as expected in a Prefix LM) and generated tokens can only attend to prefix tokens and
        previously-generated tokens (also as expected in a Prefix LM).
    To preserve the API, the original methods are renamed to `_original_forward` and
    `_original_generate`, and replaced with new `forward` and `generate` methods that wrap
    them, respectively. Although implementation details vary by model class.
    """
    if isinstance(model, _SUPPORTED_GPT_MODELS):
        return _convert_gpt_causal_lm_to_prefix_lm(model)
    else:
        raise TypeError(
            f"Cannot convert model to Prefix LM. "
            + f"Model does not belong to set of supported HF models:"
            + f"\n{_SUPPORTED_HF_MODELS}"
        )


def add_bidirectional_mask_if_missing(batch: MutableMapping):
    """Attempts to add bidirectional_mask to batch if missing.
    Raises:
        KeyError if bidirectional_mask is missing and can't be inferred
    """
    if "bidirectional_mask" not in batch:
        if batch.get("mode", None) == "icl_task":
            batch["bidirectional_mask"] = batch["attention_mask"].clone()
            for i, continuation_indices in enumerate(batch["continuation_indices"]):
                batch["bidirectional_mask"][i, continuation_indices] = 0
        elif "labels" in batch and "attention_mask" in batch:
            batch["bidirectional_mask"] = torch.logical_and(
                torch.eq(batch["attention_mask"], 1), torch.eq(batch["labels"], -100)
            ).type_as(batch["attention_mask"])
        else:
            raise KeyError(
                "No bidirectional_mask in batch and not sure how to construct one."
            )
#################################################################

#################################################################
# attention.py

"""Attention layers."""
import math
import warnings
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
from einops import rearrange
from packaging import version
from torch import nn

def is_flash_v2_installed():
    try:
        import flash_attn as flash_attn
    except:
        return False
    return version.parse(flash_attn.__version__) >= version.parse('2.0.0')

def is_flash_v1_installed():
    try:
        import flash_attn as flash_attn
    except:
        return False
    return version.parse(flash_attn.__version__) < version.parse('2.0.0')

def _reset_is_causal(num_query_tokens: int, num_key_tokens: int, original_is_causal: bool) -> bool:
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError('MPT does not support query and key with different number of tokens, unless number of query tokens is 1.')
        else:
            return False
    return original_is_causal

def repeat_kv_for_gqa(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Perform repeat of kv heads along a particular dimension.
    hidden.shape expected to be: (batch size, seq len, kv_n_heads, head_dim)
    n_rep: amount of repetitions of kv_n_heads
    Unlike torch.repeat_interleave, this function avoids allocating new memory.
    """
    if n_rep == 1:
        return hidden
    (b, s, kv_n_heads, d) = hidden.shape
    hidden = hidden[:, :, :, None, :].expand(b, s, kv_n_heads, n_rep, d)
    return hidden.reshape(b, s, kv_n_heads * n_rep, d)

def scaled_multihead_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, n_heads: int, kv_n_heads: Optional[int]=None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, 
softmax_scale: Optional[float]=None, attn_bias: Optional[torch.Tensor]=None, key_padding_mask: Optional[torch.Tensor]=None, is_causal: bool=False, dropout_p: float=0.0, training: bool=False, needs_weights: 
bool=False, multiquery: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    if multiquery:
        warnings.warn(DeprecationWarning('The direct use of the multiquery arg is deprecated. Setting kv_n_heads=1 automatically. Please set kv_n_heads=1 explicitly to remove this warning.'))
        kv_n_heads = 1
    elif kv_n_heads is None:
        warnings.warn(DeprecationWarning('Not specifying a value for the kv_n_heads arg is deprecated. Setting kv_n_heads=n_heads automatically. Please set kv_n_heads=n_heads explicitly to remove this warning.'))
        kv_n_heads = n_heads
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
    v = rearrange(value, 'b s (h d) -> b h s d', h=kv_n_heads)
    if past_key_value is not None:
        if len(past_key_value) != 0:
            k = torch.cat([past_key_value[0], k], dim=3)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v)
    (b, _, s_q, d) = q.shape
    s_k = k.size(-1)
    if kv_n_heads > 1 and kv_n_heads < n_heads:
        k = repeat_kv_for_gqa(k.transpose(1, 2), n_heads // kv_n_heads).transpose(1, 2)
        v = repeat_kv_for_gqa(v.transpose(1, 2), n_heads // kv_n_heads).transpose(1, 2)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)
    attn_weight = q.matmul(k) * softmax_scale
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - s_q)
        _s_k = max(0, attn_bias.size(3) - s_k)
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
        if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q):
            raise RuntimeError(f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.')
        attn_weight = attn_weight + attn_bias
    min_val = torch.finfo(q.dtype).min
    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn('Propagating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unnecessary computation/memory usage. Consider integrating ' + 'into 
attn_bias once and passing that to each attention ' + 'module instead.')
        attn_weight = attn_weight.masked_fill(~key_padding_mask.view((b, 1, 1, s_k)), min_val)
    if is_causal and (not q.size(2) == 1):
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float32)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=training, inplace=True)
    out = attn_weight.to(v.dtype).matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')
    if needs_weights:
        return (out, attn_weight, past_key_value)
    return (out, None, past_key_value)

def check_valid_inputs(*tensors: torch.Tensor, valid_dtypes: Optional[List[torch.dtype]]=None):
    if valid_dtypes is None:
        valid_dtypes = [torch.float16, torch.bfloat16]
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f'tensor.dtype={tensor.dtype!r} must be in valid_dtypes={valid_dtypes!r}.')
        if not tensor.is_cuda:
            raise TypeError(f'Inputs must be cuda tensors (tensor.is_cuda={tensor.is_cuda!r}).')

def flash_attn_fn(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, n_heads: int, kv_n_heads: Optional[int]=None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, softmax_scale: 
Optional[float]=None, attn_bias: Optional[torch.Tensor]=None, key_padding_mask: Optional[torch.Tensor]=None, is_causal: bool=False, dropout_p: float=0.0, training: bool=False, needs_weights: bool=False, multiquery: 
bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    try:
        from flash_attn import bert_padding, flash_attn_interface
    except:
        raise RuntimeError('Please install flash-attn==1.0.9 or flash-attn==2.3.2')
    check_valid_inputs(query, key, value)
    if multiquery:
        warnings.warn(DeprecationWarning('The direct use of the multiquery arg is deprecated. Setting kv_n_heads=1 automatically. Please set kv_n_heads=1 explicitly to remove this warning.'))
        kv_n_heads = 1
    elif kv_n_heads is None:
        warnings.warn(DeprecationWarning('Not specifying a value for the kv_n_heads arg is deprecated. Setting kv_n_heads=n_heads automatically. Please set kv_n_heads=n_heads explicitly to remove this warning.'))
        kv_n_heads = n_heads
    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)
        past_key_value = (key, value)
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
    if attn_bias is not None:
        raise NotImplementedError(f'attn_bias not implemented for flash attn.')
    (batch_size, seqlen) = query.shape[:2]
    if key_padding_mask is None:
        key_padding_mask = torch.ones_like(key[:, :, 0], dtype=torch.bool)
    query_padding_mask = key_padding_mask[:, -query.size(1):]
    (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = bert_padding.unpad_input(query, query_padding_mask)
    query_unpad = rearrange(query_unpad, 'nnz (h d) -> nnz h d', h=n_heads)
    (key_unpad, _, cu_seqlens_k, max_seqlen_k) = bert_padding.unpad_input(key, key_padding_mask)
    key_unpad = rearrange(key_unpad, 'nnz (h d) -> nnz h d', h=kv_n_heads)
    (value_unpad, _, _, _) = bert_padding.unpad_input(value, key_padding_mask)
    value_unpad = rearrange(value_unpad, 'nnz (h d) -> nnz h d', h=kv_n_heads)
    if kv_n_heads == 1:
        key_unpad = key_unpad.expand(key_unpad.size(0), n_heads, key_unpad.size(-1))
        value_unpad = value_unpad.expand(value_unpad.size(0), n_heads, value_unpad.size(-1))
    elif kv_n_heads < n_heads:
        key_unpad = repeat_kv_for_gqa(key_unpad.view(batch_size, seqlen, kv_n_heads, -1), n_heads // kv_n_heads).view(batch_size * seqlen, n_heads, -1)
        value_unpad = repeat_kv_for_gqa(value_unpad.view(batch_size, seqlen, kv_n_heads, -1), n_heads // kv_n_heads).view(batch_size * seqlen, n_heads, -1)
    dropout_p = dropout_p if training else 0.0
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    if is_flash_v1_installed():
        output_unpad = flash_attn_interface.flash_attn_unpadded_func(q=query_unpad, k=key_unpad, v=value_unpad, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, 
max_seqlen_k=max_seqlen_k, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=reset_is_causal, return_attn_probs=needs_weights)
    elif is_flash_v2_installed():
        output_unpad = flash_attn_interface.flash_attn_varlen_func(q=query_unpad, k=key_unpad, v=value_unpad, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, 
max_seqlen_k=max_seqlen_k, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=reset_is_causal, return_attn_probs=needs_weights)
    else:
        raise RuntimeError('flash-attn==1.0.9 or flash-attn==2.3.2 is required.')
    output = bert_padding.pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices_q, batch_size, seqlen)
    return (output, None, past_key_value)

def triton_flash_attn_fn(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, n_heads: int, kv_n_heads: Optional[int]=None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, softmax_scale: 
Optional[float]=None, attn_bias: Optional[torch.Tensor]=None, key_padding_mask: Optional[torch.Tensor]=None, is_causal: bool=False, dropout_p: float=0.0, training: bool=False, needs_weights: bool=False, multiquery: 
bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    try:
        from .flash_attn_triton import flash_attn_func
    except:
        _installed = False
        if version.parse(torch.__version__) < version.parse('2.0.0'):
            _installed = True
            try:
                from flash_attn.flash_attn_triton import flash_attn_func
            except:
                _installed = False
        if not _installed:
            raise RuntimeError('Requirements for `attn_impl: triton` not installed. Either (1) have a CUDA-compatible GPU ' + 'and `pip install .[gpu]` if installing from llm-foundry source or ' + '`pip install 
triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python` ' + 'if installing from pypi, or (2) use torch attn model.attn_config.attn_impl=torch (torch attn_impl will be slow). ' + 
'Note: (1) requires you have CMake and PyTorch already installed.')
    check_valid_inputs(query, key, value)
    if multiquery:
        warnings.warn(DeprecationWarning('The direct use of the multiquery arg is deprecated. Setting kv_n_heads=1 automatically. Please set kv_n_heads=1 explicitly to remove this warning.'))
        kv_n_heads = 1
    elif kv_n_heads is None:
        warnings.warn(DeprecationWarning('Not specifying a value for the kv_n_heads arg is deprecated. Setting kv_n_heads=n_heads automatically. Please set kv_n_heads=n_heads explicitly to remove this warning.'))
        kv_n_heads = n_heads
    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)
        past_key_value = (key, value)
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
    if dropout_p:
        raise NotImplementedError(f'Dropout not implemented for attn_impl: triton.')
    dropout_p = dropout_p if training else 0.0
    if needs_weights:
        raise NotImplementedError(f'attn_impl: triton cannot return attn weights.')
    if key_padding_mask is not None:
        warnings.warn('Propagating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unnecessary computation/memory usage. Consider integrating ' + 'into 
attn_bias once and passing that to each attention ' + 'module instead.')
        (b_size, s_k) = key_padding_mask.shape[:2]
        if attn_bias is None:
            attn_bias = query.new_zeros(b_size, 1, 1, s_k)
        attn_bias = attn_bias.masked_fill(~key_padding_mask.view((b_size, 1, 1, s_k)), torch.finfo(query.dtype).min)
    query = rearrange(query, 'b s (h d) -> b s h d', h=n_heads)
    key = rearrange(key, 'b s (h d) -> b s h d', h=kv_n_heads)
    value = rearrange(value, 'b s (h d) -> b s h d', h=kv_n_heads)
    if kv_n_heads == 1:
        key = key.repeat(1, 1, n_heads, 1)
        value = value.repeat(1, 1, n_heads, 1)
    elif kv_n_heads < n_heads:
        key = repeat_kv_for_gqa(key, n_heads // kv_n_heads)
        value = repeat_kv_for_gqa(value, n_heads // kv_n_heads)
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_func(query, key, value, attn_bias, reset_is_causal, softmax_scale)
    output = attn_output.view(*attn_output.shape[:2], -1)
    return (output, None, past_key_value)

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) is a generalization of Multi-head (MHA).
    and Multi-query attention (MQA).
    This allows the user to set a variable of number of kv_n_heads, rather than
    just n_heads or 1, as in MHA and MQA. Using torch or triton attention
    implementation enables user to also use additive bias.
    """

    def __init__(self, d_model: int, n_heads: int, kv_n_heads: int, attn_impl: str='triton', clip_qkv: Optional[float]=None, qk_ln: bool=False, softmax_scale: Optional[float]=None, attn_pdrop: float=0.0, norm_type: 
str='low_precision_layernorm', fc_type: str='torch', device: Optional[str]=None, bias: bool=True):
        super().__init__()
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln
        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_n_heads = kv_n_heads
        self.head_dim = d_model // n_heads
        if self.kv_n_heads <= 0:
            raise ValueError('kv_n_heads should be greater than zero.')
        if self.kv_n_heads > self.n_heads:
            raise ValueError('The number of KV heads should be less than or equal to Q heads.')
        if self.n_heads % self.kv_n_heads != 0:
            raise ValueError('Each Q head should get the same number of KV heads, so n_heads must be divisible by kv_n_heads.')
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop
        fc_kwargs: dict[str, Any] = {'bias': bias}
        if fc_type != 'te':
            fc_kwargs['device'] = device
        self.Wqkv = FC_CLASS_REGISTRY[fc_type](self.d_model, self.d_model + 2 * self.kv_n_heads * self.head_dim, **fc_kwargs)
        fuse_splits = [i * self.head_dim for i in range(1, self.n_heads + 2 * self.kv_n_heads)]
        self.Wqkv._fused = (0, fuse_splits)
        if self.qk_ln:
            norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
            self.q_ln = norm_class(self.d_model, device=device)
            self.k_ln = norm_class(self.kv_n_heads * self.head_dim, device=device)
        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
        else:
            raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
        self.out_proj = FC_CLASS_REGISTRY[fc_type](self.d_model, self.d_model, **fc_kwargs)
        self.out_proj._is_residual = True

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        del position_ids  # unused.
        qkv, _ = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.qk_ln:
            q = self.q_ln(q)
            k = self.k_ln(k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.out_proj(attn_output)
        return output

class MultiheadAttention(GroupedQueryAttention):
    """Multi-head self attention.
    Using torch or triton attention implementation enables user to also use
    additive bias.
    """

    def __init__(self, d_model: int, n_heads: int, attn_impl: str='triton', clip_qkv: Optional[float]=None, qk_ln: bool=False, softmax_scale: Optional[float]=None, attn_pdrop: float=0.0, norm_type: 
str='low_precision_layernorm', fc_type: str='torch', device: Optional[str]=None, bias: bool=True):
        super().__init__(d_model=d_model, n_heads=n_heads, kv_n_heads=n_heads, attn_impl=attn_impl, clip_qkv=clip_qkv, qk_ln=qk_ln, softmax_scale=softmax_scale, attn_pdrop=attn_pdrop, norm_type=norm_type, 
fc_type=fc_type, device=device, bias=bias)

class MultiQueryAttention(GroupedQueryAttention):
    """Multi-Query self attention.
    Using torch or triton attention implementation enables user to also use
    additive bias.
    """

    def __init__(self, d_model: int, n_heads: int, attn_impl: str='triton', clip_qkv: Optional[float]=None, qk_ln: bool=False, softmax_scale: Optional[float]=None, attn_pdrop: float=0.0, norm_type: 
str='low_precision_layernorm', fc_type: str='torch', device: Optional[str]=None, bias: bool=True):
        super().__init__(d_model=d_model, n_heads=n_heads, kv_n_heads=1, attn_impl=attn_impl, clip_qkv=clip_qkv, qk_ln=qk_ln, softmax_scale=softmax_scale, attn_pdrop=attn_pdrop, norm_type=norm_type, fc_type=fc_type, 
device=device, bias=bias)

def attn_bias_shape(attn_impl: str, n_heads: int, seq_len: int, alibi: bool, prefix_lm: bool, causal: bool, use_sequence_id: bool) -> Optional[Tuple[int, int, int, int]]:
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            if (prefix_lm or not causal) or use_sequence_id:
                return (1, n_heads, seq_len, seq_len)
            return (1, n_heads, 1, seq_len)
        elif prefix_lm or use_sequence_id:
            return (1, 1, seq_len, seq_len)
        return None
    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')

def build_attn_bias(attn_impl: str, attn_bias: torch.Tensor, n_heads: int, seq_len: int, causal: bool=False, alibi: bool=False, alibi_bias_max: int=8) -> Optional[torch.Tensor]:
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            (device, dtype) = (attn_bias.device, attn_bias.dtype)
            attn_bias = attn_bias.add(build_alibi_bias(n_heads, seq_len, full=not causal, alibi_bias_max=alibi_bias_max, device=device, dtype=dtype))
        return attn_bias
    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')

def gen_slopes(n_heads: int, alibi_bias_max: int=8, device: Optional[torch.device]=None) -> torch.Tensor:
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = 1.0 / torch.pow(2, m)
    if _n_heads != n_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes.view(1, n_heads, 1, 1)

def build_alibi_bias(n_heads: int, seq_len: int, full: bool=False, alibi_bias_max: int=8, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, 1, seq_len)
    if full:
        alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)
    slopes = gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)
ATTN_CLASS_REGISTRY = {'multihead_attention': MultiheadAttention, 'multiquery_attention': MultiQueryAttention, 'grouped_query_attention': GroupedQueryAttention}


def _get_alibi_slopes(
    total_num_heads: int,
    alibi_bias_max: int,
) -> torch.Tensor:
    next_power_of_2 = 2**math.ceil(math.log2(total_num_heads))
    m = torch.arange(1, next_power_of_2 + 1, dtype=torch.float32)
    m = m.mul(alibi_bias_max / next_power_of_2)
    slopes = 1.0 / torch.pow(2, m)
    if next_power_of_2 != total_num_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:total_num_heads]
    return slopes


class MPTAttention(nn.Module):

    def __init__(
        self,
        config: MPTConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.total_num_heads = config.n_heads
        self.head_dim = self.d_model // self.total_num_heads
        self.clip_qkv = config.attn_config["clip_qkv"]
        self.qk_ln = config.attn_config["qk_ln"]
        self.alibi_bias_max = config.attn_config["alibi_bias_max"]
        if "kv_n_heads" in config.attn_config:
            self.total_num_kv_heads = config.attn_config['kv_n_heads']
        else:
            self.total_num_kv_heads = self.total_num_heads
        assert not config.attn_config["prefix_lm"]
        assert config.attn_config["alibi"]

        # pylint: disable=invalid-name
        self.Wqkv = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=not config.no_bias,
            linear_method=linear_method,
        )
        if self.qk_ln:
            self.q_ln = nn.LayerNorm(self.d_model)
            self.k_ln = nn.LayerNorm(self.d_model)
        self.out_proj = RowParallelLinear(
            self.d_model,
            self.d_model,
            bias=not config.no_bias,
            linear_method=linear_method,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Create the alibi slopes and slice them.
        tp_rank = get_tensor_model_parallel_rank()
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        alibi_slopes = _get_alibi_slopes(self.total_num_heads,
                                         self.alibi_bias_max)
        alibi_slopes = alibi_slopes[head_start:head_end].tolist()

        self.head_dim = self.d_model // self.total_num_heads
        scaling = self.head_dim**-0.5
        self.attn = PagedAttention(self.num_heads,
                                   self.head_dim,
                                   scaling,
                                   alibi_slopes=alibi_slopes,
                                   num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        del position_ids  # unused.
        qkv, _ = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.qk_ln:
            q = self.q_ln(q)
            k = self.k_ln(k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.out_proj(attn_output)
        return output
#################################################################

#################################################################
# blocks.py

"""GPT Blocks used for the GPT Model."""
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

class MPTBlock(nn.Module):

    def __init__(
        self,
        config: MPTConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        hidden_size = config.d_model
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.attn = MPTAttention(config, linear_method)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.ffn = MPTMLP(config, linear_method)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        x = self.norm_1(hidden_states)
        x = self.attn(
            position_ids=position_ids,
            hidden_states=x,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )
        hidden_states = hidden_states + x
        x = self.norm_2(hidden_states)
        x = self.ffn(x)
        hidden_states = hidden_states + x
        return hidden_states
#################################################################

#################################################################
# custom_embedding.py
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SharedEmbedding(nn.Embedding):

    def forward(self, input: Tensor, unembed: bool=False) -> Tensor:
        if unembed:
            return F.linear(input, self.weight)
        return super().forward(input)
#################################################################

#################################################################
# flash_attn_triton.py
try:
    """
    Copied from https://github.com/HazyResearch/flash-attention/blob/eff9fe6b8076df59d64d7a3f464696738a3c7c24/flash_attn/flash_attn_triton.py
    update imports to use 'triton_pre_mlir'
    *Experimental* implementation of FlashAttention in Triton.
    Tested with triton==2.0.0.dev20221202.
    Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
    other than 64:
    https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
    We'll update this implementation with the new Triton backend once this is fixed.
    We use the FlashAttention implementation from Phil Tillet a starting point.
    https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py
    Changes:
    - Implement both causal and non-causal attention.
    - Implement both self-attention and cross-attention.
    - Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
    - Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
    - Support attention bias.
    - Speed up the forward pass a bit, and only store the LSE instead of m and l.
    - Make the backward for d=128 much faster by reducing register spilling.
    - Optionally parallelize the backward pass across seqlen_k, to deal with the case of
    small batch size * nheads.
    Caution:
    - This is an *experimental* implementation. The forward pass should be quite robust but
    I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
    - This implementation has only been tested on A100.
    - If you plan to use headdim other than 64 and 128, you should test for race conditions
    (due to the Triton compiler), as done in tests/test_flash_attn.py
    "test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
    for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
    that there are none left for other head dimensions.
    Differences between this Triton version and the CUDA version:
    - Triton version doesn't support dropout.
    - Triton forward is generally faster than CUDA forward, while Triton backward is
    generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
    than CUDA forward + backward.
    - Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
    - Triton version supports attention bias, while CUDA version doesn't.
    """
    import math
    import torch
    import triton_pre_mlir as triton
    import triton_pre_mlir.language as tl

    @triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args['BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == 
args['BLOCK_HEADDIM']})
    @triton.jit
    def _fwd_kernel(Q, K, V, Bias, Out, Lse, TMP, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, stride_ob, 
stride_oh, stride_om, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_HEADDIM: tl.constexpr, EVEN_M: 
tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        start_m = tl.program_id(0)
        off_hb = tl.program_id(1)
        off_b = off_hb // nheads
        off_h = off_hb % nheads
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_HEADDIM)
        q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
        if BIAS_TYPE == 'vector':
            b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
        elif BIAS_TYPE == 'matrix':
            b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + (offs_m[:, None] * stride_bm + offs_n[None, :])
        t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
        lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
        acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
        if EVEN_M & EVEN_N:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs)
            else:
                q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        elif EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
        for start_n in range(0, end_n, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    k = tl.load(k_ptrs + start_n * stride_kn)
                else:
                    k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
            elif EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k, trans_b=True)
            if not EVEN_N:
                qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float('-inf'))
            if IS_CAUSAL:
                qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float('-inf'))
            if BIAS_TYPE != 'none':
                if BIAS_TYPE == 'vector':
                    if EVEN_N:
                        bias = tl.load(b_ptrs + start_n).to(tl.float32)
                    else:
                        bias = tl.load(b_ptrs + start_n, mask=start_n + offs_n < seqlen_k, other=0.0).to(tl.float32)
                    bias = bias[None, :]
                elif BIAS_TYPE == 'matrix':
                    if EVEN_M & EVEN_N:
                        bias = tl.load(b_ptrs + start_n).to(tl.float32)
                    else:
                        bias = tl.load(b_ptrs + start_n, mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k), other=0.0).to(tl.float32)
                qk = qk * softmax_scale + bias
                m_ij = tl.maximum(tl.max(qk, 1), lse_i)
                p = tl.exp(qk - m_ij[:, None])
            else:
                m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
                p = tl.exp(qk * softmax_scale - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            acc_o_scale = tl.exp(m_i - m_ij)
            tl.store(t_ptrs, acc_o_scale)
            acc_o_scale = tl.load(t_ptrs)
            acc_o = acc_o * acc_o_scale[:, None]
            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    v = tl.load(v_ptrs + start_n * stride_vn)
                else:
                    v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
            elif EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)
            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)
        o_scale = tl.exp(m_i - lse_i)
        tl.store(t_ptrs, o_scale)
        o_scale = tl.load(t_ptrs)
        acc_o = acc_o * o_scale[:, None]
        start_m = tl.program_id(0)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
        tl.store(lse_ptrs, lse_i)
        offs_d = tl.arange(0, BLOCK_HEADDIM)
        out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
        if EVEN_M:
            if EVEN_HEADDIM:
                tl.store(out_ptrs, acc_o)
            else:
                tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
        elif EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

    @triton.jit
    def _bwd_preprocess_do_o_dot(Out, DO, Delta, stride_ob, stride_oh, stride_om, stride_dob, stride_doh, stride_dom, nheads, seqlen_q, seqlen_q_rounded, headdim, BLOCK_M: tl.constexpr, BLOCK_HEADDIM: tl.constexpr):
        start_m = tl.program_id(0)
        off_hb = tl.program_id(1)
        off_b = off_hb // nheads
        off_h = off_hb % nheads
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_HEADDIM)
        o = tl.load(Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :], mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
        do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :], mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
        delta = tl.sum(o * do, axis=1)
        tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)

    @triton.jit
    def _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k, headdim, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr):
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                tl.store(dv_ptrs, dv)
                tl.store(dk_ptrs, dk)
            else:
                tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
                tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
        elif EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))

    @triton.jit
    def _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn, seqlen_q, seqlen_k, 
headdim, ATOMIC_ADD: tl.constexpr, BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr, BLOCK_M: tl.constexpr, 
BLOCK_N: tl.constexpr):
        begin_m = 0 if not IS_CAUSAL else start_n * BLOCK_N // BLOCK_M * BLOCK_M
        offs_qm = begin_m + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_HEADDIM)
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
        do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
        dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
        if BIAS_TYPE == 'vector':
            b_ptrs = Bias + offs_n
        elif BIAS_TYPE == 'matrix':
            b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
        dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
        if begin_m >= seqlen_q:
            dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
            dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
            _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k, headdim, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM)
            return
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs)
                v = tl.load(v_ptrs)
            else:
                k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
                v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        elif EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
        for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m_curr = start_m + offs_m
            if EVEN_M & EVEN_HEADDIM:
                q = tl.load(q_ptrs)
            elif EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(q_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            qk = tl.dot(q, k, trans_b=True)
            if not EVEN_N:
                qk = tl.where(offs_n[None, :] < seqlen_k, qk, float('-inf'))
            if IS_CAUSAL:
                qk = tl.where(offs_m_curr[:, None] >= offs_n[None, :], qk, float('-inf'))
            if BIAS_TYPE != 'none':
                tl.debug_barrier()
                if BIAS_TYPE == 'vector':
                    if EVEN_N:
                        bias = tl.load(b_ptrs).to(tl.float32)
                    else:
                        bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                    bias = bias[None, :]
                elif BIAS_TYPE == 'matrix':
                    if EVEN_M & EVEN_N:
                        bias = tl.load(b_ptrs).to(tl.float32)
                    else:
                        bias = tl.load(b_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k), other=0.0).to(tl.float32)
                qk = qk * softmax_scale + bias
            if not EVEN_M & EVEN_HEADDIM:
                tl.debug_barrier()
            lse_i = tl.load(LSE + offs_m_curr)
            if BIAS_TYPE == 'none':
                p = tl.exp(qk * softmax_scale - lse_i[:, None])
            else:
                p = tl.exp(qk - lse_i[:, None])
            if EVEN_M & EVEN_HEADDIM:
                do = tl.load(do_ptrs)
            else:
                do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            dv += tl.dot(p.to(do.dtype), do, trans_a=True)
            if not EVEN_M & EVEN_HEADDIM:
                tl.debug_barrier()
            dp = tl.dot(do, v, trans_b=True)
            if not EVEN_HEADDIM:
                tl.debug_barrier()
            Di = tl.load(D + offs_m_curr)
            ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
            dk += tl.dot(ds, q, trans_a=True)
            if not EVEN_M & EVEN_HEADDIM:
                tl.debug_barrier()
            if not ATOMIC_ADD:
                if EVEN_M & EVEN_HEADDIM:
                    dq = tl.load(dq_ptrs, eviction_policy='evict_last')
                    dq += tl.dot(ds, k)
                    tl.store(dq_ptrs, dq, eviction_policy='evict_last')
                elif EVEN_HEADDIM:
                    dq = tl.load(dq_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0, eviction_policy='evict_last')
                    dq += tl.dot(ds, k)
                    tl.store(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q, eviction_policy='evict_last')
                else:
                    dq = tl.load(dq_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0, eviction_policy='evict_last')
                    dq += tl.dot(ds, k)
                    tl.store(dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), eviction_policy='evict_last')
            else:
                dq = tl.dot(ds, k)
                if EVEN_M & EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq)
                elif EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim))
            dq_ptrs += BLOCK_M * stride_dqm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_dom
            if BIAS_TYPE == 'matrix':
                b_ptrs += BLOCK_M * stride_bm
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k, headdim, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM)

    def init_to_zero(name):
        return lambda nargs: nargs[name].zero_()

    @triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'SEQUENCE_PARALLEL': False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 
'SEQUENCE_PARALLEL': True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ'))], key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM'])
    @triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args['BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == 
args['BLOCK_HEADDIM']})
    @triton.jit
    def _bwd_kernel(Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, 
stride_dob, stride_doh, stride_dom, stride_dqb, stride_dqh, stride_dqm, stride_dkb, stride_dkh, stride_dkn, stride_dvb, stride_dvh, stride_dvn, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, 
CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_HEADDIM: tl.constexpr, SEQUENCE_PARALLEL: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: 
tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        off_hb = tl.program_id(1)
        off_b = off_hb // nheads
        off_h = off_hb % nheads
        Q += off_b * stride_qb + off_h * stride_qh
        K += off_b * stride_kb + off_h * stride_kh
        V += off_b * stride_vb + off_h * stride_vh
        DO += off_b * stride_dob + off_h * stride_doh
        DQ += off_b * stride_dqb + off_h * stride_dqh
        DK += off_b * stride_dkb + off_h * stride_dkh
        DV += off_b * stride_dvb + off_h * stride_dvh
        if BIAS_TYPE != 'none':
            Bias += off_b * stride_bb + off_h * stride_bh
        D += off_hb * seqlen_q_rounded
        LSE += off_hb * seqlen_q_rounded
        if not SEQUENCE_PARALLEL:
            num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
            for start_n in range(0, num_block_n):
                _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn, seqlen_q, seqlen_k, 
headdim, ATOMIC_ADD=False, BIAS_TYPE=BIAS_TYPE, IS_CAUSAL=IS_CAUSAL, BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
        else:
            start_n = tl.program_id(0)
            _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn, seqlen_q, seqlen_k, 
headdim, ATOMIC_ADD=True, BIAS_TYPE=BIAS_TYPE, IS_CAUSAL=IS_CAUSAL, BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
        (batch, seqlen_q, nheads, d) = q.shape
        (_, seqlen_k, _, _) = k.shape
        assert k.shape == (batch, seqlen_k, nheads, d)
        assert v.shape == (batch, seqlen_k, nheads, d)
        assert d <= 128, 'FlashAttention only support head dimensions up to 128'
        assert q.dtype == k.dtype == v.dtype, 'All tensors must have the same type'
        assert q.dtype in [torch.float16, torch.bfloat16], 'Only support fp16 and bf16'
        assert q.is_cuda and k.is_cuda and v.is_cuda
        softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
        has_bias = bias is not None
        bias_type = 'none'
        if has_bias:
            assert bias.dtype in [q.dtype, torch.float]
            assert bias.is_cuda
            assert bias.dim() == 4
            if bias.stride(-1) != 1:
                bias = bias.contiguous()
            if bias.shape[2:] == (1, seqlen_k):
                bias_type = 'vector'
            elif bias.shape[2:] == (seqlen_q, seqlen_k):
                bias_type = 'matrix'
            else:
                raise RuntimeError('Last 2 dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)')
            bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
        seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
        lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)
        BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
        BLOCK = 128
        num_warps = 4 if d <= 64 else 8
        grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), batch * nheads)
        _fwd_kernel[grid](q, k, v, bias, o, lse, tmp, softmax_scale, q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1), v.stride(0), v.stride(2), v.stride(1), *bias_strides, o.stride(0), 
o.stride(2), o.stride(1), nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d, seqlen_q // 32, seqlen_k // 32, bias_type, causal, BLOCK_HEADDIM, BLOCK_M=BLOCK, BLOCK_N=BLOCK, num_warps=num_warps, num_stages=1)
        return (o, lse, softmax_scale)

    def _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None):
        if do.stride(-1) != 1:
            do = do.contiguous()
        (batch, seqlen_q, nheads, d) = q.shape
        (_, seqlen_k, _, _) = k.shape
        assert d <= 128
        seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
        assert lse.shape == (batch, nheads, seqlen_q_rounded)
        assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
        assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
        softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
        dq_accum = torch.empty_like(q, dtype=torch.float32)
        delta = torch.empty_like(lse)
        BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
        grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), batch * nheads)
        _bwd_preprocess_do_o_dot[grid](o, do, delta, o.stride(0), o.stride(2), o.stride(1), do.stride(0), do.stride(2), do.stride(1), nheads, seqlen_q, seqlen_q_rounded, d, BLOCK_M=128, BLOCK_HEADDIM=BLOCK_HEADDIM)
        has_bias = bias is not None
        bias_type = 'none'
        if has_bias:
            assert bias.dtype in [q.dtype, torch.float]
            assert bias.is_cuda
            assert bias.dim() == 4
            assert bias.stride(-1) == 1
            if bias.shape[2:] == (1, seqlen_k):
                bias_type = 'vector'
            elif bias.shape[2:] == (seqlen_q, seqlen_k):
                bias_type = 'matrix'
            else:
                raise RuntimeError('Last 2 dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)')
            bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
        grid = lambda META: (triton.cdiv(seqlen_k, META['BLOCK_N']) if META['SEQUENCE_PARALLEL'] else 1, batch * nheads)
        _bwd_kernel[grid](q, k, v, bias, do, dq_accum, dk, dv, lse, delta, softmax_scale, q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1), v.stride(0), v.stride(2), v.stride(1), 
*bias_strides, do.stride(0), do.stride(2), do.stride(1), dq_accum.stride(0), dq_accum.stride(2), dq_accum.stride(1), dk.stride(0), dk.stride(2), dk.stride(1), dv.stride(0), dv.stride(2), dv.stride(1), nheads, 
seqlen_q, seqlen_k, seqlen_q_rounded, d, seqlen_q // 32, seqlen_k // 32, bias_type, causal, BLOCK_HEADDIM)
        dq.copy_(dq_accum)

    class FlashAttnQKVPackedFunc(torch.autograd.Function):

        @staticmethod
        def forward(ctx, qkv, bias=None, causal=False, softmax_scale=None):
            """
                qkv: (batch, seqlen, 3, nheads, headdim)
                bias: optional, shape broadcastible to (batch, nheads, seqlen, seqlen).
                    For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen).
                    ALiBi mask for non-causal would have shape (1, nheads, seqlen, seqlen)
            """
            if qkv.stride(-1) != 1:
                qkv = qkv.contiguous()
            (o, lse, ctx.softmax_scale) = _flash_attn_forward(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], bias=bias, causal=causal, softmax_scale=softmax_scale)
            ctx.save_for_backward(qkv, o, lse, bias)
            ctx.causal = causal
            return o

        @staticmethod
        def backward(ctx, do):
            (qkv, o, lse, bias) = ctx.saved_tensors
            assert not ctx.needs_input_grad[1], 'FlashAttention does not support bias gradient yet'
            with torch.inference_mode():
                dqkv = torch.empty_like(qkv)
                _flash_attn_backward(do, qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], o, lse, dqkv[:, :, 0], dqkv[:, :, 1], dqkv[:, :, 2], bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
            return (dqkv, None, None, None)
    flash_attn_qkvpacked_func = FlashAttnQKVPackedFunc.apply

    class FlashAttnKVPackedFunc(torch.autograd.Function):

        @staticmethod
        def forward(ctx, q, kv, bias=None, causal=False, softmax_scale=None):
            """
                q: (batch, seqlen_q, nheads, headdim)
                kv: (batch, seqlen_k, 2, nheads, headdim)
                bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
                    For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
                    ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
            """
            (q, kv) = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, kv]]
            (o, lse, ctx.softmax_scale) = _flash_attn_forward(q, kv[:, :, 0], kv[:, :, 1], bias=bias, causal=causal, softmax_scale=softmax_scale)
            ctx.save_for_backward(q, kv, o, lse, bias)
            ctx.causal = causal
            return o

        @staticmethod
        def backward(ctx, do):
            (q, kv, o, lse, bias) = ctx.saved_tensors
            if len(ctx.needs_input_grad) >= 3:
                assert not ctx.needs_input_grad[2], 'FlashAttention does not support bias gradient yet'
            with torch.inference_mode():
                dq = torch.empty_like(q)
                dkv = torch.empty_like(kv)
                _flash_attn_backward(do, q, kv[:, :, 0], kv[:, :, 1], o, lse, dq, dkv[:, :, 0], dkv[:, :, 1], bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
            return (dq, dkv, None, None, None)
    flash_attn_kvpacked_func = FlashAttnKVPackedFunc.apply

    class FlashAttnFunc(torch.autograd.Function):

        @staticmethod
        def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None):
            """
                q: (batch_size, seqlen_q, nheads, headdim)
                k, v: (batch_size, seqlen_k, nheads, headdim)
                bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
                    For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
                    ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
            """
            (q, k, v) = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
            (o, lse, ctx.softmax_scale) = _flash_attn_forward(q, k, v, bias=bias, causal=causal, softmax_scale=softmax_scale)
            ctx.save_for_backward(q, k, v, o, lse, bias)
            ctx.causal = causal
            return o

        @staticmethod
        def backward(ctx, do):
            (q, k, v, o, lse, bias) = ctx.saved_tensors
            assert not ctx.needs_input_grad[3], 'FlashAttention does not support bias gradient yet'
            with torch.inference_mode():
                dq = torch.empty_like(q)
                dk = torch.empty_like(k)
                dv = torch.empty_like(v)
                _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
            return (dq, dk, dv, None, None, None)
    flash_attn_func = FlashAttnFunc.apply
except:
    pass
#################################################################


class MPTPreTrainedModel(PreTrainedModel):
    config_class = MPTConfig
    base_model_prefix = "model"
    _no_split_modules = ["MPTBlock"]
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module: nn.Module, value=False) -> None:
        if (
            isinstance(module, MPTModel)
            or isinstance(module, MultiheadAttention)
            or isinstance(module, MultiQueryAttention)
        ):
            module.gradient_checkpointing = value


class MPTModel(MPTPreTrainedModel):
    def __init__(self, 
                config: MPTConfig, 
                linear_method: Optional[LinearMethodBase] = None,):
        config._validate_config()
        super().__init__(config)
        self.gradient_checkpointing = False
        self.attn_impl = config.attn_config["attn_impl"]
        self.prefix_lm = config.attn_config["prefix_lm"]
        self.attn_uses_sequence_id = config.attn_config["attn_uses_sequence_id"]
        self.alibi = config.attn_config["alibi"]
        self.alibi_bias_max = config.attn_config["alibi_bias_max"]
        self.learned_pos_emb = config.learned_pos_emb
        if config.init_device == "mixed":
            if dist.get_local_rank() == 0:
                config.init_device = "cpu"
            else:
                config.init_device = "meta"
        if config.norm_type.lower() not in NORM_CLASS_REGISTRY.keys():
            norm_options = " | ".join(NORM_CLASS_REGISTRY.keys())
            raise NotImplementedError(
                f"Requested norm type ({config.norm_type}) is not implemented within this repo (Options: {norm_options})."
            )
        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]
        self.embedding_fraction = config.embedding_fraction
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
        )
        if self.learned_pos_emb:
            self.wpe = torch.nn.Embedding(
                config.max_seq_len, config.d_model, device=config.init_device
            )
        self.emb_drop = nn.Dropout(config.emb_pdrop)
        self.blocks = nn.ModuleList(
            [
                MPTBlock(config, linear_method)
                for _ in range(config.n_layers)
            ]
        )
        self.norm_f = norm_class(config.d_model, device=config.init_device)
        if config.init_device != "meta":
            log.info(
                f'We recommend using config.init_device="meta" with Composer + FSDP for faster initialization.'
            )
            self.apply(self.param_init_fn)
        self.is_causal = not self.prefix_lm
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attn_bias_shape(
            self.attn_impl,
            config.n_heads,
            config.max_seq_len,
            self.alibi,
            prefix_lm=self.prefix_lm,
            causal=self.is_causal,
            use_sequence_id=self.attn_uses_sequence_id,
        )
        if config.no_bias:
            for module in self.modules():
                if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                    log.info(f"Removing bias ({module.bias}) from {module}.")
                    module.register_parameter("bias", None)
                if hasattr(module, "use_bias"):
                    log.info(f"Setting use_bias=False for {module}.")
                    module.use_bias = False
        log.debug(self)
        log.debug(f"Using {self.config.init_config['name']} initialization.")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.wte = value

    @torch.no_grad()
    def _attn_bias(
        self,
        device: torch.device,
        dtype: torch.dtype,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.ByteTensor]]:
        if not self._attn_bias_initialized:
            if self.attn_bias_shape:
                self.attn_bias = torch.zeros(
                    self.attn_bias_shape, device=device, dtype=dtype
                )
                self.attn_bias = build_attn_bias(
                    self.attn_impl,
                    self.attn_bias,
                    self.config.n_heads,
                    self.config.max_seq_len,
                    causal=self.is_causal,
                    alibi=self.alibi,
                    alibi_bias_max=self.alibi_bias_max,
                )
            self._attn_bias_initialized = True
        if self.attn_impl == "flash":
            return (self.attn_bias, attention_mask)
        if self.attn_bias is not None:
            self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)
        attn_bias = self.attn_bias
        if self.prefix_lm:
            assert isinstance(attn_bias, torch.Tensor)
            assert isinstance(prefix_mask, torch.Tensor)
            attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)
        if self.attn_uses_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)
            attn_bias = self._apply_sequence_id(attn_bias, sequence_id)
        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
            else:
                _s_k = max(0, attn_bias.size(-1) - s_k)
                attn_bias = attn_bias[:, :, :, _s_k:]
            if prefix_mask is not None and attention_mask.shape != prefix_mask.shape:
                raise ValueError(
                    f"attention_mask shape={attention_mask.shape} "
                    + f"and prefix_mask shape={prefix_mask.shape} are not equal."
                )
            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(
                ~attention_mask.view(-1, 1, 1, s_k), min_val
            )
        return (attn_bias, None)

    def _apply_prefix_mask(
        self, attn_bias: torch.Tensor, prefix_mask: torch.Tensor
    ) -> torch.Tensor:
        (s_k, s_q) = attn_bias.shape[-2:]
        if s_k != self.config.max_seq_len or s_q != self.config.max_seq_len:
            raise ValueError(
                "attn_bias does not match the expected shape. "
                + f"The last two dimensions should both be {self.config.max_length} "
                + f"but are {s_k} and {s_q}."
            )
        seq_len = prefix_mask.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"prefix_mask sequence length cannot exceed max_seq_len={self.config.max_seq_len}"
            )
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        causal = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=prefix_mask.device)
        ).view(1, 1, seq_len, seq_len)
        prefix = prefix_mask.view(-1, 1, 1, seq_len)
        cannot_attend = ~torch.logical_or(causal, prefix.bool())
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        return attn_bias

    def _apply_sequence_id(
        self, attn_bias: torch.Tensor, sequence_id: torch.LongTensor
    ) -> torch.Tensor:
        seq_len = sequence_id.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence_id sequence length cannot exceed max_seq_len={self.config.max_seq_len}"
            )
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        cannot_attend = torch.logical_not(
            torch.eq(sequence_id.view(-1, seq_len, 1), sequence_id.view(-1, 1, seq_len))
        ).unsqueeze(1)
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        return attn_bias

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            hidden_states = block(
                position_ids,
                hidden_states,
                kv_caches[i],
                input_metadata,
            )
        hidden_states = self.norm_f(hidden_states)
        return hidden_states

    def param_init_fn(self, module: nn.Module) -> None:
        init_fn_name = self.config.init_config["name"]
        MODEL_INIT_REGISTRY[init_fn_name](
            module=module,
            n_layers=self.config.n_layers,
            d_model=self.config.d_model,
            **self.config.init_config,
        )

    def fsdp_wrap_fn(self, module: nn.Module) -> bool:
        return isinstance(module, MPTBlock)

    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        return isinstance(module, MPTBlock)


class MPTForCausalLM(MPTPreTrainedModel):
    def __init__(self, config: MPTConfig):
        super().__init__(config,linear_method: Optional[LinearMethodBase] = None,)
        if not config.tie_word_embeddings:
            raise ValueError("MPTForCausalLM only supports tied word embeddings")
        log.info(f"Instantiating an MPTForCausalLM model from {__file__}")
        self.linear_method = linear_method

        self.transformer: MPTModel = MPTModel(config,linear_method)
        for child in self.transformer.children():
            if isinstance(child, torch.nn.ModuleList):
                continue
            if isinstance(child, torch.nn.Module):
                child._fsdp_wrap = True
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == "inv_sqrt_d_model":
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(
                        f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
                    )
            self.logit_scale = logit_scale

        self.sampler = Sampler(config.vocab_size)

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head_weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(self, value: Union[SharedEmbedding, nn.Embedding]) -> None:
        self.transformer.wte = value

    def get_output_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_output_embeddings(
        self, new_embeddings: Union[SharedEmbedding, nn.Embedding]
    ) -> None:
        self.transformer.wte = new_embeddings

    def set_decoder(self, decoder: MPTModel) -> None:
        self.transformer = decoder

    def get_decoder(self) -> MPTModel:
        return self.transformer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         input_metadata)
        return hidden_states

    def param_init_fn(self, module: nn.Module) -> None:
        init_fn_name = self.config.init_config["name"]
        MODEL_INIT_REGISTRY[init_fn_name](
            module=module,
            n_layers=self.config.n_layers,
            d_model=self.config.d_model,
            **self.config.init_config,
        )

    def fsdp_wrap_fn(self, module: nn.Module) -> bool:
        return isinstance(module, MPTBlock)

    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        return isinstance(module, MPTBlock)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds is not implemented for MPT yet")
        attention_mask = kwargs["attention_mask"].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError(
                "MPT does not support generation with right padding."
            )
        if self.transformer.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if self.transformer.prefix_lm:
            prefix_mask = torch.ones_like(attention_mask)
            if kwargs.get("use_cache") == False:
                raise NotImplementedError(
                    "MPT with prefix_lm=True does not support use_cache=False."
                )
        else:
            prefix_mask = None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prefix_mask": prefix_mask,
            "sequence_id": sequence_id,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    @staticmethod
    def _reorder_cache(
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        beam_idx: torch.LongTensor,
    ) -> List[Tuple[torch.Tensor, ...]]:
        """Used by HuggingFace generate when using beam search with kv-caching.
        See https://github.com/huggingface/transformers/blob/3ec7a47664ebe40c40f4b722f6bb1cd30c3821ec/src/transformers/models/gpt2/modeling_gpt2.py#L1122-L1133
        for an example in transformers.
        """
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past += [
                tuple(
                    (past_state.index_select(0, beam_idx) for past_state in layer_past)
                )
            ]
        return reordered_past
