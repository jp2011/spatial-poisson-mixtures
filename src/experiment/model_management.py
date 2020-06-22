def build_lgcp_uid(*, prefix='LGCP-MATERN',
                                 chain_no=None,
                                 c_type='burglary',
                                 t_period='12015-122015',
                                 cov_interpolation='weighted',
                                 model_spec=None,
                                 resolution=400):
    """Create a unique identifier for a model with context specified through the parameters"""
    if chain_no:
        return f"{prefix}-CHAIN-{chain_no}--{c_type}--{model_spec}--{t_period}--{cov_interpolation}--{resolution}"
    else:
        return f"{prefix}--{c_type}--{model_spec}--{t_period}--{cov_interpolation}--{resolution}"


def build_block_mixture_flat_uid(*, prefix='BLOCK-MIXTURE-FLAT',
                                 chain_no=None,
                                 block_scheme='msoa',
                                 c_type='burglary',
                                 t_period='12015-122015',
                                 model_spec=None,
                                 cov_interpolation='weighted',
                                 resolution=400,
                                 K=None):
    """Create a unique identifier for a model with context specified through the parameters"""
    if chain_no:
        return f"{prefix}-CHAIN-{chain_no}--{block_scheme}--{c_type}--{t_period}--{model_spec}--{cov_interpolation}--{resolution}--{K}"
    else:
        return f"{prefix}--{block_scheme}--{c_type}--{t_period}--{model_spec}--{cov_interpolation}--{resolution}--{K}"


def build_block_mixture_gp_softmax_uid(*, prefix='BLOCK-MIXTURE-GP',
                                       chain_no=None,
                                       block_scheme='msoa',
                                       c_type='burglary',
                                       t_period='12015-122015',
                                       model_spec=None,
                                       resolution=400,
                                       lengthscale=None,
                                       K=None):
    """Create a unique identifier for a model with context specified through the parameters"""
    if chain_no:
        return f"{prefix}-CHAIN-{chain_no}--{block_scheme}--{c_type}--{t_period}--{model_spec}--weighted--{resolution}--{t_period}--{K}--{lengthscale}--0_0"
    else:
        return f"{prefix}--{block_scheme}--{c_type}--{t_period}--{model_spec}--weighted--{resolution}--{t_period}--{K}--{lengthscale}--0_0"
