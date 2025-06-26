use std::fmt::Debug;

use tch::{
    nn::{self, Module, VarStore},
    Device, Tensor,
};

#[derive(Debug)]
pub struct Attention {
    query_linear: nn::Linear,
    key_linear: nn::Linear,
    value_linear: nn::Linear,
    d_out: i64,
    num_head: i64,
    head_dim: i64
}


impl Module for Attention {
    fn forward(&self, input: &Tensor) -> Tensor {
        let queries = self.query_linear.forward(input);
        let keys = self.key_linear.forward(input);
        let values = self.value_linear.forward(input);

        let attn_scores = queries.matmul(&keys.tr());
        let d_k = keys.size()[1] as f64;
        let scale = d_k.sqrt();
        let scaled_attn_scores = attn_scores / scale;
        let attn_weights = scaled_attn_scores.softmax(-1, tch::Kind::Float);

        attn_weights
    }
}

impl Attention {
    pub fn new(d_in: i64, d_out: i64) -> Self {
        let vs = VarStore::new(Device::cuda_if_available());
        let root = vs.root();

        let query_linear = nn::linear(&root, d_in, d_out, Default::default());
        let key_linear = nn::linear(&root, d_in, d_out, Default::default());
        let value_linear = nn::linear(&root, d_in, d_out, Default::default());

        Attention {
            query_linear,
            key_linear,
            value_linear,
        }
    }
}
