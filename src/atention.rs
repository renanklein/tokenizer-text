use core::f64;
use std::fmt::Debug;

use tch::{
    nn::{self, Module, ModuleT, VarStore}, Device, IndexOp, Kind, Tensor
};

#[derive(Debug)]
pub struct Attention {
    query_linear: nn::Linear,
    key_linear: nn::Linear,
    value_linear: nn::Linear,
    out_proj: nn::Linear,
    mask_init: Tensor,
    dropout_p: f64,
    d_out: i64,
    num_head: i64,
    head_dim: i64
}

impl Attention {
    pub fn new(d_in: i64, d_out: i64, num_head: i64, context_length: i64, dropout_p: f64) -> Self {
        let vs = VarStore::new(Device::cuda_if_available());
        let root = vs.root();

        let query_linear = nn::linear(&root, d_in, d_out, Default::default());
        let key_linear = nn::linear(&root, d_in, d_out, Default::default());
        let value_linear = nn::linear(&root, d_in, d_out, Default::default());
        let out_proj = nn::linear(&root, d_in, d_out, Default::default());
        let head_dim = d_out/num_head as i64;
        let mask_init = Tensor::ones([context_length, context_length], (Kind::Float, Device::cuda_if_available())).tril(1);

        Attention {
            query_linear,
            key_linear,
            value_linear,
            out_proj,
            mask_init,
            dropout_p,
            d_out,
            num_head,
            head_dim
        }
    }

//    pub fn forward(&self, input: &Tensor) -> Tensor {
//        let queries = self.query_linear.forward(input);
//        let keys = self.key_linear.forward(input);
//        let values = self.value_linear.forward(input);
//
//        let attn_scores = queries.matmul(&keys.tr());
//        let d_k = keys.size()[1] as f64;
//        let scale = d_k.sqrt();
//        let scaled_attn_scores = attn_scores / scale;
//        let attn_weights = scaled_attn_scores.softmax(-1, tch::Kind::Float);
//
//        attn_weights
//    }

    pub fn forward(&self, input: Tensor) -> impl ModuleT {
        nn::func_t(move |input, train| {
            let (sz_b, sz_t, sz_c) = input.size3().unwrap();
            let sizes = [sz_b, sz_t, self.num_head, sz_c / self.num_head];
            let k = input.apply(&self.key_linear).view(sizes).transpose(1, 2);
            let q = input.apply(&self.query_linear).view(sizes).transpose(1, 2);
            let v = input.apply(&self.value_linear).view(sizes).transpose(1, 2);
            let mut att_scores = q.matmul(&(&k.transpose(-2, -1) * (1.0 / f64::sqrt(sizes[3] as f64))));
            att_scores = att_scores.masked_fill(&self.mask_init.i((.., .., ..sz_t, ..sz_t)).eq(0.), f64::NEG_INFINITY);
            let att_weights = att_scores.softmax(-1, Kind::Float).dropout(self.dropout_p, train);
            let context_vector = att_weights.matmul(&v).transpose(1, 2).contiguous().view([sz_b, sz_t, sz_c]);
            context_vector.apply(&self.out_proj)
        })
    }

}
