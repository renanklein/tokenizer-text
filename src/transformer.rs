use tch::{nn::Module, Device, Kind, Tensor};

use crate::{architecture::{FeedForward, LayerNorm}, atention::Attention, config::Config};

#[derive(Debug)]
pub struct Transformer {
    att: Attention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    drop_rate:f64
}

impl Transformer {
    pub fn new(cfg: Config) -> Self {
        let att = Attention::new(
            cfg.d_in,
            cfg.d_out,
            cfg.num_heads,
            cfg.context_length,
            cfg.drop_rate
        );

        let ff = FeedForward::new(cfg.emb_dim);
        let norm1 = LayerNorm::new(cfg.emb_dim);
        let norm2 = LayerNorm::new(cfg.emb_dim);

        Transformer { att, ff, norm1, norm2, drop_rate: cfg.drop_rate }
    }
}

impl Module for Transformer {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {

        let result1 = xs + xs.apply(&self.norm1).apply(&self.att).dropout(self.drop_rate, false);

        let result2 = xs.apply(&self.norm2).apply(&self.ff).dropout(self.drop_rate, false);

        result1 + result2
    }
}
