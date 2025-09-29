use tch::{nn::{self, Module}, Tensor};

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
    pub fn new(path: &nn::Path, cfg: Config) -> Self {
        let att = Attention::new(&(path / "att"), cfg.d_in, cfg.d_out, cfg.num_heads, cfg.context_length, cfg.drop_rate);
        let ff = FeedForward::new(&(path / "ff"), cfg.emb_dim);
        let norm1 = LayerNorm::new(&(path / "ln1"), cfg.emb_dim);
        let norm2 = LayerNorm::new(&(path / "ln2"), cfg.emb_dim);
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
