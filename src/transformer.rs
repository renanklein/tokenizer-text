use reqwest::redirect::Attempt;
use tch::nn::Module;

use crate::{architecture::{FeedForward, LayerNorm}, atention::Attention, config::Config};

pub struct Transformer {
    att: Attention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    drop_rate:i64
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

        Transformer { att, ff, norm1, norm2, cfg.drop_rate }
    }
}

impl Module for Transformer {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        let mut normed1 = self.norm1.forward(xs);
        normed1 = self.att.forward(normed1);
        normed1 = normed1.dropout(, train)

    }
}
