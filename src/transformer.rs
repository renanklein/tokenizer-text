use crate::{architecture::{FeedForward, LayerNorm}, config::Config};

pub struct Transformer {
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm
}

impl Transformer {
    pub fn new(cfg: Config) -> Self {
        

    }
}
