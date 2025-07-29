use tch::{nn::{embedding, Embedding, EmbeddingConfig, LayerNorm, Linear, VarStore}, Device};

use crate::{config::Config, transformer::Transformer};

pub struct Model {
    token_emb: Embedding,
    pos_emb: Embedding,
    trf_block: Vec<Transformer>,
    final_norm: LayerNorm,
    out_head: Linear
}


impl Model {
    pub fn new(cfg: Config) -> Self {
        let vs = VarStore::new(Device::cuda_if_available());
        let root = vs.root();
    }
}
