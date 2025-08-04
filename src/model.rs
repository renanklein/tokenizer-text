use tch::{
    nn::{
        embedding, linear, seq, Embedding, EmbeddingConfig, Linear, Module, Sequential, VarStore,
    },
    Device, IndexOp, Kind, Shape, Tensor,
};

use crate::{architecture::LayerNorm, config::Config, transformer::Transformer};

#[derive(Debug)]
pub struct Model {
    token_emb: Embedding,
    pos_emb: Embedding,
    trf_blocks: Sequential,
    final_norm: LayerNorm,
    out_head: Linear,
    drop_rate: f64,
}

impl Model {
    pub fn new(cfg: Config) -> Self {
        let vs = VarStore::new(Device::cuda_if_available());
        let root = vs.root();

        let token_emb = embedding(
            &root / "token_emb",
            cfg.vocab_size,
            cfg.emb_dim,
            Default::default(),
        );
        let pos_emb = embedding(
            &root / "pos_emb",
            cfg.context_length,
            cfg.emb_dim,
            Default::default(),
        );

        let mut trf_blocks = seq();

        for _ in 0..cfg.num_layers {
            let trf = Transformer::new(cfg.clone());
            trf_blocks = trf_blocks.add(trf);
        }

        let final_norm = LayerNorm::new(cfg.emb_dim);

        let out_head = linear(
            &root / "head",
            cfg.emb_dim,
            cfg.vocab_size,
            Default::default(),
        );

        Model {
            token_emb,
            pos_emb,
            trf_blocks,
            final_norm,
            out_head,
            drop_rate: cfg.drop_rate,
        }
    }

    pub fn generate_text(self, input: Tensor, max_new_tokens: i64, context_size: i64) -> Tensor {
        let mut current = input.copy();

        let next_token =  tch::no_grad(|| {
            let current_cond = current.slice(-1, -context_size, i64::MAX, 1);
            for _ in 0..max_new_tokens {
                let logits = current_cond .apply(&self).i((0, -1, ..));
                let probas = logits.softmax(-1, Kind::Float);
                probas.argmax(-1, true)
            }
        });

        Tensor::cat(&[current, next_token], 1)
    }
}

impl Module for Model {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (_, seq_length) = xs.size2().unwrap();
        let token_emb = self.token_emb.forward(xs);
        let pos_emb = self
            .pos_emb
            .forward(&Tensor::arange(seq_length, (Kind::Float, xs.device())));

        (token_emb + pos_emb)
            .dropout(self.drop_rate, false)
            .apply(&self.trf_blocks)
            .apply(&self.final_norm)
            .apply(&self.out_head)
    }
}
