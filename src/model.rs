use tch::{
    nn::{self, embedding, linear, seq, Embedding, Linear, Module, Sequential},
    Device, IndexOp, Kind, Tensor,
};
use tiktoken_rs::{
    get_bpe_from_tokenizer,
    tokenizer::{self, get_tokenizer},
    CoreBPE,
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
    pub fn new(root: &nn::Path, cfg: Config) -> Self {
        let token_emb = embedding(
            &(root / "token_emb"),
            cfg.vocab_size,
            cfg.emb_dim,
            Default::default(),
        );
        let pos_emb = embedding(
            &(root / "pos_emb"),
            cfg.context_length,
            cfg.emb_dim,
            Default::default(),
        );

        let mut trf_blocks = seq();
        for layer_idx in 0..cfg.num_layers {
            let trf = Transformer::new(&(root / format!("block_{}", layer_idx)), cfg.clone());
            trf_blocks = trf_blocks.add(trf);
        }

        let final_norm = LayerNorm::new(&(root / "final_norm"), cfg.emb_dim);

        let out_head = linear(
            &(root / "head"),
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

    fn get_tokenizer(&self) -> CoreBPE {
        let tokenizer_base = match get_tokenizer("gpt2") {
            Some(tokenizer) => tokenizer,
            None => panic!("Tokenizer not found"),
        };

        match get_bpe_from_tokenizer(tokenizer_base) {
            Ok(tokenizer) => tokenizer,
            Err(e) => panic!("Error getting BPE tokenizer: {}", e),
        }
    }

    pub fn text_to_tensor(&self, start_context: String) -> Tensor {
        let tokenizer = self.get_tokenizer();
        let encoded = tokenizer.encode_with_special_tokens(&start_context);

        let encoded_converted: Vec<i64> = encoded.iter().map(|&x| x as i64).collect();

        Tensor::from_slice(&encoded_converted)
            .to_device(Device::cuda_if_available())
            .unsqueeze(0)
    }

    pub fn logits_to_text(&self, logits: Tensor) -> String {
        let tokenizer = self.get_tokenizer();

        let token_ids: Vec<i32> = logits.try_into().unwrap();

        let converted_token_ids: Vec<u32> =
            token_ids.iter().into_iter().map(|x| *x as u32).collect();

        tokenizer.decode(converted_token_ids).unwrap()
    }

    pub fn generate_text(&self, input: Tensor, max_new_tokens: i64, context_size: i64, temperature: f64, topK: i64) -> Tensor {
        let mut current = input.copy();

        for _ in 0..max_new_tokens {
            let next_token = tch::no_grad(|| {
                let current_cond = current.slice(-1, -context_size, i64::MAX, 1);
                let logits = current_cond.apply(self).i((.., -1, ..));

                let (top_logits, _) = logits.topk(topK, -1, true, true);

                let scaled_logits = top_logits / temperature;

                let probas = scaled_logits.softmax(-1, Kind::Float);

                probas.multinomial(1, true)
            });

            current = Tensor::cat(&[current, next_token], 1)
        }

        current
    }
}

impl Module for Model {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (_, seq_length) = xs.size2().unwrap();
        let token_emb = self.token_emb.forward(xs);
        let pos_emb = self
            .pos_emb
            .forward(&Tensor::arange(seq_length, (Kind::Int64, xs.device())));

        let test = token_emb + pos_emb;

        let test1 = test.dropout(self.drop_rate, false);
        let test2 = test1.apply(&self.trf_blocks);
        let test3 = test2.apply(&self.final_norm);

        test3.apply(&self.out_head)
    }
}
