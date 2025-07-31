use tch::{nn::Module, Device, Kind, Tensor};
use tiktoken_rs::{
    get_bpe_from_tokenizer,
    tokenizer::{self, get_tokenizer},
};

use crate::{config::Config, transformer::Transformer};

mod data_modifier;
mod file_utils;
mod simple_tokenizer;
mod atention;
mod architecture;
mod config;
mod transformer;
mod model;
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::manual_seed(123);

    Ok(())
}

async fn get_text_from_file(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let file_content = file_utils::read_file(file_path).await?;

    Ok(file_content)
}

async fn test_tokenizer_chapter2() -> Result<(), Box<dyn std::error::Error>> {
    let text = get_text_from_file("the-verdict.txt").await?;
    let tokenizer_base = match get_tokenizer("gpt2") {
        Some(tokenizer) => tokenizer,
        None => panic!("Tokenizer not found"),
    };

    let tokenizer = match get_bpe_from_tokenizer(tokenizer_base) {
        Ok(tokenizer) => tokenizer,
        Err(e) => panic!("Error getting BPE tokenizer: {}", e),
    };

    let dataset = data_modifier::GPTDatasetV1::new(tokenizer, text, 4, 4);

    let input_tensor = dataset.get_input_tensor_batch(8);
    input_tensor.print();

    let vocab_size: i64 = 50257;
    let output_dim: i64 = 256;
    let var_store = tch::nn::VarStore::new(tch::Device::cuda_if_available());
    let embedding_config = tch::nn::EmbeddingConfig {
        sparse: false,
        scale_grad_by_freq: false,
        ws_init: tch::nn::Init::Randn {
            mean: 0.0,
            stdev: 1.0,
        },
        padding_idx: -1,
    };

    tch::manual_seed(123);

    let embbeding_layer =
        tch::nn::embedding(&var_store.root(), vocab_size, output_dim, embedding_config);

    let pos_embedding_layer =
        tch::nn::embedding(&var_store.root(), 4, output_dim, embedding_config);

    Ok(())
}
