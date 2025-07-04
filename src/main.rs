use tch::Tensor;
use tiktoken_rs::{
    get_bpe_from_tokenizer,
    tokenizer::{self, get_tokenizer},
};

mod data_modifier;
mod file_utils;
mod simple_tokenizer;
mod atention;
mod architecture;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::manual_seed(789);

    let data = vec![
        0.43, 0.15, 0.89, 0.55, 0.87, 0.66, 0.57, 0.85, 0.64, 0.22, 0.58, 0.33, 0.77, 0.25, 0.10,
        0.05, 0.80, 0.55,
    ];

    let shape = [6, 3];

    let tensor = tch::Tensor::from_slice(&data).reshape(shape).to_device(tch::Device::cuda_if_available())
        .to_dtype(tch::Kind::Float, true, false);

    let attention = atention::Attention::new(3, 2);

    let output = attention.forward(&tensor);

    output.print();


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
